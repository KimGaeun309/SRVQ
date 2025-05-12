import os
import argparse
import re

import librosa
import torch
import numpy as np
import torch.nn as nn
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader
import random

from utils.tools import pad_1D, pad_2D

from tqdm import tqdm

from dataset import Dataset
from evaluate import evaluate
from model import FastSpeech2Loss
from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import get_configs_of, to_device, log, synth_one_sample
import json

torch.backends.cudnn.benchmark = True

def train(rank, args, configs, batch_size, num_gpus):
    preprocess_config, model_config, train_config = configs
    if num_gpus > 1:
        init_process_group(
            backend=train_config["dist_config"]["dist_backend"],
            init_method=train_config["dist_config"]["dist_url"],
            world_size=train_config["dist_config"]["world_size"] * num_gpus,
            rank=rank
        )
    device = torch.device('cuda:{:d}'.format(rank))

    # Get Dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    data_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    group_size = 4 # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        sampler=data_sampler,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    for param in model.style_predictor.parameters():
        param.requires_grad = False
    # for param in model.ref_enc.parameters():
    #     param.requires_grad = False
    # for param in model.style_extractor.parameters():
    #     param.requires_grad = False
    # for param in model.style_extract_fc.parameters():
    #     param.requires_grad = False
    if num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True).to(device)
    scaler = amp.GradScaler(enabled=args.use_amp)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Training
    step = int(re.search(r"(\d+)", args.restore_step).group(1)) + 1
    # step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    if rank == 0:
        print("Number of FastSpeech2 Parameters: {}\n".format(get_param_num(model)))
        # Init Logger
        for p in train_config["path"].values():
            os.makedirs(p, exist_ok=True)
        train_log_path = os.path.join(train_config["path"]["log_path"], "train")
        val_log_path = os.path.join(train_config["path"]["log_path"], "val")
        os.makedirs(train_log_path, exist_ok=True)
        os.makedirs(val_log_path, exist_ok=True)
        train_logger = SummaryWriter(train_log_path)
        val_logger = SummaryWriter(val_log_path)

        outer_bar = tqdm(total=total_step, desc="Training", position=0)
        # outer_bar.n = args.restore_step
        outer_bar.n = int(re.search(r"(\d+)", args.restore_step).group(1))
        outer_bar.update()

    train = True
    init_flag = True
    model.train()
    while train:
        if rank == 0:
            inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        if num_gpus > 1:
            data_sampler.set_epoch(epoch)
        for batchs in loader:
            if train == False:
                break
            for batch in batchs:
                batch = to_device(batch, device)

                
                # 감정 mapping 정보 불러오기
                with open("preprocessed_data/emo_kr_22050/emotions.json", "r") as f:
                    emotion_map = json.load(f)
                reverse_emo_map = {v: k for k, v in emotion_map.items()}
                emotion_list = list(emotion_map.keys())  # ['neu', 'ang', ..., 'hap']

                # emotion blending vector 생성
                style_vectors = []
                blended_labels = []
                for _ in batch[0]:  # for each utterance
                    emo_a, emo_b = random.sample(emotion_list, 2)
                    alpha = random.uniform(0.0, 1.0)

                    vec_a = np.load(f"emotion_style_vectors_mode/{emo_a}_style.npy")
                    vec_b = np.load(f"emotion_style_vectors_mode/{emo_b}_style.npy")
                    blended_vec = alpha * vec_a + (1 - alpha) * vec_b
                    blended_vec = torch.from_numpy(blended_vec).float().to(device)
                    style_vectors.append(blended_vec)

                    # soft label 생성
                    label_vec = torch.zeros(len(emotion_list)).to(device)
                    idx_a = emotion_map[emo_a]
                    idx_b = emotion_map[emo_b]
                    if idx_a == idx_b:
                        label_vec[idx_a] = 1.0
                    else:
                        label_vec[idx_a] = alpha
                        label_vec[idx_b] = 1 - alpha

                    # 안정화 및 정규화
                    # label_vec += 1e-8
                    # label_vec = label_vec / label_vec.sum()
                    blended_labels.append(label_vec)

                # 텐서화
                style_vector = torch.stack(style_vectors, dim=0)
                blended_label = torch.stack(blended_labels, dim=0)

                # print("blended_label", blended_label)

                
                basenames = batch[0]
                
                pitch_mel, energy_mel = [], []

                # for basename in basenames:
                    # pitch_path = f"/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/normalized_data/pitch_only/{basename}_pitch.npy"
                    # energy_path = f"/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/normalized_data/energy_only/{basename}_energy.npy"
                    # pitch_mel.append(np.load(pitch_path).T)
                    # energy_mel.append(np.load(energy_path).T)
                
                # pitch_mel = pad_2D(pitch_mel)
                # pitch_mel = torch.from_numpy(pitch_mel).to('cuda')
                # energy_mel = pad_2D(energy_mel)
                # energy_mel = torch.from_numpy(energy_mel).to('cuda')
                

                with amp.autocast(args.use_amp):
                    # Forward
                    output = model(*(batch[2:]), step=step, inference=False, pitch_mel=pitch_mel, energy_mel=energy_mel,  init_flag=init_flag, style_vector=style_vector, blended_label=blended_label) # To do Step
                    init_flag = False

                    # Cal Loss
                    losses = Loss(batch, output, step=step) # To do Step
                    total_loss = losses[0]
                    total_loss = total_loss / grad_acc_step

                # Backward
                scaler.scale(total_loss).backward()

                # Clipping gradients to avoid gradient explosion
                if step % grad_acc_step == 0:
                    scaler.unscale_(optimizer._optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                # Update weights
                optimizer.step_and_update_lr(scaler)
                scaler.update()
                optimizer.zero_grad()

                if rank == 0:
                    if step % log_step == 0:
                        losses_ = [sum(l.values()).item() if isinstance(l, dict) else l.item() for l in losses]
                        message1 = "Step {}/{}, ".format(step, total_step)
                        message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Style_loss: {:.4f}, Guided_loss: {:.4f}, vq_loss: {:.4f}, cls_loss(indices): {:.4f}, recon_loss: {:.4f}".format( 
                            ### " 주석 - utils/tools 에도 주석 , evaluate.py에도 주석, tools.py에도 주석
                            *losses_
                        )

                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(message1 + message2 + "\n")

                        outer_bar.write(message1 + message2)

                        log(train_logger, step, losses=losses)

                    if step % synth_step == 0:
                        model.eval()
                        fig, wav_reconstruction, wav_prediction, tag, style_attn = synth_one_sample(
                            batch,
                            model,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )
                        log(
                            train_logger,
                            fig=fig,
                            tag="Training/step_{}_{}".format(step, tag),
                        )
                        sampling_rate = preprocess_config["preprocessing"]["audio"][
                            "sampling_rate"
                        ]
                        log(
                            train_logger,
                            audio=wav_reconstruction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_reconstructed".format(step, tag),
                        )
                        log(
                            train_logger,
                            audio=wav_prediction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_synthesized".format(step, tag),
                        )

                    if step % val_step == 0:
                        torch.cuda.empty_cache()
                        model.eval()
                        message = evaluate(device, model, step, configs, val_logger, vocoder, losses)
                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(message + "\n")
                        outer_bar.write(message)

                        model.train()
                        
                        if losses[9].mean() > 0.4:
                            init_flag = True   

                        # if epoch < 5:
                        #     init_flag = True

                        # if epoch < 5:
                        #     model.style_extractor.vq_layer1.random_restart()
                        #     model.style_extractor.vq_layer2.random_restart()
                        #     model.style_extractor.vq_layer3.random_restart()
                        
                        # model.style_extractor.vq_layer1.reset_dead_codes_kmeans()
                        # model.style_extractor.vq_layer2.reset_dead_codes_kmeans()
                        # model.style_extractor.vq_layer3.reset_dead_codes_kmeans()
                        # model.style_extractor.vq_layer1.reset_usage()
                        # model.style_extractor.vq_layer2.reset_usage()
                        # model.style_extractor.vq_layer3.reset_usage()
                        

                    if step % save_step == 0:
                        torch.save(
                            {
                                "model": model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                            },
                            os.path.join(
                                train_config["path"]["ckpt_path"],
                                "{}.pth.tar".format(step),
                            ),
                        )

                        os.system(f"python3 check_code_index.py --dataset icassp_2024 --restore_step {step} --dataset {args.dataset}")


                if step == total_step:
                    train = False
                    break
                step += 1
                if rank == 0:
                    outer_bar.update(1)

            if rank == 0:
                inner_bar.update(1)
        epoch += 1

        val_path =  '/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/preprocessed_data/emo_kr_22050/train.txt'

        with open(val_path, encoding='utf-8') as f:
            val_infos = [line.strip().split("|") for line in f]

        with open("preprocessed_data/emo_kr_22050/emotions.json") as f:
            emotion_map = json.load(f)

        val_basenames = []
        emotions = []
        styles = []
        ref_embs = []

        for i in range(len(val_infos)):
            if i % 25 != 0: continue
            val_info = val_infos[i]
            val_basenames.append(val_info[0])
            emotions.append(emotion_map[val_info[2]])
        
        for i in range(len(val_basenames)):
            val_basename = val_basenames[i]
            emotion = torch.tensor(emotions[i], device=device).unsqueeze(0)
            mel = np.load(f'preprocessed_data/emo_kr_22050/mel/{val_basename[:3]}-mel-{val_basename}.npy')
            mel = torch.from_numpy(mel).float().to(device)
            mel = mel.unsqueeze(0)

            # pitch_path = f"/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/normalized_data/pitch_only/{val_basename}_pitch.npy"
            # energy_path = f"/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/normalized_data/energy_only/{val_basename}_energy.npy"
            
            # pitch_mel = torch.from_numpy(np.load(pitch_path).T).to(device).unsqueeze(0)
            # energy_mel = torch.from_numpy(np.load(energy_path).T).to(device).unsqueeze(0)
            
            # pitch_mel = pad_2D(pitch_mel)
            # pitch_mel = torch.from_numpy(pitch_mel).to('cpu')
            # energy_mel = pad_2D(energy_mel)print
            # energy_mel = torch.from_numpy(energy_mel).to('cpu')

            ref_emb, cls_loss = model.ref_enc(mel, emotion)
            style, _, _, codebooks = model.style_extractor(ref_emb, cls_loss)

            ref_embs.append(ref_emb)
            styles.append(style)

        ref_embs = torch.cat(ref_embs, dim=0)
        styles = torch.cat(styles, dim=0)
        
        torch.cuda.empty_cache()

        # if model.style_extractor.vq_layers[0].dead_codes_count() < (7/2):
        #     model.style_extractor.vq_layers[0].greedy_restart()
        # else:
        #     model.style_extractor.vq_layers[0].reset_dead_codes_kmeans(ref_embs)
        
        # if model.style_extractor.vq_layers[1].dead_codes_count() < (7/2):
        #     model.style_extractor.vq_layers[1].greedy_restart()
        # else:
        #     model.style_extractor.vq_layers[1].reset_dead_codes_kmeans(ref_embs - styles[:, :256])
        
        # if model.style_extractor.vq_layers[2].dead_codes_count() < (7/2):
        #     model.style_extractor.vq_layers[2].greedy_restart()
        # else:
        #     model.style_extractor.vq_layers[2].reset_dead_codes_kmeans(ref_embs - styles[:, :256] - styles[:, 256:512])

        torch.cuda.empty_cache()

        # model.style_extractor.RVQ1.vq_layers[0].reset_dead_codes_kmeans(z_mels)
        # model.style_extractor.RVQ2.vq_layers[0].reset_dead_codes_kmeans(z_pitchs)
        # model.style_extractor.RVQ3.vq_layers[0].reset_dead_codes_kmeans(z_energies)

        # model.style_extractor.RVQ1.vq_layers[1].reset_dead_codes_kmeans(z_mels - styles[:, :128])
        # model.style_extractor.RVQ2.vq_layers[1].reset_dead_codes_kmeans(z_pitchs - styles[:, 256:384])
        # model.style_extractor.RVQ3.vq_layers[1].reset_dead_codes_kmeans(z_energies - styles[:, 512:640])
        
        
        



if __name__ == "__main__":
    assert torch.cuda.is_available(), 'CPU training is not allowed.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--restore_step', type=str, default=0)
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Name of dataset'
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)

    # Set Device
    torch.manual_seed(train_config["seed"])
    torch.cuda.manual_seed(train_config["seed"])
    num_gpus = torch.cuda.device_count()
    batch_size = int(train_config["optimizer"]["batch_size"] / num_gpus)

    # Log Configuration
    print("\n==================================== Training Configuration ====================================")
    print(' ---> Automatic Mixed Precision:', args.use_amp)
    print(' ---> Number of used GPU:', num_gpus)
    print(' ---> Batch size per GPU:', batch_size)
    print(' ---> Batch size in total:', batch_size * num_gpus)
    print(" ---> Type of Building Block:", model_config["block_type"])
    print("=================================================================================================")
    print("Prepare training ...")

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(args, configs, batch_size, num_gpus))
    else:
        train(0, args, configs, batch_size, num_gpus)
