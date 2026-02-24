import os
import argparse

import librosa
import torch
import numpy as np
import torch.nn as nn

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader

from utils.tools import pad_1D, pad_2D

from tqdm import tqdm

from dataset import Dataset
from evaluate import evaluate
from model import FastSpeech2Loss
from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import get_configs_of, to_device, log, synth_one_sample

import random
import numpy as np
import torch

torch.backends.cudnn.enabled = False

def set_all_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 연산 재현성 강화 옵션(선택)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train(rank, args, configs, batch_size, num_gpus):
    preprocess_config, model_config, train_config = configs

    set_all_seeds(train_config["seed"])

    if num_gpus > 1:
        init_process_group(
            backend=train_config["dist_config"]["dist_backend"],
            init_method=train_config["dist_config"]["dist_url"],
            world_size=train_config["dist_config"]["world_size"] * num_gpus,
            rank=rank
        )
    device = torch.device('cuda:{:d}'.format(rank))

    args.restore_step = str(args.restore_step)


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
        num_workers=os.cpu_count(),
        sampler=data_sampler,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    if num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True).to(device)
    
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Load vocoder

    vocoder = get_vocoder(model_config, device)

    # Training
    step = int(args.restore_step.split('_')[0]) + 1
    print("Starting training from step {}...".format(step))
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    extractor_only_step = train_config["step"]["extractor_only_step"]

    SAVE_STEPS = {290000, 350000, 400000, 450000, 500000, 600000, 1000000}

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
        outer_bar.n = int(args.restore_step.split('_')[0])
        outer_bar.update()

    train = True
    did_x0_init = False
    # classifier_loss_small = Fals

    model.train()
    optimizer.zero_grad()

    while train:
        # # === NEW: style predictor freeze/unfreeze ===
        # if not did_x0_init:
        #     # freeze predictor
        #     if hasattr(model, "module"):
        #         for p in model.module.style_predictor.parameters():
        #             p.requires_grad = False
        #     else:
        #         for p in model.style_predictor.parameters():
        #             p.requires_grad = False
        # else:
        #     # unfreeze predictor
        #     if hasattr(model, "module"):
        #         for p in model.module.style_predictor.parameters():
        #             p.requires_grad = True
        #     else:
        #         for p in model.style_predictor.parameters():
        #             p.requires_grad = True
        # # ============================================
        fs2 = model.module if hasattr(model, "module") else model

        # Phase1: extractor only => predictor freeze
        if step < extractor_only_step:
            for p in fs2.style_predictor.parameters():
                p.requires_grad = False
        else:
            for p in fs2.style_predictor.parameters():
                p.requires_grad = True

        # # Phase3+: extractor freeze (ref_enc + style_extractor)
        # if step >= extractor_only_step + 50000: # 350000 step 이후 style extractor freeze
        #     for p in fs2.ref_enc.parameters():
        #         p.requires_grad = False
        #     for p in fs2.style_extractor.parameters():
        #         p.requires_grad = False
        # else:
        #     for p in fs2.ref_enc.parameters():
        #         p.requires_grad = True
        #     for p in fs2.style_extractor.parameters():
        #         p.requires_grad = True

        if rank == 0:
            inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        if num_gpus > 1:
            data_sampler.set_epoch(epoch)
        for batchs in loader:
            if not train:
                break
            for batch in batchs:
                batch = to_device(batch, device)
                basenames = batch[0]
                output = model(*(batch[2:]), step=step, inference=False, did_x0_init=did_x0_init) # To do Step
                losses = Loss(batch, output, step=step) # To do Step
                total_loss = losses[0]
                total_loss = total_loss / grad_acc_step
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                if step % grad_acc_step == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.update_learning_rate()
                    optimizer.step()
                    optimizer.zero_grad()

                # if did_x0_init:

                #     if hasattr(model, "module"):
                #         fs2 = model.module
                #     else:
                #         fs2 = model

                #     emotions = batch[3]                      # (B,)
                #     orig_style_ref_embs = output[16]          # 필요 없으면 제거 가능
                #     indices_list = output[15]                 # [(B,1), (B,1), (B,1)]
                #                                             # ⚠️ 실제 output index 맞게 조정

                #     neutral_mask = (emotions == fs2.neutral_id)

                #     if neutral_mask.any():

                #         from collections import Counter

                #         stage_vecs = []

                #         for s in range(3):  # VQ1, VQ2, VQ3
                #             idx_all = indices_list[s][neutral_mask].view(-1)  # (N_neu,)

                #             # batch 내 최빈 index
                #             mode_idx = Counter(idx_all.tolist()).most_common(1)[0][0]

                #             # codebook lookup
                #             vec = fs2.style_extractor.vq_layers[s].embedding.weight[mode_idx]
                #             stage_vecs.append(vec)

                #         # concat → (1, 768)
                #         batch_neu_vec = torch.cat(stage_vecs, dim=0).unsqueeze(0)
                #         batch_neu_vec = batch_neu_vec.to(fs2.neu_base.device, fs2.neu_base.dtype)

                #         # EMA
                #         alpha0 = 3e-2
                #         alpha = alpha0 / (step ** 0.3)

                #         with torch.no_grad():
                #             fs2.neu_base.mul_(1.0 - alpha).add_(alpha * batch_neu_vec)
                        
                if rank == 0:
                    if step % log_step == 0:
                        losses_ = [sum(l.values()).item() if isinstance(l, dict) else l.item() for l in losses]
                        message1 = "Step {}/{}, ".format(step, total_step)
                        message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Style_loss: {:.4f}, Guided_loss: {:.4f}, vq_loss: {:.4f}, cls_loss(indices): {:.4f}, flow_loss: {:.4f}, neu_align_loss: {:.4f}, soft_zero_loss: {:.4f}".format( 
                            ### " 주석 - utils/tools 에도 주석 , evaluate.py에도 주석, tools.py에도 주석
                            *losses_
                        )

                        # if losses[9].item() < 0.3 and step > 200000:
                        #     classifier_loss_small = True

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
                        message, cls_loss_val = evaluate(device, model, step, configs, val_logger, vocoder, losses)

                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(message + "\n")
                        outer_bar.write(message)

                        model.train()

                    # if step % save_step == 0:
                    if step in SAVE_STEPS:
                        torch.save(
                            {
                                "model": model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                            },
                            os.path.join(
                                train_config["path"]["ckpt_path"],
                                "{}_fs2-flow.pth.tar".format(step),
                            ),
                        )
                        print("Save checkpoint at step {}_fs2-flow.pth.tar".format(step))

                if step == total_step:
                    train = False
                    break
                step += 1
                if rank == 0:
                    outer_bar.update(1)

            if rank == 0:
                inner_bar.update(1)
        
        if epoch == 1:
            print("[INIT] Warm-up finished. Running K-means initialization ...")

            with torch.no_grad():
                # --- 1. 전체 train 데이터셋에서 ref_emb / style vector 수집 ---               
                dataset_full = Dataset(
                    "train.txt", preprocess_config, train_config, sort=False, drop_last=False
                )
                loader_full = DataLoader(
                    dataset_full,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=8,
                    collate_fn=dataset_full.collate_fn,
                )   

                ref_embs_all, styles_all, emotions_all = [], [], []

                for batchs in tqdm(loader_full, desc="[INIT] Extracting ref_embs for K-means"):
                    for batch in batchs:
                        batch = to_device(batch, device)
                        
                        mel = batch[7]
                        emotions = batch[3]

                        # === FAST PATH: ref_enc + RVQ only ===
                        ref_emb, cls_loss = model.ref_enc(mel, emotions)
                        style, vq_loss, min_idx, codebooks = model.style_extractor(ref_emb, cls_loss)

                        ref_embs_all.append(ref_emb)
                        styles_all.append(style)
                        emotions_all.append(emotions)

                ref_embs_all = torch.cat(ref_embs_all, dim=0)
                styles_all = torch.cat(styles_all, dim=0)
                emotions_all = torch.cat(emotions_all, dim=0)
                torch.cuda.empty_cache()

                # --- 2. RVQ K-means initialization ---
                print("[INIT] Performing K-means initialization for RVQ codebooks...")
                fs2 = model.module if hasattr(model, "module") else model
                # ref_embs_all: [N, 768] = [N, 256*3]
                
                # RVQ 3단계 초기화
                fs2.style_extractor.vq_layers[0].init_codebook_kmeans(ref_embs_all)
                fs2.style_extractor.vq_layers[1].init_codebook_kmeans(ref_embs_all - styles_all[:, :256])
                fs2.style_extractor.vq_layers[2].init_codebook_kmeans(ref_embs_all - styles_all[:, :256] - styles_all[:, 256:512])


        # dead code update
        if epoch > 1 and step < extractor_only_step:
            # val_path =  '/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/preprocessed_data/esd/train.txt'
            val_path = preprocess_config["path"]["preprocessed_path"] + "/train.txt"

            with open(val_path, encoding='utf-8') as f:
                val_infos = [line.strip().split("|") for line in f]

            import json

            with open(preprocess_config["path"]["preprocessed_path"] + "/emotions.json") as f:
                emotion_map = json.load(f)
                n_emotions = len(emotion_map)
                print("Number of emotions:", n_emotions)

            val_basenames = []
            emotions = []
            styles = []
            ref_embs = []

            for i in range(len(val_infos)):
                if i % 25 != 0:
                    continue
                val_info = val_infos[i]

                basename = val_info[0]
                speaker  = val_info[1]   # ★ 이걸 써야 함
                emotion  = val_info[2]

                val_basenames.append((basename, speaker))
                emotions.append(emotion_map[emotion])
                        
            for i in range(len(val_basenames)):
                val_basename, speaker = val_basenames[i]
                emotion = torch.tensor(emotions[i], device=device).unsqueeze(0)

                mel_path = os.path.join(
                    preprocess_config["path"]["preprocessed_path"],
                    "mel",
                    f"{speaker}-mel-{val_basename}.npy"
                )

                mel = np.load(mel_path)
                mel = torch.from_numpy(mel).float().to(device).unsqueeze(0)

                ref_emb, cls_loss = model.ref_enc(mel, emotion)
                style, _, _, codebooks = model.style_extractor(ref_emb, cls_loss)

                ref_embs.append(ref_emb)
                styles.append(style)

            ref_embs = torch.cat(ref_embs, dim=0)
            styles = torch.cat(styles, dim=0)
            
            torch.cuda.empty_cache()


            if model.style_extractor.vq_layers[0].dead_codes_count() < (n_emotions/2):
                model.style_extractor.vq_layers[0].greedy_restart()
            else:
                model.style_extractor.vq_layers[0].reset_dead_codes_kmeans(ref_embs)
            if model.style_extractor.vq_layers[1].dead_codes_count() < (n_emotions/2):
                model.style_extractor.vq_layers[1].greedy_restart()
            else:
                model.style_extractor.vq_layers[1].reset_dead_codes_kmeans(ref_embs - styles[:, :256])
            if model.style_extractor.vq_layers[2].dead_codes_count() < (n_emotions/2):
                model.style_extractor.vq_layers[2].greedy_restart()
            else:
                model.style_extractor.vq_layers[2].reset_dead_codes_kmeans(ref_embs - styles[:, :256] - styles[:, 256:512])

        # if not did_x0_init:
        if step >= extractor_only_step and not did_x0_init:
            with torch.no_grad():
                fs2 = model.module if hasattr(model, "module") else model

                # === 1. 전체 데이터를 돌며 neutral index를 stage별로 수집 ===
                dataset_full = Dataset("train.txt", preprocess_config, train_config,
                                    sort=False, drop_last=False)
                loader_full = DataLoader(dataset_full, batch_size=batch_size,
                                        shuffle=False, num_workers=os.cpu_count(),
                                        collate_fn=dataset_full.collate_fn)

                neutral_indices_stage = [[], [], []]   # stage1, stage2, stage3

                for batchs in tqdm(loader_full, desc="[INIT] Collect neutral code indices"):
                    for batch in batchs:
                        # ... inside: for batch in batchs:
                        batch = to_device(batch, device)
                        mel = batch[7]
                        emotions = batch[3]

                        # ✅ emotions를 무조건 Tensor로 만들기
                        if not torch.is_tensor(emotions):
                            emotions = torch.LongTensor(emotions)  # list/np/int 모두 대응
                        emotions = emotions.to(device)

                        # (선택) shape 보정: scalar면 (1,)로
                        if emotions.dim() == 0:
                            emotions = emotions.view(1)

                        ref_emb, cls_loss = fs2.ref_enc(mel, emotions)
                        _, _, indices_list, codebooks_list = fs2.style_extractor(ref_emb, cls_loss)

                        neu_mask = emotions.eq(fs2.neutral_id)  # Tensor mask
                        if torch.any(neu_mask):
                            for s in range(3):
                                idx_tensor = indices_list[s][neu_mask]  # (k,1)
                                neutral_indices_stage[s].extend(idx_tensor.view(-1).tolist())

                # === 2. stage별 most frequent index 구하기 ===
                from collections import Counter
                stage_mode_idx = []
                for s in range(3):
                    if len(neutral_indices_stage[s]) == 0:
                        print(f"[INIT][WARN] stage {s+1}: no neutral codes found")
                        stage_mode_idx.append(0)  # fallback
                    else:
                        c = Counter(neutral_indices_stage[s])
                        stage_mode_idx.append(c.most_common(1)[0][0])

                print("[INIT] mode indices:", stage_mode_idx)

                with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                    f.write(
                        f"[INIT] mode indices: {stage_mode_idx}, step: {step}, epoch: {epoch}\n")

                # === 3. codebook에서 vector 가져와 concat ===
                stage_vecs = []
                for s in range(3):
                    idx = stage_mode_idx[s]
                    vec = fs2.style_extractor.vq_layers[s].embedding.weight[idx]  # (256,)
                    stage_vecs.append(vec)

                neutral_vec = torch.cat(stage_vecs, dim=0).unsqueeze(0)   # (1,768)

                # === 4. neu_base에 복사 ===
                fs2.neu_base.data.copy_(neutral_vec.to(fs2.neu_base.device,
                                                    dtype=fs2.neu_base.dtype))

                print(f"[INIT] x0(neu_base) initialized with mode neutral codebook vector.")
                
            # if classifier_loss_small:
            #     with open(os.path.join(train_log_path, "log.txt"), "a") as f:
            #         f.write(f"[CLASSIFIER LOSS SMALL] did_x0_init = True \n")
                did_x0_init = True

        epoch += 1




        torch.cuda.empty_cache()



if __name__ == "__main__":
    assert torch.cuda.is_available(), 'CPU training is not allowed.'
    parser = argparse.ArgumentParser()
    # parser.add_argument('--use_amp', action='store_true')
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
    # print(' ---> Automatic Mixed Precision:', args.use_amp)
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
