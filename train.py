import os
import argparse

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
        "train_ra.txt", preprocess_config, train_config, sort=True, drop_last=True
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
    # scaler = amp.GradScaler(enabled=args.use_amp)
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

    SAVE_STEPS = {350000, 400000, 450000, 500000, 600000, 1000000}

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

    model.train()
    optimizer.zero_grad()
    while train:
        if rank == 0:
            inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        if num_gpus > 1:
            data_sampler.set_epoch(epoch)
        for batchs in loader:
            if not train:
                break
            for batch in batchs:
                batch = to_device(batch, device)

                output = model(*(batch[2:]), inference=False) # To do Step
                losses = Loss(batch, output) # To do Step
                total_loss = losses[0]
                total_loss = total_loss / grad_acc_step
                total_loss.backward()

                if step % grad_acc_step == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)                    
                    optimizer.update_learning_rate()
                    optimizer.step()
                    optimizer.zero_grad()
            

                if rank == 0:
                    if step % log_step == 0:
                        losses_ = [sum(l.values()).item() if isinstance(l, dict) else l.item() for l in losses]
                        message1 = "Step {}/{}, ".format(step, total_step)
                        message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format( 
                            ### " 주석 - utils/tools 에도 주석 , evaluate.py에도 주석, tools.py에도 주석
                            *losses_
                        )

                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(message1 + message2 + "\n")

                        outer_bar.write(message1 + message2)

                        log(train_logger, step, losses=losses)

                    if step % synth_step == 0:
                        model.eval()
                        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
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
                        message = evaluate(device, model, step, configs, val_logger, vocoder)

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
                                "{}_se.pth.tar".format(step),
                            ),
                        )
                        print("Save checkpoint at step {}_se.pth.tar".format(step))

                if step == total_step:
                    train = False
                    break
                step += 1
                if rank == 0:
                    outer_bar.update(1)

            if rank == 0:
                inner_bar.update(1)

        epoch += 1




        torch.cuda.empty_cache()



if __name__ == "__main__":
    assert torch.cuda.is_available(), 'CPU training is not allowed.'
    parser = argparse.ArgumentParser()
    # parser.add_argument('--use_amp', action='store_true', default=False)
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