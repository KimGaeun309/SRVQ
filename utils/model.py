import os
import re
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    print("device", device)

    model = FastSpeech2(preprocess_config, model_config).to(device)
    frozen_codebooks = []  # ✅ 여기서 초기화

    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=torch.device(device), weights_only=False)
        
        # for i in range(3):
        #     old_weight = ckpt["model"][f"style_extractor.vq_layers.{i}.embedding.weight"]
        #     model.style_extractor.vq_layers[i].embedding.weight.data[:7] = old_weight

        #     # freeze용 codebook 저장
        #     frozen_codebooks.append(model.style_extractor.vq_layers[i].embedding.weight[:7])

        for i, vq in enumerate(model.style_extractor.vq_layers):
            key = f"style_extractor.vq_layers.{i}.embedding.weight"
            if key in ckpt["model"]:
                old_weight = ckpt["model"][key]  # shape [7, e_dim]
                if old_weight.shape[0] > 7:
                    frozen_codebooks.append(old_weight[:7].clone().to(device))
                else:
                    print("else")
                    n_e = 64
                    e_dim = old_weight.shape[1]

                    uniform_range = 1.0 / n_e
                    new_weight = torch.empty(n_e, e_dim).uniform_(-uniform_range, uniform_range).to(old_weight.device)
                    new_weight[:7] = old_weight  # 상위 7개는 유지
                    ckpt["model"][key] = new_weight.to(device)

                    frozen_codebooks.append(old_weight.clone().to(device))  # ✅ 7개 고정값 저장

    model.load_state_dict(ckpt["model"])

    if train:
        for param in model.ref_enc.parameters():
            param.requires_grad = False
        for param in model.style_extractor.parameters():
            param.requires_grad = False
        for param in model.style_extract_fc.parameters():
            param.requires_grad = False
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim #, frozen_codebooks  # ✅ frozen_codebooks도 반환

    model.eval()
    model.requires_grad_ = False
    return model



def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=device)
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar", map_location=device)
        elif speaker == "kss":
            ckpt = torch.load("hifigan/generator_kss_16k.pth.tar", map_location=device)
        elif speaker == "icassp_2024":
            ckpt = torch.load("hifigan/g_icassp_2024_50", map_location=device)
        elif speaker == "finetune":
            ckpt = torch.load("hifigan/g_02540000", map_location=device)

        print(f'vocoder: {speaker}')
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs