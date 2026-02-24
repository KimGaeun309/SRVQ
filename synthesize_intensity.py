#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import torch

from utils.model import get_model, get_vocoder
from speechbrain.inference.vocoders import HIFIGAN


############################################################
# -------------------- UTIL -------------------------------
############################################################

def parse_line(line):
    parts = line.strip().split("|")
    if len(parts) < 5:
        return None
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def make_intensity_list(start, step):
    vals = []
    cur = start
    while cur <= 1.0 + 1e-8:
        vals.append(round(cur, 10))
        cur += step
    return vals


def chunk_list(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i + bs]


def phones_to_sequence(phones, cleaners):
    from text import text_to_sequence
    return np.array(text_to_sequence(phones, cleaners), dtype=np.int64)


def collate_batch(batch_items, spk_map, emo_map, cleaners):

    ids, raws, spks, emos, texts, lens = [], [], [], [], [], []

    for utt_id, spk, emo, phones, raw in batch_items:

        ids.append(utt_id)
        raws.append(raw)

        spks.append(spk_map[spk])
        emos.append(emo_map[emo])

        seq = phones_to_sequence(phones, cleaners)
        texts.append(seq)
        lens.append(len(seq))

    max_len = max(lens)

    padded = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        padded[i, :len(t)] = t

    return (
        ids,
        raws,
        np.array(spks),
        np.array(emos),
        padded,
        np.array(lens),
        max_len,
    )


############################################################
# -------------------- MAIN -------------------------------
############################################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--restore_step", required=True)

    parser.add_argument("--source", required=True)

    parser.add_argument("--neutral_name", default="neu")
    parser.add_argument("--only_emotion", default=None)

    parser.add_argument("--intensity_start", type=float, default=0.1)
    parser.add_argument("--intensity_step", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)

    args = parser.parse_args()

    ########################################################
    # dataset 분기 (synthesize.py 방식 그대로)
    ########################################################
    if args.dataset.lower() == "esd":
        print("Using tools_16k (SpeechBrain vocoder)")
        from utils.tools_16k import get_configs_of, to_device, synth_samples
    else:
        print("Using default tools")
        from utils.tools import get_configs_of, to_device, synth_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[TTS] device:", device)

    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)

    ########################################################
    # model
    ########################################################
    model = get_model(args, configs, device, train=False)
    model.eval()

    ########################################################
    # vocoder 분기 (핵심)
    ########################################################
    if args.dataset.lower() == "esd":
        print("Using SpeechBrain 16kHz HiFi-GAN")

        vocoder = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz",
            savedir="pretrained_models/tts-hifigan-libritts-16kHz",
            run_opts={"device": str(device)},
        )
        vocoder = vocoder.to(device)
        vocoder.eval()

    else:
        vocoder = get_vocoder(model_config, device)

    ########################################################
    # maps
    ########################################################
    with open(os.path.join(
        preprocess_config["path"]["preprocessed_path"],
        "speakers.json")) as f:
        spk_map = json.load(f)

    with open(os.path.join(
        preprocess_config["path"]["preprocessed_path"],
        "emotions.json")) as f:
        emo_map = json.load(f)

    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

    ########################################################
    # load source
    ########################################################
    items = []

    with open(args.source, encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue

            utt, spk, emo, phones, raw = parsed

            if emo == args.neutral_name:
                continue

            if args.only_emotion and emo != args.only_emotion:
                continue

            items.append((utt, spk, emo, phones, raw))

    if len(items) == 0:
        raise RuntimeError("No items.")

    base_result_path = os.path.join(
        train_config["path"]["result_path"],
        str(args.restore_step),
        "intensity",
    )

    os.makedirs(base_result_path, exist_ok=True)
    print("[INFO] result root:", base_result_path)

    intensity_list = make_intensity_list(
        args.intensity_start,
        args.intensity_step
    )

    print("[INFO] intensity grid:", intensity_list)

    ########################################################
    # group by emotion
    ########################################################
    emo_groups = {}
    for it in items:
        emo_groups.setdefault(it[2], []).append(it)

    ########################################################
    # synth loop
    ########################################################
    for emo, emo_items in emo_groups.items():

        print(f"\n==== Emotion {emo} ({len(emo_items)}) ====")

        for intensity in intensity_list:
            out_dir = os.path.join(
                base_result_path,
                f"{emo}_{intensity:.1f}"
            )
            os.makedirs(out_dir, exist_ok=True)

            for mini in chunk_list(emo_items, args.batch_size):

                batch = collate_batch(
                    mini, spk_map, emo_map, cleaners
                )

                batch = to_device(batch, device)

                with torch.no_grad():

                    output = model(
                        *(batch[2:]),
                        p_control=args.pitch_control,
                        e_control=args.energy_control,
                        d_control=args.duration_control,
                        inference=True,
                        intensity=float(intensity),
                    )

                synth_samples(
                    batch,
                    output,
                    vocoder,
                    model_config,
                    preprocess_config,
                    out_dir,
                    args,
                )

            print("[SAVE]", out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()