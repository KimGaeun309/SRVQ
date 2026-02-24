#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import torch

from utils.model import get_model, get_vocoder
from speechbrain.inference.vocoders import HIFIGAN


def parse_line(line):
    parts = line.strip().split("|")
    if len(parts) < 5:
        return None
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def make_intensity_list(start, step):
    vals = []
    cur = float(start)
    while cur <= 1.0 + 1e-8:
        vals.append(round(cur, 10))
        cur += float(step)
    return vals


def chunk_list(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i + bs]


def phones_to_sequence(phones, cleaners):
    from text import text_to_sequence
    return np.array(text_to_sequence(phones, cleaners), dtype=np.int64)


def collate_batch(batch_items, spk_map, emo_map, cleaners):

    ids, raw_texts = [], []
    speakers, emotions = [], []
    texts, src_lens = [], []

    for utt_id, spk, emo, phones, raw_text in batch_items:
        ids.append(utt_id)
        raw_texts.append(raw_text)

        speakers.append(spk_map[spk])
        emotions.append(emo_map[emo])

        seq = phones_to_sequence(phones, cleaners)
        texts.append(seq)
        src_lens.append(len(seq))

    max_src_len = max(src_lens)

    padded = np.zeros((len(texts), max_src_len), dtype=np.int64)
    for i, seq in enumerate(texts):
        padded[i, :len(seq)] = seq

    return (
        ids,
        raw_texts,
        np.array(speakers, dtype=np.int64),
        np.array(emotions, dtype=np.int64),
        padded,
        np.array(src_lens, dtype=np.int64),
        max_src_len,
    )


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

    # --------------------------------------------------
    # dataset별 tools 분기 (synthesize.py 동일)
    # --------------------------------------------------
    if args.dataset.lower() == "esd":
        print("Using tools_16k (SpeechBrain HiFi-GAN)")
        from utils.tools_16k import (
            get_configs_of,
            to_device,
            synth_samples,
        )
    else:
        print("Using default tools")
        from utils.tools import (
            get_configs_of,
            to_device,
            synth_samples,
        )

    # configs
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)

    # output path 자동 생성
    result_root = os.path.join(
        train_config["path"]["result_path"],
        str(args.restore_step),
        "intensity",
    )
    os.makedirs(result_root, exist_ok=True)
    print("[RESULT PATH]", result_root)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TTS] device = {device}")

    # model
    model = get_model(args, configs, device, train=False)
    model.eval()

    # vocoder
    if args.dataset.lower() == "esd":
        print("Using SpeechBrain 16kHz HiFi-GAN for ESD")
        vocoder = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz",
            savedir="pretrained_models/tts-hifigan-libritts-16kHz",
            run_opts={"device": str(device)},
        )
        vocoder = vocoder.to(device)
        vocoder.eval()
    else:
        vocoder = get_vocoder(model_config, device)

    # maps
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
        spk_map = json.load(f)

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json")) as f:
        emo_map = json.load(f)

    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

    # load source
    items = []

    with open(args.source, "r", encoding="utf-8") as f:
        for line in f:

            parsed = parse_line(line)
            if parsed is None:
                continue

            utt_id, spk, emo, phones, raw_text = parsed

            if emo == args.neutral_name:
                continue

            if args.only_emotion and emo != args.only_emotion:
                continue

            if spk not in spk_map:
                raise KeyError(spk)
            if emo not in emo_map:
                raise KeyError(emo)

            items.append((utt_id, spk, emo, phones, raw_text))

    if len(items) == 0:
        raise RuntimeError("No items to synthesize.")

    intensity_list = make_intensity_list(
        args.intensity_start,
        args.intensity_step,
    )

    print("[INTENSITY GRID]", intensity_list)

    # emotion grouping
    emo_groups = {}
    for it in items:
        emo_groups.setdefault(it[2], []).append(it)

    # synth
    for emo, emo_items in emo_groups.items():

        print(f"\n===== Emotion {emo} ({len(emo_items)}) =====")

        for intensity in intensity_list:

            out_dir = os.path.join(
                result_root,
                f"{emo}_{intensity:.1f}"
            )
            os.makedirs(out_dir, exist_ok=True)

            for mini in chunk_list(emo_items, args.batch_size):

                batch = collate_batch(mini, spk_map, emo_map, cleaners)
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