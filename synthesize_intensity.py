#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import numpy as np
import torch

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, synth_samples


def parse_line(line):
    # utt_id|speaker|emotion|{phonemes}|raw_text
    parts = line.strip().split("|")
    if len(parts) < 5:
        return None
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def make_intensity_list(step):
    n = int(round(1.0 / step))
    vals = [round(i * step, 10) for i in range(n + 1)]
    vals[0] = 0.0
    vals[-1] = 1.0
    return vals


def chunk_list(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i + bs]


def phones_to_sequence(phones, cleaners):
    """
    phones: "{ᄀ ᅳ ...}" 형태 문자열
    text_to_sequence는 이미 프로젝트 안에 있음
    """
    from text import text_to_sequence
    seq = np.array(text_to_sequence(phones, cleaners), dtype=np.int64)
    return seq


def collate_batch(batch_items, spk_map, emo_map, cleaners):
    """
    batch_items: list of (utt_id, speaker_str, emotion_str, phones_str, raw_text)

    return tuple that matches to_device() 7-field case:
        (ids, raw_texts, speakers, emotions, texts, src_lens, max_src_len)
    """
    ids = []
    raw_texts = []
    speakers = []
    emotions = []
    texts = []
    src_lens = []

    for utt_id, spk, emo, phones, raw_text in batch_items:
        ids.append(utt_id)
        raw_texts.append(raw_text)

        speakers.append(spk_map[spk])
        emotions.append(emo_map[emo])

        seq = phones_to_sequence(phones, cleaners)
        texts.append(seq)
        src_lens.append(len(seq))

    max_src_len = max(src_lens)

    # pad texts -> (B, max_src_len)
    padded_texts = np.zeros((len(texts), max_src_len), dtype=np.int64)
    for i, seq in enumerate(texts):
        padded_texts[i, : len(seq)] = seq

    speakers = np.array(speakers, dtype=np.int64)
    emotions = np.array(emotions, dtype=np.int64)
    src_lens = np.array(src_lens, dtype=np.int64)

    return (ids, raw_texts, speakers, emotions, padded_texts, src_lens, max_src_len)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--restore_step", type=str, required=True)

    parser.add_argument("--source", type=str, required=True,
                        help="test.txt path (utt_id|speaker|emotion|{phonemes}|raw_text)")
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--neutral_name", type=str, default="neu")
    parser.add_argument("--only_emotion", type=str, default=None)

    parser.add_argument("--ckpt_ser", type=str, default=None)  # not used, just for compatibility if needed
    parser.add_argument("--intensity_step", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TTS] device = {device}")

    # configs
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)

    # model/vocoder
    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    model.eval()

    # maps
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
        spk_map = json.load(f)
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json")) as f:
        emo_map = json.load(f)

    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

    # load source file
    with open(args.source, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    items = []
    for line in lines:
        parsed = parse_line(line)
        if parsed is None:
            continue
        utt_id, spk, emo, phones, raw_text = parsed

        # skip neutral
        if emo == args.neutral_name:
            continue

        if args.only_emotion is not None and emo != args.only_emotion:
            continue

        if spk not in spk_map:
            raise KeyError(f"[ERROR] speaker '{spk}' not in speakers.json")
        if emo not in emo_map:
            raise KeyError(f"[ERROR] emotion '{emo}' not in emotions.json")

        items.append((utt_id, spk, emo, phones, raw_text))

    if len(items) == 0:
        raise RuntimeError("No items to synthesize (check source / filters).")

    os.makedirs(args.output_path, exist_ok=True)

    intensity_list = make_intensity_list(args.intensity_step)
    print("[INFO] intensity grid:", intensity_list)
    print("[INFO] total non-neutral samples:", len(items))

    # group by emotion (to create folders emo_0.0, emo_0.1 ...)
    emo_groups = {}
    for it in items:
        emo_groups.setdefault(it[2], []).append(it)

    # synth loop
    for emo, emo_items in emo_groups.items():
        print(f"\n========== Emotion: {emo} (N={len(emo_items)}) ==========")

        for intensity in intensity_list:
            out_dir = os.path.join(args.output_path, f"{emo}_{intensity:.1f}")
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

            print(f"[SAVE] {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()