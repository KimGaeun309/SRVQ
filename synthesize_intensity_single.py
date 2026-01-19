#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import torch

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, synth_samples

# =========================
# Helpers
# =========================
def make_intensity_list(step):
    # inclusive [0.0, 1.0]
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    vals = [float(f"{v:.1f}") for v in vals]  # 1 decimal fixed
    if vals[0] != 0.0:
        vals = [0.0] + vals
    if vals[-1] != 1.0:
        vals.append(1.0)
    vals = sorted(list(set(vals)))
    return vals


def phones_to_sequence_korean(text, cleaners):
    """
    raw text -> cleaned -> phone string -> sequence
    (기존 synthesize.py 흐름 최대한 재사용)
    """
    import re
    from g2pk import G2p
    from jamo import h2j
    from text import _clean_text
    from text.korean import tokenize, normalize_nonchar
    from text import text_to_sequence

    g2p = G2p()
    filters = r'([.,!?])"'
    text = re.sub(re.compile(filters), "", text)

    # cleaners 강제 korean_cleaners 쓰는게 안전함
    # (기존 코드처럼)
    cleaners = ["korean_cleaners"]

    text = _clean_text(text, cleaners)
    text = h2j(g2p(text))

    # tokenize -> "{...}" phone string
    phones = []
    words = filter(None, re.split(r"([,;.\-\?\!\s+])", text))
    for w in words:
        phones += list(filter(lambda p: p != " ", tokenize(w, norm=False)))

    phones = "{" + "}{".join(phones) + "}"
    phones = normalize_nonchar(phones, inference=True)
    phones = phones.replace("}{", " ")

    seq = np.array(text_to_sequence(phones, cleaners), dtype=np.int64)
    return phones, seq, cleaners


def build_single_batch(
    utt_id,
    raw_text,
    seq,
    speaker_id,
    emotion_id,
):
    """
    to_device() 의 7-field 케이스에 맞춰 batch 구성
    return:
        (ids, raw_texts, speakers, emotions, texts, src_lens, max_src_len)
    """
    ids = [utt_id]
    raw_texts = [raw_text]

    speakers = np.array([speaker_id], dtype=np.int64)
    emotions = np.array([emotion_id], dtype=np.int64)

    src_len = len(seq)
    texts = np.zeros((1, src_len), dtype=np.int64)
    texts[0, :src_len] = seq
    src_lens = np.array([src_len], dtype=np.int64)

    max_src_len = src_len
    return (ids, raw_texts, speakers, emotions, texts, src_lens, max_src_len)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--restore_step", type=str, required=True)

    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--speaker", type=str, default="CHY")

    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--neutral_name", type=str, default="neu")

    parser.add_argument("--intensity_step", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=5)

    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)

    parser.add_argument("--seed_base", type=int, default=1234)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TTS] device = {device}")

    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)

    # model / vocoder
    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    model.eval()

    # maps
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
        spk_map = json.load(f)
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json")) as f:
        emo_map = json.load(f)

    if args.speaker not in spk_map:
        raise KeyError(f"[ERROR] speaker '{args.speaker}' not in speakers.json")

    # target emotions = all except neutral
    emo_list = [e for e in emo_map.keys() if e != args.neutral_name]
    emo_list = sorted(emo_list)

    intensity_list = make_intensity_list(args.intensity_step)

    print("[INFO] emotions:", emo_list)
    print("[INFO] intensity grid:", intensity_list)
    print("[INFO] repeat:", args.repeat)
    print("[INFO] total outputs =", len(emo_list) * len(intensity_list) * args.repeat)

    os.makedirs(args.output_path, exist_ok=True)

    # preprocess one text once
    phones, seq, cleaners = phones_to_sequence_korean(args.text, preprocess_config["preprocessing"]["text"]["text_cleaners"])
    print("[TEXT]", args.text)
    print("[PHONES]", phones)

    speaker_id = spk_map[args.speaker]

    # main loop
    for emo in emo_list:
        emotion_id = emo_map[emo]

        for intensity in intensity_list:
            out_dir = os.path.join(args.output_path, f"{emo}_{intensity:.1f}")
            os.makedirs(out_dir, exist_ok=True)

            for r in range(args.repeat):
                # 랜덤성 조금 주고 싶으면 seed 흔들기
                torch.manual_seed(args.seed_base + r)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed_base + r)

                utt_id = f"single_{args.speaker}_{emo}_{intensity:.1f}_r{r}"

                batch = build_single_batch(
                    utt_id=utt_id,
                    raw_text=args.text,
                    seq=seq,
                    speaker_id=speaker_id,
                    emotion_id=emotion_id,
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

            print(f"[SAVE] {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()