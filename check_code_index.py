# check_code_index.py

import json
from collections import defaultdict, Counter
import torch
import numpy as np
import os
from utils.model import get_model
from utils.tools import get_configs_of
import argparse


def run_check_code_index(restore_step: str, dataset: str, source_path: str):
    device = "cpu"

    preprocess_config, model_config, train_config = get_configs_of(dataset)
    configs = (preprocess_config, model_config, train_config)

    class Args:
        pass
    args = Args()
    args.restore_step = restore_step
    args.dataset = dataset

    model = get_model(args, configs, device, train=False)
    model.eval()

    with open(preprocess_config["path"]["preprocessed_path"] + "/emotions.json") as f:
        emo_map = json.load(f)

    num_rvq = model_config["residual_vq"]["num_rvq"]
    emotion_code_info = defaultdict(lambda: [[] for _ in range(num_rvq)])

    with open(source_path) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split("|")
        basename, _, emotion_label, _, _ = parts
        emotion_idx = emo_map[emotion_label]

        mel_path = f"{preprocess_config['path']['preprocessed_path']}/mel/{basename[:3]}-mel-{basename}.npy"
        mel = np.load(mel_path)
        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).to(device)

        with torch.no_grad():
            style, _, indices, codebooks = model.style_extractor(mel_tensor)

            for i in range(num_rvq):
                idx = indices[i].item()
                vec = codebooks[i].squeeze(0).cpu()
                emotion_code_info[emotion_label][i].append((idx, vec))

    os.makedirs("emotion_style_vectors_mode", exist_ok=True)
    for emo, rvq_info in emotion_code_info.items():
        final_vecs = []
        for i, pairs in enumerate(rvq_info):
            index_counts = Counter([p[0] for p in pairs])
            most_common_idx = index_counts.most_common(1)[0][0]
            vec = [v for idx, v in pairs if idx == most_common_idx][0]
            final_vecs.append(vec)
        style_vector = torch.cat(final_vecs, dim=0).numpy()
        print(f"Emotion: {emo} | Shape: {style_vector.shape}")
        np.save(f"emotion_style_vectors_mode/{emo}_style.npy", style_vector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--source", type=str, default='preprocessed_data/emo_kr_22050/train.txt')
    args = parser.parse_args()

    run_check_code_index(
        restore_step=args.restore_step,
        dataset=args.dataset,
        source_path=args.source
    )
