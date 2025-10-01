import argparse
import json
from collections import defaultdict, Counter
import torch
import numpy as np
from utils.model import get_model
from utils.tools import get_configs_of
import os

parser = argparse.ArgumentParser()
parser.add_argument("--restore_step", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--source", type=str, default='preprocessed_data/emo_kr_22050/train.txt')
args = parser.parse_args()

device = "cpu"

preprocess_config, model_config, train_config = get_configs_of(args.dataset)
configs = (preprocess_config, model_config, train_config)

model = get_model(args, configs, device, train=False)
model.eval()

# Load emotion map
with open(preprocess_config["path"]["preprocessed_path"] + "/emotions.json") as f:
    emo_map = json.load(f)

num_rvq = model_config["residual_vq"]["num_rvq"]

# 감정별로 (index, vector) 리스트 저장
emotion_code_info = defaultdict(lambda: [[] for _ in range(num_rvq)])

# Source 파일에서 데이터 읽기
with open(args.source) as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split("|")
    basename, _, emotion_label, _, _ = parts
    emotion_idx = emo_map[emotion_label]

    mel = np.load(f"{preprocess_config['path']['preprocessed_path']}/mel/{basename[:3]}-mel-{basename}.npy")
    mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).to(device)
    emotion_tensor = torch.tensor([emotion_idx], device=device)

    with torch.no_grad():
        ref_emb, cls_loss = model.ref_enc(mel_tensor, emotion_tensor)
        _, _, indices, codebooks = model.style_extractor(ref_emb, cls_loss)
        # codebooks: [z_q_1, z_q_2, z_q_3, concat]

        for i in range(num_rvq):
            idx = indices[i].item()
            vec = codebooks[i].squeeze(0).cpu()
            emotion_code_info[emotion_label][i].append((idx, vec))

# 각 감정별로 가장 자주 등장한 index에 해당하는 vector 선택 → concat 후 저장
os.makedirs("emotion_style_vectors", exist_ok=True)
for emo, rvq_info in emotion_code_info.items():
    final_vecs = []
    for i, pairs in enumerate(rvq_info):
        index_counts = Counter([p[0] for p in pairs])
        most_common_idx = index_counts.most_common(1)[0][0]
        # most_common_idx에 해당하는 vector 추출
        vec = [v for idx, v in pairs if idx == most_common_idx][0]  # 첫 번째 vector 사용
        final_vecs.append(vec)
    style_vector = torch.cat(final_vecs, dim=0).numpy()
    print(f"Emotion: {emo} | Shape: {style_vector.shape}")
    np.save(f"emotion_style_vectors_mode/{emo}_style.npy", style_vector)

