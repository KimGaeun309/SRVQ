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

# emotion label -> index
with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json")) as f:
    emo_map = json.load(f)

num_rvq = model_config["residual_vq"]["num_rvq"]

# 감정별로 (index, vector) 리스트 저장 (stage별)
emotion_code_info = defaultdict(lambda: [[] for _ in range(num_rvq)])

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
        # 1) ref_enc
        ref_emb, cls_loss = model.ref_enc(mel_tensor, emotion_tensor)
        # 2) style_extractor -> indices, codebooks(list: [z1, z2, z3, concat])
        _, _, indices, codebooks = model.style_extractor(ref_emb, cls_loss)

        # 각 stage에서 사용된 코드 인덱스 기록 + 해당 벡터 저장
        for i in range(num_rvq):
            idx = indices[i].item()
            vec = codebooks[i].squeeze(0).cpu()  # [256]
            emotion_code_info[emotion_label][i].append((idx, vec))

# 감정별 대표 코드 선택 → concat(256*num_rvq) → 3) style_extract_fc → 256D 저장
out_dir = "emotion_style_vectors"
os.makedirs(out_dir, exist_ok=True)

with torch.no_grad():
    for emo, rvq_info in emotion_code_info.items():
        picked_vecs = []
        for i, pairs in enumerate(rvq_info):
            if len(pairs) == 0:
                raise RuntimeError(f"No samples collected for emotion={emo}, stage={i}")
            # 가장 빈도 높은 코드 인덱스 선택
            most_common_idx = Counter([p[0] for p in pairs]).most_common(1)[0][0]
            # 해당 인덱스의 첫 벡터 사용
            vec = next(v for idx, v in pairs if idx == most_common_idx)  # [256]
            picked_vecs.append(vec)

        # [256*num_rvq] -> [1, 256*num_rvq] -> style_extract_fc -> [1,256]
        concat_vec = torch.cat(picked_vecs, dim=0).unsqueeze(0)  # [1, 256*num_rvq]
        # model.style_extract_fc: 256*num_rvq -> 256
        style_256 = model.style_extract_fc(concat_vec.to(device)).squeeze(0).cpu().numpy()  # [256]

        np.save(os.path.join(out_dir, f"{emo}_style.npy"), style_256)
        print(f"Saved {emo}: {style_256.shape} -> {os.path.join(out_dir, f'{emo}_style.npy')}")