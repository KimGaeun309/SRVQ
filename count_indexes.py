from collections import defaultdict, Counter
import argparse, json, torch, numpy as np
from utils.model import get_model
from utils.tools import get_configs_of

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
emotion_rvq_index_counts = defaultdict(lambda: [Counter() for _ in range(num_rvq)])

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
        _, _, indices, _ = model.style_extractor(ref_emb, cls_loss)
        for i in range(num_rvq):
            idx = indices[i].item()
            emotion_rvq_index_counts[emotion_label][i][idx] += 1

# 예쁘게 출력
for emo, rvq_counters in emotion_rvq_index_counts.items():
    print(f"\nEmotion: {emo}")
    for i, counter in enumerate(rvq_counters):
        print(f"  RVQ Layer {i+1}:")
        for idx, count in sorted(counter.items()):
            print(f"    Index {idx:2d} → {count} times")
