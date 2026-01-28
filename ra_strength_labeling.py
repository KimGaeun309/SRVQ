import os
import random
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import opensmile
from sklearn.svm import LinearSVC


# -----------------------------
# Utils
# -----------------------------
def read_metadata(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 6:
                continue
            utt, wav, spk, emo, text, split = parts
            items.append({
                "utt": utt,
                "wav": wav,
                "speaker": spk,
                "emotion": emo,
                "text": text,
                "split": split,
            })
    return items


def minmax_norm(x):
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-8:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


# -----------------------------
# Main
# -----------------------------
def main(args):
    random.seed(1234)
    np.random.seed(1234)

    print("[1] Read metadata")
    items = read_metadata(args.metadata)

    # emotion 목록
    emotions = sorted(list(set(it["emotion"] for it in items)))
    assert "neutral" in emotions, "neutral emotion is required"
    emotions_wo_neu = [e for e in emotions if e != "neutral"]

    print("Emotions:", emotions)

    print("[2] openSMILE init")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    print("[3] Extract openSMILE features")
    X = {}
    for it in tqdm(items):
        feat = smile.process_file(it["wav"]).values.squeeze()
        X[it["utt"]] = feat.astype(np.float32)

    D = next(iter(X.values())).shape[0]
    print(f"Feature dim = {D}")

    # utt → emotion mapping
    utt2emo = {it["utt"]: it["emotion"] for it in items}

    # -----------------------------
    # 4. Emotion-wise ranking SVM
    # -----------------------------
    utt2strength = {}

    for emo in emotions_wo_neu:
        print(f"[4] Train ranking SVM for emotion = {emo}")

        Xe = [X[u] for u in X if utt2emo[u] == emo]
        Xn = [X[u] for u in X if utt2emo[u] == "neutral"]

        Xe = np.stack(Xe)
        Xn = np.stack(Xn)

        assert len(Xe) > 0 and len(Xn) > 0

        Z = []
        t = []

        # ordered pairs: emo > neutral
        for i in range(len(Xe)):
            j = random.randrange(len(Xn))
            Z.append(Xe[i] - Xn[j])
            t.append(1)

        # similar pairs: emo ~ emo
        for i in range(len(Xe)):
            j = random.randrange(len(Xe))
            Z.append(Xe[i] - Xe[j])
            t.append(0)

        # similar pairs: neu ~ neu
        for i in range(len(Xn)):
            j = random.randrange(len(Xn))
            Z.append(Xn[i] - Xn[j])
            t.append(0)

        Z = np.stack(Z)
        t = np.array(t)

        clf = LinearSVC(C=1.0, max_iter=5000)
        clf.fit(Z, t)

        w = clf.coef_.squeeze()   # (D,)

        # raw score
        r_e = Xe @ w
        r_n = Xn @ w

        # normalize jointly (emo + neutral)
        r_all = np.concatenate([r_e, r_n], axis=0)
        r_all_norm = minmax_norm(r_all)

        r_e_norm = r_all_norm[:len(r_e)]
        r_n_norm = r_all_norm[len(r_e):]

        # assign strength
        idx_e = 0
        idx_n = 0
        for u in X:
            if utt2emo[u] == emo:
                utt2strength[u] = float(r_e_norm[idx_e])
                idx_e += 1
            elif utt2emo[u] == "neutral":
                # neutral은 0 근처로 몰리게 됨
                utt2strength[u] = float(r_n_norm[idx_n])
                idx_n += 1

    # 혹시 neutral이 여러 emotion에서 overwrite된 경우 대비
    for u in X:
        if utt2emo[u] == "neutral" and u not in utt2strength:
            utt2strength[u] = 0.0

    # -----------------------------
    # 5. Write ra txt files
    # -----------------------------
    print("[5] Write train/val/test_ra.txt")

    outs = {
        "train": open("train_ra.txt", "w", encoding="utf-8"),
        "val": open("val_ra.txt", "w", encoding="utf-8"),
        "test": open("test_ra.txt", "w", encoding="utf-8"),
    }

    for it in items:
        strength = utt2strength[it["utt"]]
        line = (
            f"{it['utt']}|{it['speaker']}|{it['emotion']}|"
            f"{strength:.6f}|{it['text']}\n"
        )
        outs[it["split"]].write(line)

    for f in outs.values():
        f.close()

    print("Done.")
    print("Generated: train_ra.txt / val_ra.txt / test_ra.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    args = parser.parse_args()
    main(args)