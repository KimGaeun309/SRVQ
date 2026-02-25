#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
import numpy as np
import librosa
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import silhouette_score
from collections import defaultdict


############################################################
# -------------------- AUDIO FEATURES ----------------------
############################################################

def extract_f0_mean(wav, sr):
    f0, _, _ = librosa.pyin(
        wav,
        sr=sr,
        fmin=50,
        fmax=500,
    )
    f0 = f0[~np.isnan(f0)]
    if len(f0) == 0:
        return 0.0
    return float(np.mean(f0))


def extract_energy(wav):
    rms = librosa.feature.rms(y=wav)[0]
    return float(np.mean(rms))


def extract_duration(wav, sr):
    return len(wav) / sr


############################################################
# -------------------- SER PLACEHOLDER ---------------------
############################################################

def load_ser_model():
    """
    TODO:
    너 SER model 로딩 코드 넣어라.
    return model
    """
    return None


def run_ser(model, wav, sr):
    """
    TODO:
    너 SER inference 코드 넣어라.

    return:
        pred_intensity : float
        embedding      : np.ndarray (D,)
    """
    pred_intensity = np.random.rand()  # placeholder
    embedding = np.random.randn(256)
    return pred_intensity, embedding


############################################################
# -------------------- HELPERS -----------------------------
############################################################

def parse_folder(folder_name):
    # angry_0.1
    emo, inten = folder_name.rsplit("_", 1)
    return emo, float(inten)


def collect_files(root, allowed_intensity):
    data = []

    for folder in os.listdir(root):
        full = os.path.join(root, folder)
        if not os.path.isdir(full):
            continue

        emo, inten = parse_folder(folder)
        if inten not in allowed_intensity:
            continue

        wavs = glob.glob(os.path.join(full, "*.wav"))
        for w in wavs:
            uid = os.path.basename(w)
            data.append({
                "emotion": emo,
                "target": inten,
                "uid": uid,
                "wav": w,
            })

    return data


############################################################
# -------------------- METRICS -----------------------------
############################################################

def compute_intensity_corr(targets, preds):
    r, _ = pearsonr(targets, preds)
    return r


def compute_mae(targets, preds):
    return np.mean(np.abs(np.array(targets) - np.array(preds)))


def compute_spearman(targets, preds):
    rho, _ = spearmanr(targets, preds)
    return rho


def compute_monotonic_success(records, eps=0.0):
    """
    same uid, same emotion 기준 triplet 검사
    """
    grouped = defaultdict(dict)

    for r in records:
        key = (r["emotion"], r["uid"])
        grouped[key][r["target"]] = r["pred"]

    success = 0
    total = 0

    for k, v in grouped.items():
        if len(v) < 3:
            continue

        if 0.1 in v and 0.5 in v and 0.9 in v:
            w = v[0.1]
            m = v[0.5]
            s = v[0.9]

            total += 1
            if (w + eps < m) and (m + eps < s):
                success += 1

    if total == 0:
        return 0.0

    return success / total


def compute_f0_corr(targets, f0s):
    r, _ = pearsonr(targets, f0s)
    return r


def compute_energy_corr(targets, energies):
    r, _ = pearsonr(targets, energies)
    return r


def compute_duration_corr(targets, durations):
    r, _ = pearsonr(targets, durations)
    return r


def compute_intensity_separability(embeddings, labels):
    """
    silhouette score
    """
    emb = np.stack(embeddings, axis=0)
    labels = np.array(labels)

    if len(np.unique(labels)) < 2:
        return 0.0

    return silhouette_score(emb, labels, metric="cosine")


############################################################
# -------------------- MAIN -------------------------------
############################################################

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--num_intensity", type=int, default=3)
    args = parser.parse_args()

    if args.num_intensity == 3:
        intensity_list = [0.1, 0.5, 0.9]
    elif args.num_intensity == 5:
        intensity_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    else:
        raise ValueError("num_intensity must be 3 or 5")

    print("[INFO] intensity list:", intensity_list)

    records = collect_files(args.root, intensity_list)

    print("[INFO] total wavs:", len(records))

    ser_model = load_ser_model()

    targets = []
    preds = []

    f0s = []
    energies = []
    durations = []

    embeddings = []
    labels = []

    for r in records:

        wav, sr = librosa.load(r["wav"], sr=16000)

        pred, emb = run_ser(ser_model, wav, sr)

        f0 = extract_f0_mean(wav, sr)
        energy = extract_energy(wav)
        dur = extract_duration(wav, sr)

        r["pred"] = pred

        targets.append(r["target"])
        preds.append(pred)

        f0s.append(f0)
        energies.append(energy)
        durations.append(dur)

        embeddings.append(emb)
        labels.append(r["target"])

    ####################################################
    # METRICS
    ####################################################

    print("\n========== RESULTS ==========")

    print("Intensity Corr(Pearson):",
        compute_intensity_corr(targets, preds))

    print("Intensity MAE:",
        compute_mae(targets, preds))

    print("Spearman Monotonicity:",
        compute_spearman(targets, preds))

    print("Monotonic Success Rate:",
        compute_monotonic_success(records, eps=0.01))

    print("F0 Pearson:",
        compute_f0_corr(targets, f0s))

    print("Energy Pearson:",
        compute_energy_corr(targets, energies))

    print("Duration Pearson:",
        compute_duration_corr(targets, durations))

    print("Intensity Separability (silhouette):",
        compute_intensity_separability(embeddings, labels))


if __name__ == "__main__":
    main()