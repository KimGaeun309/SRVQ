#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ra_strength_labeling.py
Parikh-style primal Newton RankSVM for emotion strength labeling

Input txt (FS2):
  utt|speaker|emotion|phoneme|text

Wav path:
  {wav_root}/{speaker}/{utt}.wav

Baseline emotion:
  neu

Output:
  utt|speaker|emotion|phoneme|text|strength
"""

import os
import random
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import opensmile

import scipy
import scipy.sparse
from scipy.optimize import least_squares


# ============================================================
# Parikh-style RankSVM (primal, Newton)
# ============================================================

_X = None     # (n,d) matrix
_A = None     # (p,n) sparse
_n0 = None    # number of ordered constraints


def _obj_fun_linear(w, C, out):
    global _X, _A, _n0

    out = out.copy()
    out[:_n0, 0] = np.maximum(out[:_n0, 0], 0.0)

    obj = 0.5 * (w.T @ w) + 0.5 * (C.T @ np.multiply(out, out))
    grad = w - ((_A.T @ (np.multiply(C, out))).T @ _X).T

    sv_ordered = (out[:_n0, 0] > 0)
    sv_sim = (np.abs(out[_n0:, 0]) > 0)
    sv = np.vstack([sv_ordered.reshape(-1, 1),
                    sv_sim.reshape(-1, 1)]).astype(bool)

    return float(obj[0, 0]), grad, sv


def _hess_vect_mult(v, sv, C, grad):
    global _X, _A

    v = np.matrix(v).T
    y = v
    z = np.multiply(np.multiply(C, sv), _A @ (_X @ v))
    y = y + (((z.T @ _A) @ _X).T) + grad
    return np.asarray(y).reshape(-1)


def _line_search_linear(w, d, out, C):
    global _X, _A, _n0

    t = 0.0
    Xd = _A @ (_X @ d)
    wd = float(w.T @ d)
    dd = float(d.T @ d)

    while True:
        out2 = out - t * Xd

        sv_mask = np.vstack([
            (out2[:_n0, 0] > 0).reshape(-1, 1),
            (np.abs(out2[_n0:, 0]) > 0).reshape(-1, 1)
        ]).astype(bool).reshape(-1)

        idx = np.nonzero(sv_mask)[0]
        if idx.size == 0:
            return t, out2

        Cout = C[idx, 0] * out2[idx, 0]
        g = wd + t * dd - float(Cout.reshape(1, -1) @ Xd[idx, 0].reshape(-1, 1))
        h = dd + float(Xd[idx, 0].reshape(1, -1)
                       @ (Xd[idx, 0] * C[idx, 0]).reshape(-1, 1))

        t = t - g / h
        if (g * g / h) < 1e-8:
            return t, out2


def rank_svm_parikh(X, O, S, C_O, C_S,
                    max_itr=10, prec=1e-8, cg_prec=1e-8):
    global _X, _A, _n0

    _X = np.matrix(X)
    _n0 = O.shape[0]
    _A = scipy.sparse.vstack([O, S]).tocsr()

    d = _X.shape[1]
    w = np.matrix(np.zeros((d, 1), dtype=np.float64))

    C = np.vstack([C_O, C_S]).astype(np.float64)
    C = np.matrix(C)

    out = np.vstack([
        np.ones((_n0, 1)),
        np.zeros((S.shape[0], 1))
    ])
    out = np.matrix(out) - (_A @ (_X @ w))

    itr = 0
    while True:
        itr += 1
        if itr > max_itr:
            break

        obj, grad, sv = _obj_fun_linear(w, C, out)

        res = least_squares(
            _hess_vect_mult,
            x0=np.zeros((d,)),
            ftol=cg_prec, xtol=cg_prec, gtol=cg_prec,
            args=(sv, C, grad)
        )
        step = np.matrix(res.x).T
        t, out = _line_search_linear(w, step, out, C)
        w = w + t * step

        check = float((-step.T @ grad)[0, 0])
        if check < prec * obj:
            break

    return np.asarray(w).reshape(-1)


# ============================================================
# Utilities
# ============================================================

BASELINE = "neu"


def minmax_norm(x):
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def read_txt(path, split, wav_root):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("|")
            if len(p) < 5:
                continue
            utt, spk, emo, ph, txt = p[:5]
            wav = os.path.join(wav_root, spk, f"{utt}.wav")
            items.append(dict(
                utt=utt, speaker=spk, emotion=emo,
                phoneme=ph, text=txt, wav=wav, split=split
            ))
    return items


def build_pref_matrix(pairs, n):
    if len(pairs) == 0:
        return scipy.sparse.csr_matrix((0, n))

    row = np.arange(len(pairs))
    col_i = np.array([i for i, j in pairs])
    col_j = np.array([j for i, j in pairs])

    data = np.concatenate([np.ones(len(pairs)), -np.ones(len(pairs))])
    rows = np.concatenate([row, row])
    cols = np.concatenate([col_i, col_j])

    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(pairs), n))


# ============================================================
# Main
# ============================================================

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    items = []
    items += read_txt(args.train_txt, "train", args.wav_root)
    items += read_txt(args.val_txt, "val", args.wav_root)
    items += read_txt(args.test_txt, "test", args.wav_root)

    emotions = sorted({it["emotion"] for it in items})
    assert BASELINE in emotions
    emos = [e for e in emotions if e != BASELINE]

    print("[1] Read metadata")
    print("Emotions:", emotions)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    print("smile feature sets", smile.feature_names)

    # --- select feature indices (384-dim) ---
    dummy_wav = next(it["wav"] for it in items if os.path.isfile(it["wav"]))
    feat0 = smile.process_file(dummy_wav)
    names = feat0.columns.tolist()

    lld_kw = ["mfcc", "pcm_RMSenergy", "pcm_zcr", "F0final", "voicingFinalUnclipped"]
    fn_kw = ["max", "min", "range", "amean", "stddev",
             "skewness", "kurtosis", "linregc1", "linregerr", "posmax", "posmin"]

    keep_idx = [i for i, n in enumerate(names)
                if any(k in n for k in lld_kw)
                and any(k in n for k in fn_kw)]

    print("Selected feature dim:", len(keep_idx))

    # --- extract features ---
    X_map = {}
    for it in tqdm(items, desc="[2] Extract features"):
        if it["utt"] in X_map:
            continue
        f = smile.process_file(it["wav"])
        X_map[it["utt"]] = f.iloc[:, keep_idx].values.squeeze().astype(np.float64)

    utts = sorted(X_map.keys())
    idx = {u: i for i, u in enumerate(utts)}
    X = np.stack([X_map[u] for u in utts], axis=0)

    utt2emo = {it["utt"]: it["emotion"] for it in items}
    utt2strength = {}
    neu_scores = defaultdict(list)

    for emo in emos:
        print(f"[3] RankSVM for {emo}")
        emo_utts = [u for u in utts if utt2emo[u] == emo]
        neu_utts = [u for u in utts if utt2emo[u] == BASELINE]
        if not emo_utts or not neu_utts:
            continue

        emo_idx = [idx[u] for u in emo_utts]
        neu_idx = [idx[u] for u in neu_utts]

        O_pairs = [(i, random.choice(neu_idx)) for i in emo_idx]
        S_pairs = [(i, random.choice(emo_idx)) for i in emo_idx] + \
                  [(i, random.choice(neu_idx)) for i in neu_idx]

        O = build_pref_matrix(O_pairs, len(utts))
        S = build_pref_matrix(S_pairs, len(utts))

        C_O = np.ones((O.shape[0], 1))
        C_S = np.ones((S.shape[0], 1))

        w = rank_svm_parikh(X, O, S, C_O, C_S)

        r_e = X[emo_idx] @ w
        r_n = X[neu_idx] @ w
        r = minmax_norm(np.concatenate([r_e, r_n]))

        for u, s in zip(emo_utts, r[:len(r_e)]):
            utt2strength[u] = float(s)
        for u, s in zip(neu_utts, r[len(r_e):]):
            neu_scores[u].append(float(s))

    for u, v in neu_scores.items():
        utt2strength[u] = float(np.mean(v))

    os.makedirs(args.out_dir, exist_ok=True)
    outs = {
        "train": open(os.path.join(args.out_dir, "train_ra.txt"), "w", encoding="utf-8"),
        "val":   open(os.path.join(args.out_dir, "val_ra.txt"),   "w", encoding="utf-8"),
        "test":  open(os.path.join(args.out_dir, "test_ra.txt"),  "w", encoding="utf-8"),
    }

    for it in items:
        s = utt2strength[it["utt"]]
        outs[it["split"]].write(
            f"{it['utt']}|{it['speaker']}|{it['emotion']}|"
            f"{it['phoneme']}|{it['text']}|{s:.6f}\n"
        )

    for f in outs.values():
        f.close()

    print("[Done] RA labels generated.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", required=True)
    ap.add_argument("--val_txt", required=True)
    ap.add_argument("--test_txt", required=True)
    ap.add_argument("--wav_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    main(args)