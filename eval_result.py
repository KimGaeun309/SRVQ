import os
import argparse
import glob
import librosa
import numpy as np
import pyworld
import pysptk
import math

from utils.tools import get_configs_of


SAMPLING_RATE = 22050
FRAME_PERIOD = 5.0


#############################
# WORLD MCEP
#############################

def load_wav(wav_file):
    wav, _ = librosa.load(wav_file, sr=SAMPLING_RATE, mono=True)
    return wav


def MCD(x, y):
    const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    diff = x - y
    return const * math.sqrt(np.inner(diff, diff))


def extract_mcep(wav, alpha=0.65, fft_size=512, mcep_size=24):

    wav = wav.astype(np.double)

    _, spectral_envelop, _ = pyworld.wav2world(
        wav,
        fs=SAMPLING_RATE,
        frame_period=FRAME_PERIOD,
        fft_size=fft_size,
    )

    mcep = pysptk.sptk.mcep(
        spectral_envelop,
        order=mcep_size,
        alpha=alpha,
        maxiter=0,
        etype=1,
        eps=1E-8,
        min_det=0.0,
        itype=3,
    )

    return mcep


#############################
# F0
#############################

def extract_f0(wav):

    _f0, t = pyworld.harvest(
        wav.astype(np.double),
        fs=SAMPLING_RATE,
        frame_period=FRAME_PERIOD,
    )

    f0 = pyworld.stonemask(
        wav.astype(np.double),
        _f0,
        t,
        SAMPLING_RATE,
    )

    return f0[f0 > 0]


#############################
# filename parser
#############################

def get_pair_key(name):

    name = name.replace("-gt.wav", "")
    name = name.replace("-pred.wav", "")
    name = name.replace("_gt.wav", "")
    name = name.replace("_pred.wav", "")
    return name

def get_pred_path(dataset, result_path, file_id, emotion):

    if dataset == "inter_2026":
        # CHY_ang_000005.wav
        return os.path.join(
            result_path,
            f"{file_id}.wav",
        )

    elif dataset == "esd":
        # Angry-0011_000525-pred.wav
        return os.path.join(
            result_path,
            f"{emotion}-{file_id}-pred.wav",
        )

    else:
        raise ValueError(dataset)

def get_gt_path(dataset, raw_data, file_id, speaker, emotion):

    if dataset == "inter_2026":
        # wavs/CHY/CHY_ang_000005.wav
        return os.path.join(
            raw_data,
            "wavs",
            speaker,
            f"{file_id}.wav",
        )

    elif dataset == "esd":
        # ESD/0012/Sad/0012_001051.wav
        return os.path.join(
            raw_data,
            speaker,
            emotion,
            f"{file_id}.wav",
        )

    else:
        raise ValueError(dataset)
    

#############################
# main
#############################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--restore_step", required=True)
    parser.add_argument("--raw_data", required=True)

    parser.add_argument(
        "--source",
        required=True,
        help="test.txt or test_ra.txt"
    )


    args = parser.parse_args()

    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    result_path = os.path.join(
        train_config["path"]["result_path"],
        args.restore_step
    )
    configs = (preprocess_config, model_config, train_config)

    result_path = os.path.join(
        train_config["path"]["result_path"],
        args.restore_step
    )

    if not os.path.exists(result_path):
        raise ValueError(f"Result path not found: {result_path}")
    

    with open(args.source, encoding="utf-8") as f:
        infos = [line.strip().split("|") for line in f]
    
    print("Total eval samples:", len(infos))
        
    mcd_all = []
    f0rmse_all = []

    for info in infos:
        file_id, speaker, emotion = info[0], info[1], info[2]

        pred_path = get_pred_path(args.dataset, result_path, file_id, emotion)
        gt_path = get_gt_path(args.dataset, args.raw_data, file_id, speaker, emotion)

        if not os.path.exists(pred_path):
            print(f"Predicted wav not found: {pred_path}")
            continue

        if not os.path.exists(gt_path):
            print(f"Ground truth wav not found: {gt_path}")
            continue

        gt_wav = load_wav(gt_path)
        pred_wav = load_wav(pred_path)

        ################ MCD ################
        mcep_gt = extract_mcep(gt_wav)
        mcep_pred = extract_mcep(pred_wav)

        D, wp = librosa.sequence.dtw(
            mcep_gt[:, 1:].T,
            mcep_pred[:, 1:].T,
            metric=MCD,
        )

        wp = wp[::-1]

        costs = [
            MCD(
                mcep_gt[i, 1:],
                mcep_pred[j, 1:]
            )
            for i, j in wp
        ]

        mcd = np.mean(costs)
        mcd_all.append(mcd)

        ################ F0 RMSE ################
        f0_gt = extract_f0(gt_wav)
        f0_pred = extract_f0(pred_wav)

        L = min(len(f0_gt), len(f0_pred))
        if L == 0:
            continue

        rmse = np.sqrt(
            np.mean(
                (np.log(f0_gt[:L]) - np.log(f0_pred[:L])) ** 2
            )
        )

        f0rmse_all.append(rmse)

    print("================================")
    print("DATASET:", args.dataset)
    print("MCD:", np.mean(mcd_all))
    print("F0 RMSE:", np.mean(f0rmse_all))
    print("================================")