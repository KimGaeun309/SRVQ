import os
import json
import yaml
import argparse
import numpy as np
from matplotlib import pyplot as plt


def expand(values, durations):
    out = []
    for v, d in zip(values, durations):
        out += [v] * max(0, int(d))
    return np.array(out)


def add_axis(fig, old_ax):
    ax = fig.add_axes(old_ax.get_position(), anchor="W")
    ax.set_facecolor("None")
    return ax


def plot_mel_fastspeech2_style(mel, pitch, energy, stats, basename):
    """Same visualization style as utils/tools.py in FastSpeech2"""
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean
    pitch = pitch * pitch_std + pitch_mean

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(mel, origin="lower")
    ax.set_aspect(2.5, adjustable="box")
    ax.set_ylim(0, mel.shape[0])
    ax.set_title(basename, fontsize="medium")
    ax.tick_params(labelsize="x-small", left=False, labelleft=False)
    ax.set_anchor("W")

    # pitch overlay
    ax1 = add_axis(fig, ax)
    ax1.plot(pitch, color="tomato")
    ax1.set_xlim(0, mel.shape[1])
    ax1.set_ylim(0, pitch_max)
    ax1.set_ylabel("F0", color="tomato")
    ax1.tick_params(labelsize="x-small", colors="tomato", bottom=False, labelbottom=False)

    # energy overlay
    ax2 = add_axis(fig, ax)
    ax2.plot(energy, color="darkviolet")
    ax2.set_xlim(0, mel.shape[1])
    ax2.set_ylim(energy_min, energy_max)
    ax2.set_ylabel("Energy", color="darkviolet")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(
        labelsize="x-small",
        colors="darkviolet",
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
    )
    return fig


def main(basename, base_dir, cfg_path):
    spk = basename.split("_")[0]

    # Load arrays
    mel = np.load(f"{base_dir}/mel/{spk}-mel-{basename}.npy").T
    pitch = np.load(f"{base_dir}/pitch/{spk}-pitch-{basename}.npy")
    energy = np.load(f"{base_dir}/energy/{spk}-energy-{basename}.npy")
    dur = np.load(f"{base_dir}/duration/{spk}-duration-{basename}.npy")

    # Configs
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    p_feat = cfg["preprocessing"]["pitch"]["feature"]
    e_feat = cfg["preprocessing"]["energy"]["feature"]

    with open(f"{base_dir}/stats.json") as f:
        stats = json.load(f)
    stats_combined = stats["pitch"] + stats["energy"][:2]

    # expand if phoneme-level
    if p_feat == "phoneme_level":
        pitch = expand(pitch, dur)
    if e_feat == "phoneme_level":
        energy = expand(energy, dur)

    # match lengths
    T = mel.shape[1]
    pitch = pitch[:T] if len(pitch) >= T else np.pad(pitch, (0, T - len(pitch)))
    energy = energy[:T] if len(energy) >= T else np.pad(energy, (0, T - len(energy)))

    # Plot identical to FS2
    fig = plot_mel_fastspeech2_style(mel, pitch, energy, stats_combined, basename)

    save_dir = "./preprocessed_mels"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{basename}.png")
    plt.savefig(save_path, format="png", dpi=200)
    plt.close(fig)
    print(f"[SAVED] {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True,
                        help="basename without extension, e.g. 0011_angry_0011_000353")
    parser.add_argument("--base", type=str, default="./preprocessed_data/esd",
                        help="base directory containing mel/pitch/energy/duration folders")
    parser.add_argument("--config", type=str, default="./config/esd/preprocess.yaml",
                        help="path to preprocess.yaml config")
    args = parser.parse_args()

    main(args.file, args.base, args.config)