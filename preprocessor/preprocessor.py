import os
import random
import json
import tgt
import librosa
import numpy as np
import pyworld as pw
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import audio as Audio
import natsort
import re

from text import text_to_sequence


class Preprocessor:
    def __init__(self, preprocess_config, model_config, train_config):
        random.seed(train_config["seed"])
        self.preprocess_config = preprocess_config
        self.multi_speaker = model_config["multi_speaker"]
        self.in_dir = preprocess_config["path"]["raw_path"]
        self.out_dir = preprocess_config["path"]["preprocessed_path"]
        self.mfa_dir = preprocess_config["path"]["mfa_path"]

        self.val_size = preprocess_config["preprocessing"]["val_size"]
        self.test_size = preprocess_config["preprocessing"]["test_size"]
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]

        self.pitch_phoneme_averaging = (
            preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = preprocess_config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = preprocess_config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs(os.path.join(self.out_dir, "mel"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "pitch"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "energy"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "duration"), exist_ok=True)

        print("[INFO] Processing dataset ...")

        metadata_path = os.path.join(self.out_dir, "metadata.csv")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_lines = f.readlines()

        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        data_infos = []
        n_frames = 0

        for line in tqdm(metadata_lines):
            line = line.strip()
            if len(line.split("|")) != 4:
                continue
            wav_rel, speaker, emotion, raw_text = line.split("|")
            basename = os.path.basename(wav_rel).replace(".wav", "")
            emotion = emotion.lower()
            wav_path = os.path.join(self.out_dir, wav_rel)

            tg_path = os.path.join(self.mfa_dir, f"{speaker}/{basename}.TextGrid")
            if not os.path.exists(tg_path):
                print(f"[WARN] Missing TextGrid: {tg_path}")
                continue

            # MFA alignment
            textgrid = tgt.io.read_textgrid(tg_path)
            phone_tier = textgrid.get_tier_by_name("phones")
            phones, durations, start, end = self.get_alignment(phone_tier)

            if len(phones) == 0 or sum(durations) == 0:
                continue

            phones_str = "{" + " ".join(phones) + "}"

            ret = self.process_utterance(speaker, basename, emotion, wav_path, durations, phones, raw_text)
            if ret is None:
                continue

            info, pitch, energy, n = ret
            data_infos.append(info)
            n_frames += n

            if len(pitch) > 0:
                pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
            if len(energy) > 0:
                energy_scaler.partial_fit(energy.reshape((-1, 1)))

        # -------- Normalization --------
        print("[INFO] Computing statistics ...")
        pitch_mean = pitch_scaler.mean_[0] if self.pitch_normalization else 0
        pitch_std = pitch_scaler.scale_[0] if self.pitch_normalization else 1
        energy_mean = energy_scaler.mean_[0] if self.energy_normalization else 0
        energy_std = energy_scaler.scale_[0] if self.energy_normalization else 1

        pitch_min, pitch_max = self.normalize(os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std)
        energy_min, energy_max = self.normalize(os.path.join(self.out_dir, "energy"), energy_mean, energy_std)

        stats = {
            "pitch": [float(pitch_min), float(pitch_max), float(pitch_mean), float(pitch_std)],
            "energy": [float(energy_min), float(energy_max), float(energy_mean), float(energy_std)],
        }
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=4)

        print(f"[INFO] Total time: {n_frames * self.hop_length / self.sampling_rate / 3600:.2f} hours")
        self.split_dataset(data_infos)

    def process_utterance(self, speaker, basename, emotion, wav_path, durations, phones, raw_text):
        if not os.path.exists(wav_path):
            return None

        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav.astype(np.float32)

        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        if np.sum(pitch != 0) <= 1:
            return None

        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)

        # Average over phoneme durations
        if self.pitch_phoneme_averaging:
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids, pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            pos = 0
            for i, d in enumerate(durations):
                pitch[i] = np.mean(pitch[pos:pos + d]) if d > 0 else 0
                pos += d
            pitch = pitch[:len(durations)]

        if self.energy_phoneme_averaging:
            pos = 0
            for i, d in enumerate(durations):
                energy[i] = np.mean(energy[pos:pos + d]) if d > 0 else 0
                pos += d
            energy = energy[:len(durations)]

        # Save
        duration_filename = f"{speaker}-duration-{speaker}_{emotion}_{basename}.npy"
        pitch_filename = f"{speaker}-pitch-{speaker}_{emotion}_{basename}.npy"
        energy_filename = f"{speaker}-energy-{speaker}_{emotion}_{basename}.npy"
        mel_filename = f"{speaker}-mel-{speaker}_{emotion}_{basename}.npy"

        np.save(os.path.join(self.out_dir, "duration", duration_filename), np.array(durations))
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)
        np.save(os.path.join(self.out_dir, "mel", mel_filename), mel_spectrogram.T)

        info = f"{speaker}_{emotion}_{basename}|{speaker}|{emotion}|{{ {' '.join(['@'+p for p in phones if p not in ['sil','sp','spn']])} }}|{raw_text}"
        return info, pitch, energy, mel_spectrogram.shape[1]

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]
        phones, durations = [], []
        start_time, end_time, end_idx = 0, 0, 0

        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text.strip()
            if phones == [] and p in sil_phones:
                continue
            if p not in sil_phones:
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                phones.append(p)
            durations.append(int(np.round(e * self.sampling_rate / self.hop_length) -
                                 np.round(s * self.sampling_rate / self.hop_length)))
        phones, durations = phones[:end_idx], durations[:end_idx]
        return phones, durations, start_time, end_time

    def normalize(self, in_dir, mean, std):
        max_v, min_v = np.finfo(np.float64).min, np.finfo(np.float64).max
        for f in os.listdir(in_dir):
            path = os.path.join(in_dir, f)
            values = (np.load(path) - mean) / std
            np.save(path, values)
            max_v = max(max_v, max(values))
            min_v = min(min_v, min(values))
        return min_v, max_v

    def split_dataset(self, data_infos):
        random.shuffle(data_infos)
        n_total = len(data_infos)
        n_val, n_test = self.val_size, self.test_size
        n_train = n_total - n_val - n_test

        train = data_infos[:n_train]
        val = data_infos[n_train:n_train + n_val]
        test = data_infos[n_train + n_val:]

        for name, data in zip(["train", "val", "test", "all"], [train, val, test, data_infos]):
            with open(os.path.join(self.out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                for line in natsort.natsorted(data):
                    f.write(line + "\n")

        print(f"[INFO] Train:{len(train)} Val:{len(val)} Test:{len(test)} All:{n_total}")