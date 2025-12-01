import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio
import natsort


class Preprocessor:
    def __init__(self, preprocess_config, model_config, train_config):
        random.seed(train_config['seed'])
        self.preprocess_config_config = preprocess_config
        self.multi_speaker = model_config["multi_speaker"]
        self.in_dir = preprocess_config["path"]["raw_path"]
        self.out_dir = preprocess_config["path"]["preprocessed_path"]
        self.val_size = preprocess_config["preprocessing"]["val_size"]
        self.test_size = preprocess_config["preprocessing"]["test_size"]
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]

        assert preprocess_config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert preprocess_config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
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
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = []
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # === metadata.csv 읽기 ===
        metadata_path = os.path.join(self.out_dir, "metadata.csv")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_lines = f.readlines()

        # speaker dict 생성
        speakers = {}

        for line in tqdm(metadata_lines):
            line = line.strip()
            if len(line.split("|")) != 4:
                continue

            wav_rel, speaker, emotion, raw_text = line.split("|")
            wav_path = os.path.join(self.out_dir, wav_rel)
            basename = os.path.splitext(os.path.basename(wav_rel))[0]

            if speaker not in speakers:
                speakers[speaker] = len(speakers)

            tg_path = os.path.join(self.out_dir, "mfa", "aligned_arpa", speaker, f"{basename}.TextGrid")
            if not os.path.exists(tg_path):
                print(f"[WARN] Missing TextGrid: {tg_path}")
                continue

            ret = self.process_utterance(speaker, basename, emotion, wav_path, tg_path, raw_text)
            if ret is None:
                continue
            else:
                info, pitch, energy, n = ret
            out.append(info)

            if len(pitch) > 0:
                pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
            if len(energy) > 0:
                energy_scaler.partial_fit(energy.reshape((-1, 1)))

            n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.seed(777)
        random.shuffle(out)
        out = [r for r in out if r is not None]

        train_set = natsort.natsorted(out[self.val_size:])
        temp_set = out[:self.val_size]
        val_set = temp_set[self.test_size:]
        test_set = temp_set[:self.test_size]

        # Write metadata
        for name, data in zip(["train", "val", "test"], [train_set, val_set, test_set]):
            with open(os.path.join(self.out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                for m in data:
                    f.write(m + "\n")

        return out


    def process_utterance(self, speaker, basename, emotion, wav_path, tg_path, raw_text):
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        # 이전
        # text = "{" + " ".join(phone) + "}"

        # 수정
        text = "{ " + " ".join(f"@{p}" for p in phone) + " }"
        if start >= end:
            return None

        # === Load & Trim ===
        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[int(self.sampling_rate * start): int(self.sampling_rate * end)].astype(np.float32)

        # === Mel & Energy ===
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)

        # === 길이 기준을 duration 합으로 통일 ===
        T = int(sum(duration))                 # ← 핵심: mel 기준이 아니라 duration 기준
        mel_spectrogram = mel_spectrogram[:, :T]
        energy = energy[:T]

        # === F0 ===
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        pitch = pitch[:T]

        # (삭제) diff 보정 블록
        # diff = T_mel - sum(duration) ...
        # duration[-1] += diff

        # === Phoneme-level 평균 ===
        if self.pitch_phoneme_averaging:
            nonzero_ids = np.where(pitch != 0)[0]
            if len(nonzero_ids) > 1:
                interp_fn = interp1d(
                    nonzero_ids,
                    pitch[nonzero_ids],
                    fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                    bounds_error=False,
                )
                pitch = interp_fn(np.arange(len(pitch)))

            pos = 0
            averaged = np.zeros(len(duration))
            for i, d in enumerate(duration):
                averaged[i] = np.mean(pitch[pos:pos + d]) if d > 0 else 0
                pos += d
            pitch = averaged.astype(np.float32)

        if self.energy_phoneme_averaging:
            pos = 0
            averaged = np.zeros(len(duration))
            for i, d in enumerate(duration):
                averaged[i] = np.mean(energy[pos:pos + d]) if d > 0 else 0
                pos += d
            energy = averaged.astype(np.float32)

        # === Save ===
        dur_filename = f"{speaker}-duration-{speaker}_{emotion}_{basename}.npy"
        pitch_filename = f"{speaker}-pitch-{speaker}_{emotion}_{basename}.npy"
        energy_filename = f"{speaker}-energy-{speaker}_{emotion}_{basename}.npy"
        mel_filename = f"{speaker}-mel-{speaker}_{emotion}_{basename}.npy"

        duration = np.array(duration, dtype=np.int32)
        pitch = np.array(pitch, dtype=np.float32)
        energy = np.array(energy, dtype=np.float32)

        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)
        np.save(os.path.join(self.out_dir, "mel", mel_filename), mel_spectrogram.T)

        return (
            "|".join([f"{speaker}_{emotion}_{basename}", speaker, emotion, text, raw_text]),
            pitch,
            energy,
            mel_spectrogram.shape[1],
        )


    def get_alignment(self, tier):
        sil = {"sil", "sp", "spn"}
        phones, durations = [], []
        start_time, end_time = 0.0, 0.0
        started = False

        for itv in tier._objects:
            s, e, p = itv.start_time, itv.end_time, itv.text.strip()

            if p in sil:
                # 모든 silence는 duration/phones에서 제외
                if not started:
                    continue
                else:
                    continue

            if not started:
                started = True
                start_time = s

            phones.append(p)
            dur = int(
                np.round(e * self.sampling_rate / self.hop_length)
                - np.round(s * self.sampling_rate / self.hop_length)
            )
            durations.append(max(dur, 0))
            end_time = e

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)
        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)
            max_value = max(max_value, np.max(values))
            min_value = min(min_value, np.min(values))
        return min_value, max_value