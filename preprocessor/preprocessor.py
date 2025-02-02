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
import re

from g2pk import G2p
from jamo import h2j
from text import _clean_text


from text import text_to_sequence
from text.korean import tokenize, normalize_nonchar

def preprocess_korean(text, cleaners):
    # lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    words = filter(None, re.split(r"([,;.\-\?\!\s+])", text))
    for w in words:
        # if w in lexicon:
        #     phones += lexicon[w]
        # else:
        phones += list(filter(lambda p: p != " ", tokenize(w, norm=False)))
    phones = "{" + "}{".join(phones) + "}"
    phones = normalize_nonchar(phones, inference=True)
    phones = phones.replace("}{", " ")

    # print("Raw Text Sequence: {}".format(text))
    # print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, cleaners
        )
    )

    return phones

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
        out_6 = list()
        out_all = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        # speakers = {}
        new_data_path = "/media/gaeun/fee580d1-6954-462b-bdaf-3b8ccc4254311/data/015.Emotion_and_Style_TTS"
        metadata_path = os.path.join(new_data_path, "metadata3.csv")
        metadata_lines = []

        with open(metadata_path, "r", encoding='utf-8') as metadata_file:
            metadata_lines = metadata_file.readlines()

        for metadata_line in metadata_lines:
            basename, speaker, emotion, text = metadata_line.split('|')
            wav_path = os.path.join(new_data_path, "wavs", speaker, "{}.wav".format(basename))
            lab_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
            # print("==============================") 
            # print("wav_path", wav_path)
            # print("text", text)
            # with open(lab_path, 'r') as f:
            #     print(f.readline())

            # if not os.path.exists(wav_path): continue

            # tg_path = os.path.join( 주석
            #     self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
            # )
            # if os.path.exists(tg_path):
            #     ret = self.process_utterance(speaker, basename, emotion)
            #     if ret is None:
            #         print("ret is None")
            #         continue
            #     else:
            #         info, pitch, energy, n = ret
            #     print("info", info)
            #     out_all.append(info)
            #     if "ASH" in speaker or "CST" in speaker or "BJS" in speaker or "JCH" in speaker or "GJY" in speaker or "CHY" in speaker or "OES" in speaker or "LSY" in speaker:
            #         out_6.append(info)
            # else:
            #     print("TextGrid not exists:", tg_path)
            #     continue

            # if not ("CHY" in speaker or "OES" in speaker or "JCH" in speaker or "LSY" in speaker or "CST" in speaker or "ASH" in speaker or "GJY" in speaker or "BJS" in speaker): # 추가
            #     continue

            ret = self.process_utterance(speaker, basename, emotion)

    
            if ret is None:
                print("ret is None")
                continue
            else:
                
                info, pitch, energy, n = ret
                
            #     info = ret[0]
            # print("info", info)

            
            # out_all.append(info)

            # if "ASH" in speaker or "CST" in speaker or "BJS" in speaker or "JCH" in speaker or "GJY" in speaker or "CHY" in speaker or "OES" in speaker or "LSY" in speaker:
            #     out_6.append(info)

            
            if len(pitch) > 0:
                pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
            if len(energy) > 0:
                energy_scaler.partial_fit(energy.reshape((-1, 1)))

            n_frames += n
            


        # Compute pitch, energy, duration, and mel-spectrogram
        # speakers = {}
        # for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
        #     speakers[speaker] = i
        #     for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
        #         if ".wav" not in wav_name:
        #             continue

        #         basename = wav_name.split(".")[0]
        #         tg_path = os.path.join(
        #             self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        #         )
        #         if os.path.exists(tg_path):
        #             ret = self.process_utterance(speaker, basename)
        #             if ret is None:
        #                 continue
        #             else:
        #                 info, pitch, energy, n = ret
        #             out.append(info)

        #         if len(pitch) > 0:
        #             pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
        #         if len(energy) > 0:
        #             energy_scaler.partial_fit(energy.reshape((-1, 1)))

        #         n_frames += n

        
        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
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
        # with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
        #     f.write(json.dumps(speakers))

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
        random.shuffle(out_6)
        # out_6 = [r for r in out_6 if r is not None]

        # train_set = natsort.natsorted(out_6[self.val_size:])
        # temp_set = out_6[:self.val_size] # 
        # val_set = temp_set[self.test_size:]
        # test_set = temp_set[:self.test_size]

        # # Write metadata
        # with open(os.path.join(self.out_dir, "train_6.txt"), "w", encoding="utf-8") as f:
        #     for m in train_set:
        #         f.write(m + "\n")
        # with open(os.path.join(self.out_dir, "val_6.txt"), "w", encoding="utf-8") as f:
        #     val_set = natsort.natsorted(val_set)
        #     for m in val_set:
        #         f.write(m + "\n")
        # with open(os.path.join(self.out_dir, "test_6.txt"), "w", encoding="utf-8") as f:
        #     test_set = natsort.natsorted(test_set)
        #     for m in test_set:
        #         f.write(m + "\n")

        # with open(os.path.join(self.out_dir, "train_all.txt"), "w", encoding="utf-8") as f:
        #     train_set = natsort.natsorted(out_all)
        #     for m in train_set:
        #         f.write(m + "\n")

        # return out_6
        return None
        

    def process_utterance(self, speaker, basename, emotion):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        dur_path = os.path.join(self.out_dir, "duration", dur_filename)
        
        
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Emotions
        # emotion = wav_path.split('/')[-1][4:7]

        # Get alignments 주석
        # textgrid = tgt.io.read_textgrid(tg_path)
        # phone, tg_duration, start, end = self.get_alignment(
        #     textgrid.get_tier_by_name("phones")
        # )
        # text = "{" + " ".join(phone) + "}"
        # if start >= end:
        #     return None
        
        print("========================")
        # print("textgrid:", tg_path)
        # print("text:", text)
        # print("phone:", list(phone))
        # print("tg_duration:", tg_duration, len(tg_duration), sum(tg_duration))
        # print("start, end:", start, end)

        if not os.path.exists(wav_path):
            print("wav", wav_path, "not exists")
            return None
        
        if not os.path.exists(text_path):
            print("txt", text_path, "not exists")
            return None
        
        if not os.path.exists(dur_path):
            print("dur", dur_path, "not exists")
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        # wav = wav[
        #     int(self.sampling_rate * start) :
        # ].astype(np.float32)
        wav = wav.astype(np.float32)


        duration = list(np.load(dur_path).astype(int))

        print("duration:", duration, len(duration), sum(duration))



        # Read raw text 
        # with open(text_path, "r") as f: 주석
        #     raw_text = f.readline().strip("\n")

        # g2p = G2p()
        # filters = '([.,!?])"'
        # cleaners = ["korean_cleaners"]
        # raw_text = re.sub(re.compile(filters), '', raw_text)
        # raw_text = _clean_text(raw_text, cleaners)
        # raw_text = h2j(g2p(raw_text))

        # phone = np.array([preprocess_korean(raw_text, cleaners)])

        # print("text", raw_text)
        # print("phone", phone)




        # print("raw_text:", raw_text)

        # Compute fundamental frequency
        
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)

        # print("mel shape", mel_spectrogram.shape)

        # mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        # energy = energy[: mel_spectrogram.shape[1]+1]
        # pitch = pitch[: mel_spectrogram.shape[1]+1]
        # energy = energy[: sum(duration)]
        # pitch = pitch[: sum(duration)]


        if self.pitch_phoneme_averaging: 

            # print("original pitch", pitch)
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

            print("revised pitch", pitch, len(pitch))

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # # Save files 주석
        # dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        # np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)

        # np.load(os.path.join(self.out_dir, "pitch", pitch_filename))

        # print("pitch", pitch)


        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        # mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        # np.save(
        #     os.path.join(self.out_dir, "mel", mel_filename),
        #     mel_spectrogram.T,
        # )
        
        return ( #주석
            # "|".join([basename, speaker, emotion, phone[0], raw_text]), 
            None, 
            pitch, energy,
            
            # self.remove_outlier(pitch),
            # self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # print("get_alignment============")
        # print("phones", phones)
        # print("durations", durations)
        
        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        # print("phones", phones)
        # print("durations", durations)

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

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
