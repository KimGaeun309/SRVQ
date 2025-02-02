import os
import re

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text
from g2pk import G2p
from jamo import h2j

def prepare_align(config):
    '''
        in_dir: Input wave path
        out_dir: Save foler preprocessed data
        wav_tag: Wave file tag
        txt_dir: txt dir name
        wav_dir: wav dir name
        sampling_rate: Set sampling rate (default=22050)
        max_wav_value: max wav value (default=32768.0)
        cleaners: Text preprocessing Select
    '''

    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    # Read multi_kr/metadata.csv
    with open(os.path.join(in_dir, "metadata2.csv"), encoding="utf-8") as fr:
        lines = [lines.strip() for lines in fr.readlines()]

    g2p = G2p()
    for line in tqdm(lines, total=len(lines)):
        parts = line.strip().split("|")
        base_name = parts[0]
        spk_folder = parts[1]
        emotion = parts[2]
        text = parts[3]

        # Special Character Extract with filters
        filters = '([.,!?])"'
        text = re.sub(re.compile(filters), '', text)
        text = _clean_text(text, cleaners)
        text = h2j(g2p(text))
        print(text)

        wav_path = os.path.join(in_dir, "wavs", spk_folder, "{}.wav".format(base_name))


        # ffmpeg
        temp_wav_path = wav_path.replace(".wav", "_temp.wav")
        ffmpeg_command = f"ffmpeg -i {wav_path} -ar 22050 -ac 1 {temp_wav_path}"
        os.system(ffmpeg_command)
        mv_command = f"mv {temp_wav_path} {wav_path}"
        os.system(mv_command)

        if os.path.exists(wav_path):
            os.makedirs(os.path.join(out_dir, spk_folder), exist_ok=True)
            wav, _ = librosa.load(wav_path, sr=sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                wav_path,
                sampling_rate,
                wav.astype(np.int16),
            )
            try:
                os.symlink(wav_path, os.path.join(out_dir, spk_folder, "{}.wav".format(base_name)))
            except:
                print(wav_path, "exists")
                continue
            with open(
                os.path.join(out_dir, spk_folder, "{}.lab".format(base_name)),
                "w",
            ) as f1:
                f1.write(text)
