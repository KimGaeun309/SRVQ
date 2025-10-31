import os
import glob
import json
import shutil
import re
import subprocess

def prepare_align(config):
    """
    Prepares ESD English dataset for MFA alignment:
    - Resamples from 16kHz → 22.05kHz (if needed)
    - Generates metadata.csv
    - Creates cleaned .lab files for MFA (lowercase + no punctuation)
    """

    esd_root = config["path"]["corpus_path"]
    out_root = config["path"]["preprocessed_path"]
    speaker_json = os.path.join(out_root, "speakers.json")
    emotion_json = os.path.join(out_root, "emotions.json")

    # Load speaker and emotion dicts
    with open(speaker_json, "r") as f:
        speaker_dict = json.load(f)
    with open(emotion_json, "r") as f:
        emotion_dict = json.load(f)

    # Output directories
    wavs_out = os.path.join(out_root, "wavs")
    os.makedirs(wavs_out, exist_ok=True)

    metadata_path = os.path.join(out_root, "metadata.csv")
    metadata_file = open(metadata_path, "w", encoding="utf-8")

    print(f"[INFO] Speakers to process: {list(speaker_dict.keys())}")

    # Traverse speakers
    for spk_id in speaker_dict.keys():
        spk_dir = os.path.join(esd_root, spk_id)
        if not os.path.isdir(spk_dir):
            print(f"[WARN] Speaker {spk_id} not found, skipping.")
            continue

        text_path = os.path.join(spk_dir, f"{spk_id}.txt")
        if not os.path.exists(text_path):
            print(f"[WARN] Missing transcript: {text_path}")
            continue

        # Load transcript
        with open(text_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        transcript_dict = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            utt_id = parts[0]
            emotion = parts[-1].capitalize()
            text = " ".join(parts[1:-1])

            # === Clean text for MFA ===
            # Remove unwanted characters (keep only a-z, 0-9, and space)
            text = text.lower()
            text = re.sub(r"[^a-z0-9' ]+", "", text).strip()  # remove .,!? etc.

            transcript_dict[utt_id] = (text, emotion)

        # Process each emotion folder
        for emo_folder in os.listdir(spk_dir):
            emo_path = os.path.join(spk_dir, emo_folder)
            if not os.path.isdir(emo_path):
                continue
            if emo_folder not in emotion_dict:
                continue

            wav_list = glob.glob(os.path.join(emo_path, "*.wav"))
            wav_list.sort()

            spk_out_dir = os.path.join(wavs_out, spk_id)
            os.makedirs(spk_out_dir, exist_ok=True)

            for wav_path in wav_list:
                base = os.path.basename(wav_path).replace(".wav", "")
                if base not in transcript_dict:
                    continue

                text, emotion = transcript_dict[base]
                emo_label = emotion.lower()

                new_wav_path = os.path.join(spk_out_dir, f"{base}.wav")
                lab_path = os.path.join(spk_out_dir, f"{base}.lab")

                # # 1. Check sample rate
                # try:
                #     sr = subprocess.check_output(
                #         ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate",
                #          "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
                #         universal_newlines=True
                #     ).strip()
                # except Exception:
                #     sr = "0"

                # if sr != "22050":
                #     temp_wav_path = new_wav_path.replace(".wav", "_temp.wav")
                #     ffmpeg_command = f'ffmpeg -y -i "{wav_path}" -ar 22050 -ac 1 "{temp_wav_path}"'
                #     os.system(ffmpeg_command)
                #     shutil.move(temp_wav_path, new_wav_path)
                # else:
                #     shutil.copy2(wav_path, new_wav_path)

                # 2. Write .lab (cleaned, lowercase, newline ensured)
                clean_text = re.sub(r"[^a-zA-Z ]+", "", text).lower().strip()
                with open(lab_path, "w", encoding="utf-8") as f_lab:
                    f_lab.write(clean_text + "\n")
                    
                # 3. Write metadata
                rel_wav = os.path.relpath(new_wav_path, out_root)
                metadata_file.write(
                    f"{rel_wav}|{spk_id}|{emo_label}|{text}\n"
                )

    metadata_file.close()
    print(f"[DONE] metadata.csv and .lab files ready for MFA under {wavs_out}")