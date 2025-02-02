
import os 

wav_data_path = "./raw_data/emo_kr_22050/"
mel_data_path = "./preprocessed_data/emo_kr_22050/mel"

old_text_files = ["./preprocessed_data/emo_kr_22050/train1.txt", "./preprocessed_data/emo_kr_22050/val1.txt", "./preprocessed_data/emo_kr_22050/test1.txt"]

new_text_files = ["./preprocessed_data/emo_kr_22050/train.txt", "./preprocessed_data/emo_kr_22050/val.txt", "./preprocessed_data/emo_kr_22050/test.txt"]


for i in range(3):
    new_lines = []
    with open(old_text_files[i], 'r') as f:
        for old_line in f.readlines():
            basename, speaker_folder, emotion, tg, txt = old_line.split('|')
            wav_path = os.path.join(wav_data_path, speaker_folder, "{}.wav".format(basename))
            mel_path = os.path.join(mel_data_path, f"{speaker_folder}-mel-{basename}.npy")
            if os.path.exists(wav_path) == False: 
                print("wav path not exist", basename)
                continue
            if os.path.getsize(wav_path) == 0: 
                print("wav size is 0", basename)
                continue
            if os.path.exists(mel_path) == False: 
                print("mel path not exist", mel_path)
                continue
            if os.path.getsize(mel_path) == 0: 
                print("mel size is 0", mel_path)
                continue
            new_lines.append(old_line)

    with open(new_text_files[i], 'w') as f:
        for new_line in new_lines:
            f.write(new_line)

