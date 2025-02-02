import os

preprocessed_path = "./preprocessed_data/emo_kr_22050/"
datas = ["train_all.txt", "train_4.txt", "train_6.txt", "test_4.txt", "test_6.txt", "val_4.txt", "val_6.txt", "train_val_test.txt", "train_val.txt"]


for data in datas:
    new_lines = []
    with open(os.path.join(preprocessed_path, data), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            basename, speaker, emotion, phoneme, text = line.split('|')
            mel_path = os.path.join(
                preprocessed_path,
                "mel",
                "{}-mel-{}.npy".format(speaker, basename),
            )
            if os.path.exists(mel_path):
                new_lines.append(line)



    with open(os.path.join(preprocessed_path, data), 'w', encoding='utf-8') as f2:
        for line in new_lines:
            f2.write(line)