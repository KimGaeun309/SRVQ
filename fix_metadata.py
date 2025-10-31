import os
import csv

def fix_metadata_esd(metadata_path):
    fixed_path = metadata_path.replace(".csv", "_fixed.csv")

    with open(metadata_path, "r", encoding="utf-8") as fin, \
         open(fixed_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin, delimiter="|")
        writer = csv.writer(fout, delimiter="|")

        count = 0
        for row in reader:
            if len(row) < 4:
                continue

            wav_field, speaker, emotion, text = row

            # basename 추출: "wavs/0011/0011_001410.wav" -> "0011_001410"
            base = os.path.splitext(os.path.basename(wav_field))[0]

            # emotion 첫 글자만 대문자
            emotion = emotion.capitalize()

            writer.writerow([base, speaker, emotion, text])
            count += 1

    print(f"[DONE] Saved fixed metadata to: {fixed_path}")
    print(f"[INFO] Total processed lines: {count}")


if __name__ == "__main__":
    metadata_path = "/root/mydir/ICASSP2024_FS2-develop/ICASSP2024-FS2-develop/preprocessed_data/esd/metadata.csv"  # 수정
    fix_metadata_esd(metadata_path)