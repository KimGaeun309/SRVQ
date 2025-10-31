import os
import csv

def make_all_txt(out_path):
    """
    metadata.csv를 읽어 all.txt를 생성
    형식: basename|speaker|emotion|text|text
    """
    metadata_path = os.path.join(out_path, "metadata.csv")
    all_txt_path = os.path.join(out_path, "all.txt")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"[ERROR] metadata.csv not found at {metadata_path}")

    os.makedirs(out_path, exist_ok=True)

    with open(metadata_path, "r", encoding="utf-8") as f_in, \
         open(all_txt_path, "w", encoding="utf-8") as f_out:

        reader = csv.reader(f_in, delimiter="|")
        lines = list(reader)
        print(f"[INFO] Loaded {len(lines)} lines from metadata.csv")

        for row in lines:
            if len(row) < 4:
                continue

            basename, speaker, emotion, text = [x.strip() for x in row]
            # text 두 번 써서 raw_text 자리 채움
            f_out.write(f"{basename}|{speaker}|{emotion}|{text}|{text}\n")

    print(f"[DONE] all.txt saved at: {all_txt_path}")
    print(f"[INFO] Total samples: {len(lines)}")


if __name__ == "__main__":
    out_path = "/root/mydir/ICASSP2024_FS2-develop/ICASSP2024-FS2-develop/preprocessed_data/esd"
    make_all_txt(out_path)