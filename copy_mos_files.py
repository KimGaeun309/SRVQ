
import os

base_dir = "/root/mydir/MOS_RVQ"

directories = [
    "wavs/Proposed", "wavs/Proposed-abl-clu", "wavs/Proposed-abl-egi", "wavs/Proposed-abl-upd",
    "wavs/TSP-TTS", "wavs/TSP-TTS-upd", "wavs/GT"
]


filelist_path = "/root/mydir/MOS_RVQ/filelist.txt"

output_file = os.path.join(base_dir, "mos_files.txt")



# 감정별 파일명 매칭
emotion_files = {
    "ang": "emotion1_ang.txt",
    "anx": "emotion2_anx.txt",
    "emb": "emotion3_emb.txt",
    "hap": "emotion4_hap.txt",
    "hur": "emotion5_hur.txt",
    "neu": "emotion6_neu.txt",
    "sad": "emotion7_sad.txt",
}

# 감정별 파일 저장소 초기화
emotion_paths = {key: [] for key in emotion_files}
all_file_paths = []

# filelist.txt에서 파일 읽기
with open(filelist_path, "r") as f:
    filenames = [line.strip() for line in f.readlines()]

# 각 디렉토리에서 파일 경로 생성
for directory in directories:
    for filename in filenames:
        relative_path = os.path.join(directory, filename)
        all_file_paths.append(relative_path)
        
        # 감정별 파일 경로 저장
        for emotion, emotion_file in emotion_files.items():
            if emotion in filename:
                emotion_paths[emotion].append(relative_path)

# mos_files.txt 저장
with open(output_file, "w") as f:
    f.writelines(f"{path}\n" for path in all_file_paths)

# 감정별 파일 저장
for emotion, filename in emotion_files.items():
    emotion_path = os.path.join(base_dir, filename)
    with open(emotion_path, "w") as f:
        f.writelines(f"{path}\n" for path in emotion_paths[emotion])

print("모든 파일이 성공적으로 생성되었습니다.")




# # 경로 설정
# ORIGINAL_PATH = "/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/raw_wavs_dhs"
# DESTINED_PATH = "/root/mydir/MOS_RVQ/wavs/GT"

# # 파일 리스트 불러오기
# filelist_path = "/root/mydir/MOS_RVQ/filelist.txt"


# with open(filelist_path, "r") as f:
#     filelist = [line.strip() for line in f.readlines()]


# # 복사 실행
# for filename in filelist:
#     source = os.path.join(ORIGINAL_PATH, f"{filename[:3]}/{filename}")
#     destination = os.path.join(DESTINED_PATH, filename)
#     os.system(f'cp "{source}" "{destination}"')

# print("파일 복사가 완료되었습니다.")
