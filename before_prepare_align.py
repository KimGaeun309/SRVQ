import os
import glob
import json
import shutil

new_data_path = "/media/gaeun/fee580d1-6954-462b-bdaf-3b8ccc425431/data/015.Emotion_and_Style_TTS"
wavs_path = os.path.join(new_data_path, "wavs")
metadata_path = os.path.join(new_data_path, "metadata.csv")
# pathes = glob.glob("/DB/015.Emotion_and_Style_TTS/01.Data/2.Validation/LabelingData/*/1.Emotion/*/*/*.json")
pathes = glob.glob("/media/gaeun/fee580d1-6954-462b-bdaf-3b8ccc425431/data/015.Emotion_and_Style_TTS/01.Data/1.Training/LabelingData/*/1.Emotion/*/*/*.json")


# with open(metadata_path, "a", encoding='utf-8') as metadata_file:


for path in pathes:
    lab_path = path       
    wav_path = lab_path.replace("LabelingData/TL", "SourceData/TS").replace("LabelingData/VL", "SourceData/VS").replace(".json", ".wav") 

    if not os.path.exists(wav_path): 
        print("wav path not exists : ", wav_path)
        continue

    base_name = os.path.basename(lab_path).replace(".json", "")
    
    with open(lab_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    text = json_data["전사정보"]["TransLabelText"]

    number_of_speaker = json_data["기본정보"]["NumberOfSpeaker"]

    spk_folder = base_name[16:19] + "_" + number_of_speaker

        # emotion = lab_path[78:81]

        # metadata_line = f"{base_name}|{spk_folder}|{emotion}|{text}\n"
        # metadata_file.write(metadata_line)


    # 데이터 이동
    new_wav_dir = os.path.join(wavs_path, spk_folder)
    new_wav_path = os.path.join(new_wav_dir, f"{base_name}.wav")


    if not os.path.exists(new_wav_dir):
        os.makedirs(new_wav_dir)


    # ffmpeg를 사용하여 샘플링 속도와 채널 개수 변경
    temp_wav_path = new_wav_path.replace(".wav", "_temp.wav")
    ffmpeg_command = f"ffmpeg -i {wav_path} -ar 22050 -ac 1 {temp_wav_path}"
    os.system(ffmpeg_command)


        # 임시 파일을 최종 경로로 이동
    shutil.move(temp_wav_path, new_wav_path)
    print(f"Moved and converted {wav_path} to {new_wav_path} with 22050Hz sampling rate and 1 channel")

    # 원본 파일 삭제
    os.remove(wav_path)
    print(f"Converted {wav_path} to {new_wav_path} with 22050Hz sampling rate and 1 channel and deleted the original file")




        
