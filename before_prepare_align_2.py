import os
import glob
import json
import shutil

new_data_path = "/media/gaeun/fee580d1-6954-462b-bdaf-3b8ccc4254311/data/015.Emotion_and_Style_TTS"
wavs_path = os.path.join(new_data_path, "wavs")
metadata_path = os.path.join(new_data_path, "metadata3.csv")
# pathes = glob.glob("/DB/015.Emotion_and_Style_TTS/01.Data/2.Validation/LabelingData/*/1.Emotion/*/*/*.json")
pathes = glob.glob("/media/gaeun/fee580d1-6954-462b-bdaf-3b8ccc4254311/data/015.Emotion_and_Style_TTS/01.Data/*/LabelingData/*/1.Emotion/*/*/*.json")

# print(pathes[1][129:132])

speakers = {"JCH", "OES", "CHY", "LSY", "ASH", "GJY", "BJS", "CST"}

emotions = {}

with open(metadata_path, "w", encoding='utf-8') as metadata_file:
    for speaker in speakers:
        for path in pathes:
            lab_path = path       

            base_name = os.path.basename(lab_path).replace(".json", "")
            
            with open(lab_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            text = json_data["전사정보"]["TransLabelText"]

            number_of_speaker = json_data["기본정보"]["NumberOfSpeaker"]

            if base_name[16:19] != speaker:
                continue

            spk_folder = base_name[16:19] + "_" + number_of_speaker

            # emotion = lab_path[129:132]
            emotion = json_data["화자정보"]["Emotion"].lower()[:3]
            print(emotion)

            if emotion not in emotions:
                emotions[emotion] = 1
            else:
                emotions[emotion] += 1


            metadata_line = f"{base_name}|{spk_folder}|{emotion}|{text}\n"
            metadata_file.write(metadata_line)


print(emotions)
# {'anx': 8281, 'hur': 7969, 'neu': 7955, 'emb': 7992, 'hap': 7986, 'ang': 8250, 'sad': 7987}
# {'hur': 13943, 'anx': 16270, 'neu': 15916, 'emb': 13983, 'hap': 11977, 'ang': 16206, 'sad': 15965}