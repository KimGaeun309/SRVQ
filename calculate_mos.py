import os
import pandas as pd

def calculate_sample_statistics(folder_path):
    # 모델 종류
    models = ["GT", "Proposed-abl-egi", "Proposed-abl-clu", "Proposed-abl-upd", "Proposed-ft", "Proposed", "TSP-TTS-upd", "TSP-TTS"]
    # 감정 종류
    emotions = ["ang", "anx", "emb", "hap", "hur", "neu", "sad"]
    
    
    # 샘플별 결과 저장을 위한 딕셔너리 초기화
    sample_results = {}
    
    # 파일 읽기
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if not file_name.endswith(".txt"):  # 텍스트 파일만 처리
            continue
        
        # 파일 형식 확인
        if file_name.startswith("mos_"):
            category = "mos"
            max_lines = 196
        else:
            category = None
            max_lines = 28
            for emo in emotions:
                if file_name.startswith(f"{emo}_"):
                    category = "emos"
                    break
        
        if category:
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    
                    sample_path, score = parts[0], int(parts[1])
                    sample = sample_path.split("/")[2]  # 샘플 이름 추출
                    
                    if sample not in sample_results:
                        sample_results[sample] = {model: {"mos": [], "emos": []} for model in models}
                    
                    for model in models:
                        if model in sample_path:
                            sample_results[sample][model][category].append(score)
                            break
    
    # 통계 계산
    mos_summary = {}
    emos_summary = {}
    model_mos_aggregates = {model: [] for model in models}
    
    for sample, model_scores in sample_results.items():
        mos_summary[sample] = {}
        emos_summary[sample] = {}
        for model, scores in model_scores.items():
            mos_avg = sum(scores["mos"]) / len(scores["mos"]) if scores["mos"] else 0
            emos_avg = sum(scores["emos"]) / len(scores["emos"]) if scores["emos"] else 0
            mos_summary[sample][model] = mos_avg
            emos_summary[sample][model] = emos_avg
            if mos_avg > 0:
                model_mos_aggregates[model].append(mos_avg)
    
    # 데이터프레임 생성
    df_mos = pd.DataFrame.from_dict(mos_summary, orient="index")
    df_emos = pd.DataFrame.from_dict(emos_summary, orient="index")
    
    # 모델별 MOS 평균 계산
    model_mos_means = {model: sum(scores) / len(scores) if scores else 0 for model, scores in model_mos_aggregates.items()}
    df_model_means = pd.DataFrame.from_dict(model_mos_means, orient="index", columns=["Model_MOS_Mean"])
    
    # 결과를 텍스트 파일로 저장
    output_mos_file = os.path.join(folder_path, "Sample_MOS.txt")
    output_emos_file = os.path.join(folder_path, "Sample_EMOS.txt")
    output_model_avg_file = os.path.join(folder_path, "Model_MOS_Average.txt")

    df_mos.to_csv(output_mos_file, sep="\t")
    df_emos.to_csv(output_emos_file, sep="\t")
    df_model_means.to_csv(output_model_avg_file, sep="\t")
    
    print("통계 계산 완료. 결과 파일 저장됨:")
    print("- Sample MOS 평균:", output_mos_file)
    print("- Sample EMOS 평균:", output_emos_file)
    print("- 모델별 MOS 평균:", output_model_avg_file)
    print("\n[모델별 MOS 평균]")
    print(df_model_means)
    
def calculate_modelwise_statistics(folder_path):
    # 모델 종류
    models = ["GT", "Proposed-abl-egi", "Proposed-abl-clu", "Proposed-abl-upd", "Proposed-ft", "Proposed", "TSP-TTS-upd", "TSP-TTS"]
    # 감정 종류
    emotions = ["ang", "anx", "emb", "hap", "hur", "neu", "sad"]

    # --- 기존 MOS 계산 블록 ---
    model_scores = {model: {"mos": []} for model in models}
    model_emotion_scores = {model: {emo: [] for emo in emotions} for model in models}

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not file_name.endswith(".txt"):
            continue

        if file_name.startswith("mos_"):
            max_lines = 1000
        else:
            continue  # emos 및 기타 파일 무시 (아래에서 따로 처리)

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                sample_path, score = parts[0], int(parts[1])

                for model in models:
                    if model in sample_path:
                        model_scores[model]["mos"].append(score)
                        for emo in emotions:
                            if emo in sample_path:
                                model_emotion_scores[model][emo].append(score)
                                break
                        break

    model_mos_means = {
        model: sum(scores["mos"]) / len(scores["mos"]) if scores["mos"] else 0
        for model, scores in model_scores.items()
    }

    model_emotion_means = {
        model: {
            emo: sum(score_list) / len(score_list) if score_list else 0
            for emo, score_list in emo_dict.items()
        }
        for model, emo_dict in model_emotion_scores.items()
    }

    df_model_mos = pd.DataFrame.from_dict(model_mos_means, orient="index", columns=["MOS_Avg"])
    df_model_emos = pd.DataFrame.from_dict(model_emotion_means, orient="index")
    df_combined = pd.concat([df_model_mos, df_model_emos], axis=1)

    output_combined_file = os.path.join(folder_path, "Modelwise_MOS_EmotionSpecific_Averages.txt")
    df_combined.to_csv(output_combined_file, sep="\t")

    print("모델별 MOS 및 감정별 MOS 평균 계산 완료. 결과 파일:", output_combined_file)
    print(df_combined)

    # --- 추가: 감정 prefix로 시작하는 개별 파일들 처리 ---
    print("\n[Emotion MOS]")  # ← 이 라벨로 구분

    emotion_mos_scores = {model: [] for model in models}

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".txt"):
            continue

        if file_name.startswith("mos_") or file_name.startswith("emos_"):
            continue

        if not any(file_name.startswith(f"{emo}_") for emo in emotions):
            continue  # 감정 prefix로 시작하지 않는 파일은 무시

        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            print("file_path", file_path)
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                sample_path, score = parts[0], int(parts[1])

                for model in models:
                    if model in sample_path:
                        emotion_mos_scores[model].append(score)
                        break

    emotion_mos_means = {
        model: sum(scores) / len(scores) if scores else 0
        for model, scores in emotion_mos_scores.items()
    }

    df_emotion_mos = pd.DataFrame.from_dict(emotion_mos_means, orient="index", columns=["Emotion_MOS_Avg"])
    print(df_emotion_mos)


# 사용 예시
# calculate_sample_statistics("/root/mydir/ICASSP2024_FS2-develop/MOS/")  # 폴더 경로를 적절히 변경


# 사용 예시
calculate_modelwise_statistics("/root/mydir/ICASSP2024_FS2-develop/MOS/")
