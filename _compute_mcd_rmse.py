import re
import os
import argparse
from string import punctuation
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style
from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, synth_samples
from dataset import TextDataset, TextDatasetSingle
from text import text_to_sequence
from text.korean import tokenize, normalize_nonchar
import time
import json
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw
from scipy.stats import pearsonr


import re
import os
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, synth_samples
from dataset import TextDataset, TextDatasetSingle
from text import text_to_sequence
from text.korean import tokenize, normalize_nonchar

import time
import json

import os
import math
import glob
import librosa
import pyworld
import pysptk
import numpy as np
import matplotlib.pyplot as plot


SAMPLING_RATE = 22050
FRAME_PERIOD = 5.0

# ORIGINAL_PATH ='/content/in male voice Bengali and Odia/R1.Regional_Dataset_Male_Voice_in_Bengali' #copy the path of the folder that contains the .wav files of the oriinal voice
# SYNTHESIZED_PATH = '/content/in male voice Bengali and Odia/converted_Bengali' #copy the path of the folder that contains the .wav files of the generated voice


def load_wav(wav_file, sr):
    
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)

    return wav


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preprocess_korean(text, cleaners):
    # lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    words = filter(None, re.split(r"([,;.\-\?\!\s+])", text))
    for w in words:
        # if w in lexicon:
        #     phones += lexicon[w]
        # else:
        phones += list(filter(lambda p: p != " ", tokenize(w, norm=False)))
    phones = "{" + "}{".join(phones) + "}"
    phones = normalize_nonchar(phones, inference=True)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    # sequence = np.array(
    #     text_to_sequence(
    #         phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
    #     )
    # )

    return phones

# 평가 지표 계산 함수들
def compute_mcd(mel_true, mel_pred):
    # MCD(Mel Cepstral Distortion) 계산
    log_mel_true = np.log(mel_true + 1e-6)
    log_mel_pred = np.log(mel_pred + 1e-6)
    mcd = np.sqrt(np.mean((log_mel_true - log_mel_pred) ** 2))
    return mcd

def compute_cfsd(mel_true, mel_pred):
    # CFSD(Cepstral Feature Separation Distance) 계산
    return np.linalg.norm(mel_true - mel_pred)

def compute_log_f0_rmse(f0_true, f0_pred):
    # Log-F0 RMSE 계산
    return np.sqrt(mean_squared_error(np.log(f0_true + 1e-6), np.log(f0_pred + 1e-6)))

def compute_duration_dtw(duration_true, duration_pred):
    # Duration DTW 계산
    distance, _ = fastdtw(duration_true, duration_pred, dist=euclidean)
    return distance

def compute_energy_pcc(energy_true, energy_pred):
    # Energy PCC (Pearson Correlation Coefficient) 계산
    return pearsonr(energy_true, energy_pred)[0]


def MCD(x, y):
    log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    diff = x - y
    
    return log_spec_dB_const * math.sqrt(np.inner(diff, diff))


def MCEP(wavfile, mcep_target_directory, alpha=0.65, fft_size=512, mcep_size=24):

    
    if not os.path.exists(mcep_target_directory):
        os.makedirs(mcep_target_directory)

    loaded_wav_file = load_wav(wavfile, sr=SAMPLING_RATE)


    _, spectral_envelop, _ = pyworld.wav2world(loaded_wav_file.astype(np.double), fs=SAMPLING_RATE,
                                frame_period=FRAME_PERIOD, fft_size=fft_size)

    
    mcep = pysptk.sptk.mcep(spectral_envelop, order=mcep_size, alpha=alpha, maxiter=0,
                        etype=1, eps=1.0E-8, min_det=0.0, itype=3)

    fname = os.path.basename(wavfile).split('.')[0]
    np.save(os.path.join(mcep_target_directory, fname + '.npy'),
            mcep,
            allow_pickle=False)

def mcd_cal(mcep_org_files, mcep_synth_files, MCD):
    min_cost_tot = 0.0
    total_frames = 0
    
    for i in mcep_org_files:
        x=0
        for j in mcep_synth_files:
            
            split_org_file,  split_synth_file = os.path.basename(i).split('_'), os.path.basename(j).split('_')
            org_speaker, org_emo, org_speaker_id = split_org_file[0], split_org_file[1], split_org_file[-1]
            synth_speaker, synth_emo, synth_speaker_id = split_synth_file[0], split_synth_file[1], split_synth_file[-1]
            
            x+=1
            if org_speaker==synth_speaker and org_speaker_id==synth_speaker_id and org_emo == synth_emo:
                
                org_mcep_npy=np.load(i)
            
                frame_no = len(org_mcep_npy)
                synth_mcep_npy = np.load(j)
                
                min_cost, _ = librosa.sequence.dtw(org_mcep_npy[:, 1:].T, synth_mcep_npy[:, 1:].T, 
                                                metric=MCD)
    
                min_cost_tot += np.mean(min_cost)
                
                total_frames += frame_no
                
                #print(j,"    ",i,"     ",x,"   ",min_cost_tot,"   ",total_frames)
                
    mcd = min_cost_tot/total_frames
    return mcd, total_frames

# 기존 코드 유지...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    parser.add_argument(
        "--source",
        type=str,
        default='preprocessed_data/emo_kr_22050/test.txt',
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)

    path = 'preprocessed_data/emo_kr_22050/test.txt'
    with open(path, encoding='utf-8') as f:
        infos = [line.strip().split("|") for line in f]

    with open("preprocessed_data/emo_kr_22050/emotions.json") as f:
        emotion_map = json.load(f)

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
        speaker_map = json.load(f)

    file_path_list = []
    speakers = []
    emotions = []
    texts = []
    phonemes = []
    for info in infos:
        file_path_list.append(info[0])
        speakers.append(info[1])  # speaker를 인덱스로 변환
        emotions.append(info[2])
        phonemes.append(info[3])
        texts.append(info[4])

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    # batchs = []

    
    result_file = os.path.join(train_config["path"]["result_path"], f"{args.restore_step}_evaluation.txt")


    f0s_org = []
    energies_org = []
    durations_org = []

    f0s_synth = []
    energies_synth = []
    durations_synth = []

    for i in range(len(file_path_list)):
        file_path = file_path_list[i]

        # Mel, Pitch, Energy, Duration 데이터 로드
        mel = np.load(f'preprocessed_data/emo_kr_22050/mel/{file_path[:3]}-mel-{file_path}.npy')
        pitch = np.load(f'preprocessed_data/emo_kr_22050/pitch/{file_path[:3]}-pitch-{file_path}.npy')
        energy = np.load(f'preprocessed_data/emo_kr_22050/energy/{file_path[:3]}-energy-{file_path}.npy')
        duration = np.load(f'preprocessed_data/emo_kr_22050/duration/{file_path[:3]}-duration-{file_path}.npy')

        mel_true = torch.from_numpy(mel).float().to(device).unsqueeze(0)
        pitch_true = torch.from_numpy(pitch).float().to(device).unsqueeze(0)
        energy_true = torch.from_numpy(energy).float().to(device).unsqueeze(0)
        duration_true = torch.from_numpy(duration).float().to(device).unsqueeze(0)

        f0s_org.append(pitch_true)
        energies_org.append(energy_true)
        durations_org.append(duration_true)

        # target_wav_path = f"/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/raw_wavs_dhs/{speakers[i]}/{file_path}.wav"

        # target_wav_path_2 = os.path.join(train_config["path"]["result_path"], "targ_wavs", f"{file_path}.wav")

        # os.system(f"cp {target_wav_path} {target_wav_path_2}")

        # target_wav_path = target_wav_path_2

        dataset = TextDatasetSingle(preprocess_config, texts[i], phonemes[i], speakers[i], emotions[i])

        batchs = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=dataset.collate_fn,
        )
        
    # 평가 실행
    # evaluate_with_metrics(device, model, args, configs, vocoder, batchs, control_values=(1.0, 1.0, 1.0))

        mel_pred, pitch_pred, energy_pred, duration_pred, pred_wav_path = None, None, None, None, None
        preprocess_config, model_config, train_config = configs
        pitch_control, energy_control, duration_control = (1.0, 1.0, 1.0)
    

        for batch in batchs:
            batch = to_device(batch, device)

            with torch.no_grad():
                # Forward pass
                output = model(
                    *(batch[2:]),
                    p_control=pitch_control,
                    e_control=energy_control,
                    d_control=duration_control,
                    inference=True,
                )

                pred_wav_path = os.path.join(train_config["path"]["result_path"], str(args.restore_step), f"{file_path}.wav")

                # mel_true = batch[7].cpu().numpy()  # mel-spectrogram
                mel_pred = output[1].cpu().numpy()  # model output mel-spectrogram

                # pitch_true = batch[10].cpu().numpy()  # pitch
                pitch_pred = output[2].cpu().numpy()  # model output pitch

                # energy_true = batch[11].cpu().numpy()  # energy
                energy_pred = output[3].cpu().numpy()  # model output energy

                # duration_true = batch[12].cpu().numpy()  # duration
                duration_pred = output[5].cpu().numpy()  # model output duration

                f0s_synth.append(pitch_pred)

                energies_synth.append(energy_pred)

                durations_synth.append(duration_pred)



            
    alpha = 0.65  
    fft_size = 512
    mcep_size = 24

    ORIGINAL_PATH = os.path.join(train_config["path"]["result_path"], "targ_wavs")
    SYNTHESIZED_PATH = os.path.join(train_config["path"]["result_path"], str(args.restore_step))

    dir_org_speech_wav = glob.glob(ORIGINAL_PATH+'/*.wav')
    dir_org_speech_mcep = ORIGINAL_PATH+'/mceps_trg'
    dir_converted_speech_wav = glob.glob(SYNTHESIZED_PATH+'/*.wav')
    dir_converted_speech_mcep =SYNTHESIZED_PATH+'/mceps_conv'

    for wav in dir_org_speech_wav:
        MCEP(wav, dir_org_speech_mcep, fft_size=fft_size, mcep_size=mcep_size)

    for wav in dir_converted_speech_wav:
        MCEP(wav, dir_converted_speech_mcep, fft_size=fft_size, mcep_size=mcep_size)


            
    org_file = glob.glob(ORIGINAL_PATH+'/mceps_trg/*.npy')
    synth_file= glob.glob(SYNTHESIZED_PATH+'/mceps_conv/*.npy')

    cost_function = MCD

    mcd, frames_used = mcd_cal(org_file, synth_file, cost_function)

    
    print(f' MCD = {mcd} dB and total of frames {frames_used}')



    min_cost_tot=[]
    for i in range(len(f0s_org)):
        frame_len=0
        def logf0_rmse(x, y): # method to calculate cost
            #y=pad_to(y,len(x))
            log_spec_dB_const = 1/len(frame_len)
            # print(y)
            diff = x - y
            # print(x,"  ",y,"  ",len(y))
            # print(diff)
            #print(log_spec_dB_const * math.sqrt(np.inner(diff, diff)))
            return log_spec_dB_const * math.sqrt(np.inner(diff, diff))
        
        
        if len(f0s_org[i])<len(f0s_synth[i]):
            frame_len=f0s_org[i]
        else:
            frame_len=f0s_synth[i]

        cost_function = logf0_rmse
        min_cost, _ = librosa.sequence.dtw(f0s_org[i][:].T, f0s_synth[i][:].T, 
                                                        metric=cost_function)
        #print(len(min_cost))
        
        min_cost_tot.append(np.mean(min_cost))

    F0RMSE=sum(min_cost_tot)/len(min_cost_tot)
    print(f"F0_RMSE = {F0RMSE}")



    # # 결과 저장
    with open(result_file, 'a') as result_f:
        result_f.write("\nOverall Averages\n")
        result_f.write("=====================================\n")
        result_f.write(f"MCD: {mcd:.4f}\n")
    #     result_f.write(f"Average CFSD: {average_cfsd:.4f}\n")
        result_f.write(f"F0 RMSE: {F0RMSE:.4f}\n")
    #     result_f.write(f"Average Duration DTW: {average_duration_dtw:.4f}\n")
    #     result_f.write(f"Average Energy PCC: {average_energy_pcc:.4f}\n")
    #     result_f.write("=====================================\n")

    print("평가 완료! 결과는 저장된 evaluation.txt 파일을 확인하세요.")
                
