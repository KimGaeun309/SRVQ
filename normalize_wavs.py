import os
import glob
import json
import numpy as np
import soundfile as sf
import librosa
import pyworld as pw
from tqdm import tqdm
import audio as Audio

###############################################################################
# 전역 설정
###############################################################################
SAMPLE_RATE = 22050

# 원본 wav 경로 (하위 폴더까지 포함된 wav들)
AUDIO_GLOB_PATH = "/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/raw_wavs_dhs/*/*.wav"

# 통계(평균 f0, 평균 에너지) 저장 폴더/파일
STATS_DIR = "./normalized_data/stats"
os.makedirs(STATS_DIR, exist_ok=True)

AVERAGE_F0_JSON = os.path.join(STATS_DIR, "average_f0.json")
AVERAGE_ENERGY_JSON = os.path.join(STATS_DIR, "average_energy.json")

# 결과 저장 폴더: 3가지 버전을 각각 저장
PITCH_ONLY_DIR = "./normalized_data/pitch_only"
ENERGY_ONLY_DIR = "./normalized_data/energy_only"
DURATION_ONLY_DIR = "./normalized_data/duration_only"

for d in [PITCH_ONLY_DIR, ENERGY_ONLY_DIR, DURATION_ONLY_DIR]:
    os.makedirs(d, exist_ok=True)

###############################################################################
# 0. 멜 스펙트로그램 계산 함수 (librosa 예시)
###############################################################################
def compute_mel_spectrogram(wav, sr, n_fft=1024, hop_length=256, n_mels=80,
                            fmin=0, fmax=None):
    """
    librosa 기반으로 멜 스펙트로그램을 계산한 뒤 (n_mels, T) shape로 반환.
    필요 시 본인 프로젝트의 STFT/멜 함수로 교체.
    """
    if fmax is None:
        fmax = sr // 2

    # # STFT
    # D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, window='hann')
    # # 파워 스펙트럼 (magnitude)
    # mag = np.abs(D)
    # # 멜 필터 적용
    # mel_filter = librosa.filters.mel(sr, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    # mel_spec = mel_filter @ mag

    # # dB 변환 (필요 시 옵션 조정)
    # mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    fs = SAMPLE_RATE
    STFT = Audio.stft.TacotronSTFT(1024, 256, 1024, 80, fs, 0, 8000)

    mel, _ = Audio.tools.get_mel_from_wav(wav, STFT)
    
    return mel # shape=(n_mels, frames)s

###############################################################################
# 1. 전체 음성에 대한 평균 F0, 평균 에너지(유성 구간 한정) 계산
###############################################################################
def pass1_collect_stats():
    """
    모든 wav 파일을 순회하며, 유성 구간만 골라 f0, 에너지를 누적 후 평균을 구합니다.
    에너지는 각 프레임별로 (스펙트럼 합)을 구하고, 그 중 유성 구간만 사용합니다.
    """
    all_wav_files = sorted(glob.glob(AUDIO_GLOB_PATH))

    sum_f0 = 0.0
    count_f0 = 0
    sum_energy = 0.0
    count_energy = 0

    for wav_path in tqdm(all_wav_files, desc="[PASS1] Collect Stats", ncols=80):
        # 1) 음성 로드
        x, sr = sf.read(wav_path)
        if sr != SAMPLE_RATE:
            x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # 2) WORLD 분석
        f0, t = pw.harvest(x, sr)
        sp = pw.cheaptrick(x, f0, t, sr)
        # ap = pw.d4c(x, f0, t, sr)  # 에너지 계산엔 ap 불필요

        # 3) 유성 구간만 추출
        voiced_idx = (f0 > 0)

        if np.any(voiced_idx):
            sum_f0 += np.sum(f0[voiced_idx])
            count_f0 += np.count_nonzero(voiced_idx)

            # 에너지 (프레임별로 스펙트럼 합)
            frame_energy = np.sum(sp, axis=1)  # shape=(frames,)
            voiced_energy = frame_energy[voiced_idx]
            sum_energy += np.sum(voiced_energy)
            count_energy += len(voiced_energy)

    # 전역 평균 f0, 에너지
    global_f0_avg = sum_f0 / count_f0 if count_f0 > 0 else 0.0
    global_energy_avg = sum_energy / count_energy if count_energy > 0 else 0.0

    with open(AVERAGE_F0_JSON, 'w') as f:
        json.dump({"f0_global_avg": global_f0_avg}, f, indent=2)
    with open(AVERAGE_ENERGY_JSON, 'w') as f:
        json.dump({"energy_global_avg": global_energy_avg}, f, indent=2)

    print(f"=== [PASS1] Done ===\n"
          f" > f0_global_avg = {global_f0_avg:.3f}\n"
          f" > energy_global_avg = {global_energy_avg:.3f}")

###############################################################################
# 2. 합성 유틸 함수들
###############################################################################
def pitch_only_synthesis(x, sr, global_energy_avg, global_f0_avg):
    """
    [피치만 강조 + 파일별 f0 정규화]
    => 1) 이 파일의 평균 f0를 구하고, 
       2) 글로벌 평균 f0에 맞춰 스케일링.
       3) 에너지는 전역 에너지로 평탄화(유성구간 한정)
    """
    f0, t = pw.harvest(x, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)

    # 유성 구간
    voiced_idx = (f0 > 0)

    # (1) 로컬 평균 f0 (이 음성 파일의 평균)
    if np.any(voiced_idx):
        mean_f0_local = np.mean(f0[voiced_idx])
    else:
        mean_f0_local = 0.0  # 혹시 무성음만 있다면 0 처리

    # (2) f0 스케일링: mean_f0_local -> global_f0_avg
    f0_modified = f0.copy()
    if mean_f0_local > 0:
        scale = global_f0_avg / mean_f0_local
        f0_modified[voiced_idx] *= scale
    # 무성구간은 그대로 0

    # (3) 에너지 평탄화 (유성 구간만)
    frame_energy = np.sum(sp, axis=1)  # 각 프레임 에너지
    scaling_factor = np.ones_like(frame_energy)
    scaling_factor[voiced_idx] = global_energy_avg / (frame_energy[voiced_idx] + 1e-8)
    sp_modified = sp * scaling_factor[:, None]

    # WORLD 재합성
    wav_out = pw.synthesize(f0_modified, sp, ap, sr)
    return wav_out

def energy_only_synthesis(x, sr, global_f0_avg):
    """
    [에너지만 강조] => 에너지는 원본 유지, 피치는 전역평균으로 평탄화
    (무성구간은 0으로 둠)
    """
    f0, t = pw.harvest(x, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)

    # 1) f0 평탄화
    f0_modified = f0.copy()
    voiced_idx = (f0_modified > 0)
    f0_modified[voiced_idx] = global_f0_avg
    f0_modified[~voiced_idx] = 0.0

    # 2) sp, ap는 원본 그대로
    sp_modified = sp
    ap_modified = ap

    # 3) WORLD 재합성
    wav_out = pw.synthesize(f0_modified, sp_modified, ap_modified, sr)
    return wav_out

def duration_only_synthesis(x, sr, global_f0_avg, global_energy_avg, stretch_factor=1.5):
    """
    [길이만 강조] => 피치/에너지는 모두 평탄화, 길이(프레임 수)를 stretch_factor배로 변경
    - stretch_factor: 1.0이면 변화 없음, 1.5면 1.5배 길이로
    - 단순히 프레임을 반복(늘이기) 또는 샘플링(줄이기)하는 방식
      => 보컬 품질이 떨어질 수 있으므로, 실제로는 PSOLA 등 권장
    """
    f0, t = pw.harvest(x, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)

    # 1) f0 평탄화
    f0_flat = f0.copy()
    voiced_idx = (f0_flat > 0)
    f0_flat[voiced_idx] = global_f0_avg
    f0_flat[~voiced_idx] = 0.0

    # 2) 에너지 평탄화 (유성구간만)
    frame_energy = np.sum(sp, axis=1)
    scaling_factor = np.ones_like(frame_energy)
    scaling_factor[voiced_idx] = global_energy_avg / (frame_energy[voiced_idx] + 1e-8)
    sp_flat = sp * scaling_factor[:, None]

    ap_flat = ap  # ap는 그대로 두되, 프레임수 늘릴때 같이 처리를 해야 함

    # 3) 프레임 수 늘리기/줄이기
    def stretch_frames(arr_2d, factor):
        """
        arr_2d.shape = (T, dim) 형태인 WORLD 파라미터(sp, ap 등)를
        간단히 '반올림된 프레임 반복'으로 늘리거나 줄이는 함수.
        """
        # 늘릴 총 프레임 수 결정
        T = arr_2d.shape[0]
        new_T = int(np.round(T * factor))
        # 새 배열 준비
        stretched = np.zeros((new_T, arr_2d.shape[1]), dtype=arr_2d.dtype)

        # 간단히 '최근린 관계'로 매핑
        # (ex. index i -> 원본의 round(i/factor))
        for i in range(new_T):
            orig_i = int(np.round(i / factor))
            orig_i = min(orig_i, T-1)  # 범위 초과 방지
            stretched[i] = arr_2d[orig_i]
        return stretched

    def stretch_1d(arr_1d, factor):
        """
        f0처럼 shape=(T,) 인 경우
        """
        T = arr_1d.shape[0]
        new_T = int(np.round(T * factor))
        stretched = np.zeros((new_T,), dtype=arr_1d.dtype)
        for i in range(new_T):
            orig_i = int(np.round(i / factor))
            orig_i = min(orig_i, T-1)
            stretched[i] = arr_1d[orig_i]
        return stretched

    # 실제 늘리기
    f0_stretched = stretch_1d(f0_flat, stretch_factor)
    sp_stretched = stretch_frames(sp_flat, stretch_factor)
    ap_stretched = stretch_frames(ap_flat, stretch_factor)

    # 4) 재합성
    wav_out = pw.synthesize(f0_stretched, sp_stretched, ap_stretched, sr)
    return wav_out

###############################################################################
# 3. 최종 실행: 파일별로 3가지 버전 생성
###############################################################################
def pass2_synthesize_variants():
    """
    - pass1에서 저장된 전역 평균 f0/에너지 정보를 로드
    - 각 wav 파일마다 3가지 버전 (pitch_only, energy_only, duration_only) 합성
      * pitch_only에서는 (이 파일의 평균 f0) → (글로벌 평균 f0)로 스케일링
    """
    # 1) 전역 평균값 로드
    with open(AVERAGE_F0_JSON, 'r') as f:
        f0_info = json.load(f)
    global_f0_avg = f0_info["f0_global_avg"]

    with open(AVERAGE_ENERGY_JSON, 'r') as f:
        energy_info = json.load(f)
    global_energy_avg = energy_info["energy_global_avg"]

    # 2) 대상 wav 파일
    all_wav_files = sorted(glob.glob(AUDIO_GLOB_PATH))

    for wav_path in tqdm(all_wav_files, desc="[PASS2] Generate 3-versions", ncols=80):
        # (1) 원본 읽기
        x, sr = sf.read(wav_path)
        if sr != SAMPLE_RATE:
            x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        # ─────────────────────────────────────────────────────────────────────
        # (2) [피치만 강조 + 파일별 f0 정규화]
        wav_pitch = pitch_only_synthesis(
            x, sr,
            global_energy_avg=global_energy_avg,
            global_f0_avg=global_f0_avg
        )
        pitch_out_wav = os.path.join(PITCH_ONLY_DIR, f"{base_name}_pitch.wav")
        sf.write(pitch_out_wav, wav_pitch, sr)

        mel_pitch = compute_mel_spectrogram(wav_pitch, sr)
        pitch_out_mel = os.path.join(PITCH_ONLY_DIR, f"{base_name}_pitch.npy")
        np.save(pitch_out_mel, mel_pitch)

        # ─────────────────────────────────────────────────────────────────────
        # (3) [에너지만 강조: 원본 코드 그대로]
        wav_energy = energy_only_synthesis(x, sr, global_f0_avg)
        energy_out_wav = os.path.join(ENERGY_ONLY_DIR, f"{base_name}_energy.wav")
        sf.write(energy_out_wav, wav_energy, sr)

        mel_energy = compute_mel_spectrogram(wav_energy, sr)
        energy_out_mel = os.path.join(ENERGY_ONLY_DIR, f"{base_name}_energy.npy")
        np.save(energy_out_mel, mel_energy)

        # ─────────────────────────────────────────────────────────────────────
        # (4) [길이만 강조: 원본 코드 그대로]
        wav_duration = duration_only_synthesis(
            x, sr,
            global_f0_avg=global_f0_avg,
            global_energy_avg=global_energy_avg,
            stretch_factor=1.5
        )
        duration_out_wav = os.path.join(DURATION_ONLY_DIR, f"{base_name}_dur.wav")
        sf.write(duration_out_wav, wav_duration, sr)

        mel_duration = compute_mel_spectrogram(wav_duration, sr)
        duration_out_mel = os.path.join(DURATION_ONLY_DIR, f"{base_name}_dur.npy")
        np.save(duration_out_mel, mel_duration)

        print(f"[OK] {base_name} => pitch/energy/dur 3가지 버전 생성 완료.")

###############################################################################
# 메인 실행
###############################################################################
if __name__ == "__main__":
    # 1) 전체 파일 통계 추출 (유성구간 기반 f0, 에너지)
    # pass1_collect_stats()

    # 2) 3가지 버전 음성 + 멜 스펙트로그램 생성
    pass2_synthesize_variants()
