import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import pandas as pd

from dataset import TextDataset
from utils.model import get_model
from utils.tools import get_configs_of, to_device
from text import text_to_sequence

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def synthesize(device, model, batchs):
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                inference=True,
            )
            print(output[1].shape)

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

    path =  '/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/preprocessed_data/emo_kr_22050/test.txt'
    with open(path, encoding='utf-8') as f:
        infos = [line.strip().split("|") for line in f]

    with open("preprocessed_data/emo_kr_22050/emotions.json") as f:
        emotion_map = json.load(f)

    file_path_list = []
    emotions = []
    for info in infos:
        file_path_list.append(info[0])
        emotions.append(emotion_map[info[2]])

    model = get_model(args, configs, device, train=False)

    styles = []
    for i in range(len(file_path_list)):
        file_path = file_path_list[i]
        emotion = torch.tensor(emotions[i], device=device).unsqueeze(0)
        print("file_path", file_path)
        # Mel
        mel = np.load(f'preprocessed_data/emo_kr_22050/mel/{file_path[:3]}-mel-{file_path}.npy')
        mel = torch.from_numpy(mel).float().to(device).unsqueeze(0)

        pitch_path = f"/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/normalized_data/pitch_only/{file_path}_pitch.npy"
        energy_path = f"/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/normalized_data/energy_only/{file_path}_energy.npy"
            
        pitch_mel = torch.from_numpy(np.load(pitch_path).T).to(device).unsqueeze(0)
        energy_mel = torch.from_numpy(np.load(energy_path).T).to(device).unsqueeze(0)

        # style 추출
        z_mel, z_pitch, z_energy, cls_loss = model.ref_enc(mel, emotion, pitch_mel, energy_mel)
        style, _, _, codebooks = model.style_extractor(z_mel, z_pitch, z_energy, cls_loss)

        styles.append(style.cpu().data[:, :])  # 전체 임베딩 (예: 768 차원 등)

    emotions = np.array(emotions)
    styles = torch.cat(styles, dim=0)
    print("styles size:", styles.size())

    # 색상 및 라벨 정의 (감정마다 구분)
    colors = ['red','blue','green','yellow','brown','indigo','black']
    labels = ['ang','anx','emb','hap','hur','neu','sad']

    # 슬라이스 구간 (768차원 기준)
    # 6개 구간: 각각 128차원씩
    # 총 768 == 6 * 128
    data_x_1 = styles[:, 0:128].numpy()
    data_x_2 = styles[:, 128:256].numpy()
    data_x_3 = styles[:, 256:384].numpy()
    data_x_4 = styles[:, 384:512].numpy()
    data_x_5 = styles[:, 512:640].numpy()
    data_x_6 = styles[:, 640:768].numpy()
    data_x_7 = styles[:, :].numpy()  # 전체

    data_x_8 = model.style_extract_fc(styles)

    def run_tsne(data, perplexity=20, n_iter=2000):
        if data.shape[1] == 0:
            return None  # 혹시 slice 구간이 잘못되어 0차원이 되면 무시
        tsne_model = TSNE(n_components=2, random_state=0, init='random',
                          perplexity=perplexity, n_iter=n_iter)
        return tsne_model.fit_transform(data)

    tsne_1 = run_tsne(data_x_1)
    tsne_2 = run_tsne(data_x_2)
    tsne_3 = run_tsne(data_x_3)
    tsne_4 = run_tsne(data_x_4)
    tsne_5 = run_tsne(data_x_5)
    tsne_6 = run_tsne(data_x_6)
    tsne_7 = run_tsne(data_x_7)

    # 2x4 서브플롯 생성 (7개 쓰고, 1개는 비우는 형태)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Helper to scatter-plot
    def scatter_tsne(ax, tsne_data, data_y, title):
        if tsne_data is None:
            ax.set_title(f"{title} - Not Available")
            ax.grid(True)
            return
        for i, (c, label) in enumerate(zip(colors, labels)):
            ax.scatter(tsne_data[data_y==i, 0],
                       tsne_data[data_y==i, 1],
                       c=c, label=label, alpha=0.5)
        ax.set_title(title)
        ax.grid(True)

    # 각 subplot에 그리기
    scatter_tsne(axes[0, 0], tsne_1, emotions, "[:, 0:128]")
    scatter_tsne(axes[0, 1], tsne_2, emotions, "[:, 128:256]")
    scatter_tsne(axes[0, 2], tsne_3, emotions, "[:, 256:384]")
    scatter_tsne(axes[0, 3], tsne_4, emotions, "[:, 384:512]")
    scatter_tsne(axes[1, 0], tsne_5, emotions, "[:, 512:640]")
    scatter_tsne(axes[1, 1], tsne_6, emotions, "[:, 640:768]")
    scatter_tsne(axes[1, 2], tsne_7, emotions, "[:, :] (full)")

    # 마지막 subplot은 비어있음
    axes[1, 3].set_visible(False)

    # 범례: 첫 번째 subplot에만 예시로 추가(필요 시 다른 방식으로 조정 가능)
    axes[0, 0].legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig('fig.png', dpi=300)
    plt.close()


"""import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import pandas as pd

from dataset import TextDataset
from utils.model import get_model
from utils.tools import get_configs_of, to_device
from text import text_to_sequence


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def synthesize(device, model, batchs):
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                inference=True,
            )
            print(output[1].shape)

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

    path =  '/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/preprocessed_data/emo_kr_22050/test.txt'
    with open(path, encoding='utf-8') as f:
        infos = [line.strip().split("|") for line in f]

    with open("preprocessed_data/emo_kr_22050/emotions.json") as f:
        emotion_map = json.load(f)

    file_path_list = []
    emotions = []
    for info in infos:
        file_path_list.append(info[0])
        emotions.append(emotion_map[info[2]])

    model = get_model(args, configs, device, train=False)

    styles = []
    # for file_path in file_path_list:
    for i in range(len(file_path_list)):
        file_path = file_path_list[i]
        emotion = torch.tensor(emotions[i], device=device).unsqueeze(0)
        print("file_path", file_path)
        # Mel
        mel = np.load(f'preprocessed_data/emo_kr_22050/mel/{file_path[:3]}-mel-{file_path}.npy')
        mel = torch.from_numpy(mel).float().to(device)
        mel = mel.unsqueeze(0)

        pitch_path = f"/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/normalized_data/pitch_only/{file_path}_pitch.npy"
        energy_path = f"/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/normalized_data/energy_only/{file_path}_energy.npy"
            
        pitch_mel = torch.from_numpy(np.load(pitch_path).T).to(device).unsqueeze(0)
        energy_mel = torch.from_numpy(np.load(energy_path).T).to(device).unsqueeze(0)

        # style 추출
        # style, _, _, codebooks = model.style_extractor(mel, emotion, pitch_mel, energy_mel)

        z_mel, z_pitch, z_energy, cls_loss = model.ref_enc(mel, emotion, pitch_mel, energy_mel)
        style, _, _, codebooks = model.style_extractor(z_mel, z_pitch, z_energy, cls_loss)

        styles.append(style.cpu().data[:, :])  # 전체 임베딩 (예: 768 차원 등)

    emotions = np.array(emotions)
    styles = torch.cat(styles, dim=0)
    print("styles size:", styles.size())

    # 색상 및 라벨 정의 (감정마다 구분)
    colors = ['red','blue','green','yellow','brown','indigo','black']
    labels = ['ang','anx','emb','hap','hur','neu','sad']

    # 1) [:,:256]
    data_x_1 = styles[:, :256].numpy()
    # 2) [:,256:512]
    data_x_2 = styles[:, 256:512].numpy()
    # 3) [:,512:]
    data_x_3 = styles[:, 512:].numpy() if styles.size(1) > 512 else None
    # 4) [:,:]
    data_x_4 = styles[:, :].numpy()

    # tsne 함수: 부분 임베딩 데이터가 None이 아닐 때만 실행
    def run_tsne(data, perplexity=20, n_iter=2000):
        tsne_model = TSNE(n_components=2, random_state=0, init='random',
                          perplexity=perplexity, n_iter=n_iter)
        return tsne_model.fit_transform(data)

    tsne_1 = run_tsne(data_x_1)
    tsne_2 = run_tsne(data_x_2)

    tsne_3 = None
    if data_x_3 is not None and data_x_3.shape[1] > 0:
        tsne_3 = run_tsne(data_x_3)

    tsne_4 = run_tsne(data_x_4)

    # 2x2 서브플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Helper to scatter-plot
    def scatter_tsne(ax, tsne_data, data_y, title):
        for i, (c, label) in enumerate(zip(colors, labels)):
            ax.scatter(tsne_data[data_y==i, 0],
                       tsne_data[data_y==i, 1],
                       c=c, label=label, alpha=0.5)
        ax.set_title(title)
        ax.grid(True)

    # (1) top-left: [:,:256]
    scatter_tsne(axes[0, 0], tsne_1, emotions, "[:,:256]")
    # (2) top-right: [:,256:512]
    scatter_tsne(axes[0, 1], tsne_2, emotions, "[:,256:512]")

    # (3) bottom-left: [:,512:]
    if tsne_3 is not None:
        scatter_tsne(axes[1, 0], tsne_3, emotions, "[:,512:]")
    else:
        axes[1, 0].set_title("[:,512:] - Not Available")
        axes[1, 0].grid(True)

    # (4) bottom-right: [:,:]
    scatter_tsne(axes[1, 1], tsne_4, emotions, "[:,:]")

    # 범례는 맨 위 subplot 하나에만 추가(혹은 모든 곳에 추가하려면 각각 호출)
    axes[0, 0].legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig('fig.png', dpi=300)
    plt.close()
"""