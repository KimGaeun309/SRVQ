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
    style_embs = []
    for file_path in file_path_list:
        print("file_path", file_path)
        # Mel
        mel = np.load(f'preprocessed_data/emo_kr_22050/mel/{file_path[:3]}-mel-{file_path}.npy')
        mel = torch.from_numpy(mel).float().to(device)
        mel = mel.unsqueeze(0)

        # style 추출
        style, _, _, codebooks = model.style_extractor(mel)
        styles.append(style.cpu().data[:, :])  # 전체 임베딩 (예: 768 차원 등)

        style_emb = model.style_extract_fc(style)
        style_embs.append(style_emb.cpu().data[:, :])

    emotions = np.array(emotions)
    styles = torch.cat(styles, dim=0)
    style_embs = torch.cat(style_embs, dim=0)
    # print("styles size:", styles.size())

    # 색상 및 라벨 정의 (감정마다 구분)
    colors = ['red','blue','green','yellow','brown','indigo','black']
    labels = ['ang','anx','emb','hap','hur','neu','sad']

    # # 1) [:,:256]
    # data_x_1 = styles[:, :256].numpy()
    # # 2) [:,256:512]
    # data_x_2 = styles[:, 256:512].numpy()
    # # 3) [:,512:]
    # data_x_3 = styles[:, 512:].numpy() if styles.size(1) > 512 else None
    # # 4) [:,:]
    # data_x_4 = styles[:, :].numpy()

    data_x_1 = styles[:, :].numpy()
    data_x_2 = style_embs[:, :].numpy()
    data_x_3 = styles[:, 0:256].numpy()
    data_x_4 = styles[:, 256:512].numpy()
    data_x_5 = styles[:, 512:].numpy()

    
    def run_tsne(data, perplexity=20, n_iter=2000):
        if data.shape[1] == 0:
            return None
        tsne_model = TSNE(n_components=2, random_state=0, init='random', perplexity=perplexity, n_iter=n_iter)
        return tsne_model.fit_transform(data)

    tsne_1 = run_tsne(data_x_1)
    tsne_2 = run_tsne(data_x_2)
    tsne_3 = run_tsne(data_x_3)
    tsne_4 = run_tsne(data_x_4)
    tsne_5 = run_tsne(data_x_5)


    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    def scatter_tsne(ax, tsne_data, data_y, title=False):
        if tsne_data is None:
            ax.set_title(f"{title} - Not Available")
            ax.grid(True)
            return
        for i, (c, label) in enumerate(zip(colors, labels)):
            ax.scatter(tsne_data[data_y==i, 0], tsne_data[data_y==i, 1], c=c, label=label, alpha=0.5)
        if title:
            ax.set_title(title)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    scatter_tsne(axes[0, 0], tsne_1, emotions)
    scatter_tsne(axes[0, 1], tsne_2, emotions)
    scatter_tsne(axes[1, 0], tsne_3, emotions)
    scatter_tsne(axes[1, 1], tsne_4, emotions)
    scatter_tsne(axes[1, 2], tsne_5, emotions)

    axes[0, 0].legend(loc='best', fontsize=8)

    labels_pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels_text = ["(a)", "(b)", "(c)", "(d)"]
    for (i, j), text in zip(labels_pos, labels_text):
        axes[i, j].set_xlabel(text, fontsize=12, labelpad=10)

    plt.tight_layout()
    plt.savefig('tsne_combined.png', dpi=300)
    plt.close()

    for idx, (tsne_data, title) in enumerate(zip([tsne_1, tsne_2, tsne_3, tsne_4], ["RVQ", "Emb", "VQ1", "VQ2"])):
        fig, ax = plt.subplots(figsize=(5, 5))
        scatter_tsne(ax, tsne_data, emotions, title)
        # ax.legend(loc='best', fontsize=8)
        plt.savefig(f'tsne_plot_{idx+1}.png', dpi=300)
        plt.close()