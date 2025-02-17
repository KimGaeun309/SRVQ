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
# from text import text_to_sequence
from sklearn.metrics import silhouette_score

def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def synthesize(device, model, batchs):
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            output = model(
                *(batch[2:]),
                inference=True,
            )
            print(output[1].shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help="name of dataset")
    parser.add_argument("--source", type=str, default='preprocessed_data/emo_kr_22050/test.txt', help="path to a source file with format like train.txt and val.txt, for batch mode only")
    args = parser.parse_args()

    device = torch.device("cpu")
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)

    path = 'preprocessed_data/emo_kr_22050/test.txt'
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
    perplexities = []

    # emotions= []
    mels_torch = []
    emotions_torch = []


    for i in range(len(file_path_list)):
        file_path = file_path_list[i]
        emotion = torch.tensor(emotions[i], device=device).unsqueeze(0)
        mel = np.load(f'preprocessed_data/emo_kr_22050/mel/{file_path[:3]}-mel-{file_path}.npy')
        # mel = torch.from_numpy(mel).float().to(device).unsqueeze(0)
        emotions_torch.append(emotion)
        mels_torch.append(mel)

    
    emotions_torch = torch.cat(emotions_torch, dim=0)
    mels_torch = torch.from_numpy(pad_2D(mels_torch)).float().to(device) 

            # style 추출
    ref_embs, cls_loss = model.ref_enc(mels_torch, emotions_torch) #, pitch_mel, energy_mel)
    styles, _, _, codebooks, perplexities = model.style_extractor(ref_embs, cls_loss)
    

    # # Perplexity 계산
    #         with torch.no_grad():
    #             e_mean = torch.mean(F.one_hot(indices, num_classes=layer.n_e).float(), dim=0)
    #             perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
    #             perplexities.append(perplexity)

    
    perplexity = torch.mean(torch.stack(perplexities))

    print("perplexities", perplexities)
    print("perplexity :", perplexity)

    # print("styles", styles.shape)

    # print("emotions_torch", emotions_torch.shape)

    print("silhouette_score :", silhouette_score(styles.detach().numpy(), emotions_torch.detach().numpy()))



        # styles.append(style.cpu().data[:, :])
        # perplexities.append(perplexity)





    # emotions = np.array(emotions)
    # styles = torch.cat(styles, dim=0)

    # colors = ['red', 'blue', 'green', 'yellow', 'brown', 'indigo', 'black']
    # labels = ['ang', 'anx', 'emb', 'hap', 'hur', 'neu', 'sad']

    # data_x_1 = styles[:, 0:256].numpy()
    # data_x_2 = styles[:, 256:512].numpy()
    # data_x_3 = styles[:, 512:].numpy()
    # data_x_4 = styles[:, :].numpy()

    # def run_tsne(data, perplexity=20, n_iter=2000):
    #     if data.shape[1] == 0:
    #         return None
    #     tsne_model = TSNE(n_components=2, random_state=0, init='random', perplexity=perplexity, n_iter=n_iter)
    #     return tsne_model.fit_transform(data)

    # tsne_1 = run_tsne(data_x_1)
    # tsne_2 = run_tsne(data_x_2)
    # tsne_3 = run_tsne(data_x_3)
    # tsne_4 = run_tsne(data_x_4)

    # fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # def scatter_tsne(ax, tsne_data, data_y, title=False):
    #     if tsne_data is None:
    #         ax.set_title(f"{title} - Not Available")
    #         ax.grid(True)
    #         return
    #     for i, (c, label) in enumerate(zip(colors, labels)):
    #         ax.scatter(tsne_data[data_y==i, 0], tsne_data[data_y==i, 1], c=c, label=label, alpha=0.5)
    #     if title:
    #         ax.set_title(title)
    #     ax.grid(False)
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    # scatter_tsne(axes[0, 0], tsne_1, emotions)
    # scatter_tsne(axes[0, 1], tsne_2, emotions)
    # scatter_tsne(axes[1, 0], tsne_3, emotions)
    # scatter_tsne(axes[1, 1], tsne_4, emotions)

    # axes[0, 0].legend(loc='best', fontsize=8)

    # labels_pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # labels_text = ["(a)", "(b)", "(c)", "(d)"]
    # for (i, j), text in zip(labels_pos, labels_text):
    #     axes[i, j].set_xlabel(text, fontsize=12, labelpad=10)

    # plt.tight_layout()
    # plt.savefig('tsne_combined.png', dpi=300)
    # plt.close()

    # for idx, (tsne_data, title) in enumerate(zip([tsne_1, tsne_2, tsne_3, tsne_4], ["VQ1", "VQ2", "VQ3", "RVQ"])):
    #     fig, ax = plt.subplots(figsize=(5, 5))
    #     scatter_tsne(ax, tsne_data, emotions, title)
    #     ax.legend(loc='best', fontsize=8)
    #     plt.savefig(f'tsne_plot_{idx+1}.png', dpi=300)
    #     plt.close()


