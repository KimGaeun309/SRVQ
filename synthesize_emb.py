import argparse
import os
import torch
import numpy as np
import json
from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, synth_samples, to_device
from text.korean import tokenize, normalize_nonchar
from text import _clean_text
from g2pk import G2p
from jamo import h2j
from torch.utils.data import DataLoader
from dataset import TextDatasetSingle

def get_style_vector(emotion_weights, style_vector_dir, device):
    final_vec = None
    for emotion, weight in emotion_weights.items():
        vec = np.load(os.path.join(style_vector_dir, f"{emotion}_style.npy"))
        vec = torch.from_numpy(vec).float().to(device)
        final_vec = weight * vec if final_vec is None else final_vec + weight * vec
    return final_vec.unsqueeze(0)  # [1, D]

def preprocess_korean(text):
    g2p = G2p()
    cleaners = ["korean_cleaners"]
    filters = '([.,!?])"'
    text = _clean_text(text, cleaners)
    text = h2j(g2p(text))
    phones = list(filter(lambda p: p != " ", tokenize(text, norm=False)))
    phones = "{" + "}{".join(phones) + "}"
    phones = normalize_nonchar(phones, inference=True).replace("}{", " ")
    return phones, text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--speaker", type=str, default="CHY")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--restore_step", type=str, required=True)
    parser.add_argument("--style_vector_dir", type=str, default="emotion_style_vectors_mode")
    parser.add_argument("--emotion_intensity", type=str, required=True,
                        help='e.g. \'{"neu":0.6, "ang":0.4}\'')
    args = parser.parse_args()

    # Parse intensity
    emotion_weights = json.loads(args.emotion_intensity)

    # Load configs and model
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    model.eval()

    # Build style vector
    style_vec = get_style_vector(emotion_weights, args.style_vector_dir, device)

    # Text preprocessing
    phones, raw_text = preprocess_korean(args.text)
    dataset = TextDatasetSingle(preprocess_config, raw_text, phones, args.speaker, "neu")  # dummy emotion
    loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
    batch = next(iter(loader))
    batch = to_device(batch, device)

    # Forward with manual style vector
    with torch.no_grad():
        output = model(
            *(batch[2:]),
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
            inference=True,
            style_vector=style_vec,
        )
        # Override style_ref_embs with manually built style_vec
        # output = list(output)
        # output[10] = model.style_extract_fc(style_vec)  # [B, D]
        # output = tuple(output)

        synth_samples(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
            os.path.join(train_config["path"]["result_path"], str(args.restore_step)),
            args
        )
