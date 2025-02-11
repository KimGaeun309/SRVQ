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


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    # sequence = np.array(
    #     text_to_sequence(
    #         phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
    #     )
    # )

    return phones


def synthesize(device, model, args, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
                inference=True,
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                os.path.join(train_config["path"]["result_path"], str(args.restore_step)),
                args,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="CHY",
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default="ang",
        help="emotion for multi-emotion synthesis, for single-sentence mode only (ang, hap, emb, anx, neu, sad, hur)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None 


    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    os.makedirs(
        os.path.join(train_config["path"]["result_path"], str(args.restore_step)), exist_ok=True)

    # Set Device
    torch.manual_seed(train_config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(train_config["seed"])
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device of TTS: {device}")

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )

    if args.mode == "single":
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "emotions.json"
            )
        ) as f:
            emotion_map = json.load(f)

        raw_text = args.text
        emotion = args.emotion
        speaker = args.speaker

        cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        from g2pk import G2p
        from jamo import h2j
        from text import _clean_text

        g2p = G2p()
        filters = '([.,!?])"'
        cleaners = ["korean_cleaners"]
        raw_text = re.sub(re.compile(filters), '', raw_text)
        raw_text = _clean_text(raw_text, cleaners)
        raw_text = h2j(g2p(raw_text))

        phone = preprocess_korean(raw_text, cleaners)

        print("phone", phone)

        dataset = TextDatasetSingle(preprocess_config, raw_text, phone, speaker, emotion)

        batchs = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=dataset.collate_fn,
        )

        # ids = raw_texts = [args.text[:100]]
        # speakers = np.array([args.speaker_id])
        # emotion_id = emotion_map[args.emotion]
        # emotions = np.array([emotion_id])

        # phone = np.array(text_to_sequence(text, cleaners))

        # print("phone", phone)
        
        # texts = np.array([phone])
        # print("final", texts)
        # print("size", texts.shape[1])

        # text_lens = np.array([len(texts[0])])

        # print("text_lens")
        # batchs = [(ids, raw_texts, speakers, emotions, texts, text_lens, max(text_lens))]

        # GST Reference Audio
        # if model_config["gst"]["use_gst"]:
        #     from utils.tools import get_decode_config
        #     decode_config = get_decode_config(args.dataset)
        #     assert os.path.exists(decode_config["path"]["reference_audio"])

        #     reference_mel = np.load(decode_config["path"]["reference_audio"])
        #     reference_mel_len = np.array([len(reference_mel[0])])
        #     batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens), 
        #         reference_mel, reference_mel_len, max(reference_mel_len))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    start = time.time()
    synthesize(device, model, args, configs, vocoder, batchs, control_values)
    end = time.time()
    print(f'Total_time: {end - start}')
    # print(f'time per phoneme: {(end - start)/ texts.shape[1]}')
