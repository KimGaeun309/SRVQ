import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .transformers.transformer import Encoder, Decoder #, MelDecoder, LightMelDecoder
from .transformers.layers import PostNet
from .modules import VarianceAdaptor, SinusoidalPositionalEmbedding
from utils.tools import get_mask_from_lengths
from text.symbols import symbols

from typing import Optional

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config) # 수정. (TSP-TTS 없애기)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

        

        self.emotion_emb = None
        if model_config["multi_emotion"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "emotions.json"
                ),
                "r",
            ) as f:
                n_emotion = len(json.load(f))
            self.emotion_emb = nn.Embedding(
                n_emotion,
                model_config["transformer"]["encoder_hidden"],
            )

        hidden_dim = model_config["transformer"]["encoder_hidden"]

        self.padding_idx = len(symbols) + 1

        self.max_source_positions = 2000
        self.embed_positions = SinusoidalPositionalEmbedding(
            model_config["transformer"]["encoder_hidden"],
            self.padding_idx,
            init_size=self.max_source_positions + self.padding_idx + 1,
        )

        self.sf_scale = 1.0

    def forward(
        self,
        speakers,
        emotions,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        intensities=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        
        inference=False,
    ):
        
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        # Add emotion category condition

        emo = self.emotion_emb(emotions)
        emo = self.sf_scale * emo
        output = output + emo.unsqueeze(1).expand(
            -1, max_src_len, -1
        )

        # Variance Adaptor
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        # # Decoder
        # if self.model_config["residual_vq"]["num_rvq"] == 3:
        #     output, mel_masks = self.decoder(output, mel_masks, codebooks) # vq3
        # else:
        #     output, mel_masks = self.decoder(output, mel_masks) # vq2, vq4

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        # Post-net
        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
