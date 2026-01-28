import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from .text2style_aligner import Text2Style_Aligner

# # Flow style predictor
# from .style_predictor import StylePredictor, LinearNorm
from .style_predictor_flow import StylePredictorFlowMultiStage
from .style_predictor import LinearNorm
from .transformers.transformer import Encoder, Decoder #, MelDecoder, LightMelDecoder
from .transformers.layers import PostNet
from .modules import VarianceAdaptor, SinusoidalPositionalEmbedding
from utils.tools import get_mask_from_lengths
from text.symbols import symbols
# from .residual_vq_gaeun import ReferenceEncoderSRVQ3, SRVQ3WithNeutralization # 수정
from .residual_vq_gaeun import ReferenceEncoder_cls, ResidualVQ_kmeans
# from .residuaal_vq import SRVQPyworld, ResidualVQ

from .gst.style_encoder import StyleEncoder, GST_VQ

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
        # GST
        if model_config["gst"]["use_gst"]:
            self.gst = StyleEncoder(
                idim=model_config["gst"]["n_mel_channels"],
                gst_tokens=model_config["gst"]["gst_tokens"],
                gst_token_dim=model_config["gst"]["gst_token_dim"],
                gst_heads=model_config["gst"]["gst_heads"],
                conv_layers=model_config["gst"]["gst_conv_layers"],
                conv_chans_list=model_config["gst"]["gst_conv_chans_list"],
                conv_kernel_size=model_config["gst"]["gst_conv_kernel_size"],
                conv_stride=model_config["gst"]["gst_conv_stride"],
                gru_layers=model_config["gst"]["gst_gru_layers"],
                gru_units=model_config["gst"]["gst_gru_units"],
            )

        # GST_VQ
        if model_config["gst"]["use_gst_vq"]:
            self.gst_vq = GST_VQ(
                idim=model_config["gst"]["n_mel_channels"],
                gst_tokens=model_config["gst"]["gst_tokens"],
                gst_token_dim=model_config["gst"]["gst_token_dim"],
                gst_heads=model_config["gst"]["gst_heads"],
                conv_layers=model_config["gst"]["gst_conv_layers"],
                conv_chans_list=model_config["gst"]["gst_conv_chans_list"],
                conv_kernel_size=model_config["gst"]["gst_conv_kernel_size"],
                conv_stride=model_config["gst"]["gst_conv_stride"],
                gru_layers=model_config["gst"]["gst_gru_layers"],
                gru_units=model_config["gst"]["gst_gru_units"],
                vq_n_e=n_speaker+n_emotion,
            )

        self.padding_idx = len(symbols) + 1

        self.max_source_positions = 2000
        self.embed_positions = SinusoidalPositionalEmbedding(
            model_config["transformer"]["encoder_hidden"],
            self.padding_idx,
            init_size=self.max_source_positions + self.padding_idx + 1,
        )
        self.neutral_id: Optional[int] = sp_cfg.get("neutral_id", None)


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
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None,
        inference=False,
        intensity=1.0,
        did_x0_init=False,
        # pitch_mel=None,
        # energy_mel=None,
        # init_flag=False,
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

        # =========================
        # Style module (fusion predictor)
        # =========================

        # style tag / speaker emb 준비
        style_tag_emb = self.emotion_emb(emotions) if self.emotion_emb is not None else None  # [B, D_tag]
        # spk_emb = self.speaker_emb(speakers) if self.speaker_emb is not None else None        # [B, D_spk] or None

        # text mask (True=pad). src_lens 기준으로 정확히 생성
        B, T, _ = output.shape
        device = output.device
        ar = torch.arange(T, device=device).unsqueeze(0)     # [1,T]
        text_mask = ar >= src_lens.unsqueeze(1)              # [B,T]  True=pad

        guided_loss_1 = torch.tensor(0.0, device=device)     # cross-attn 대신 


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

        # Loss
        guided_loss = guided_loss_1
        attn_emo_list = None

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
            style_ref_embs,
            style_pred_embs,
            guided_loss,
            vq_loss,
            flow_loss, # Edit!
            soft_zero_loss, # Edit!
            min_encoding_indices,
            orig_style_ref_embs, # Edit!
            neu_base_for_loss, # Edit!
            orig_style_pred_embs,
        )
