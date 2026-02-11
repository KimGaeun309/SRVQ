import os
import json

import torch
import torch.nn as nn
import numpy as np

from .style_predictor_flow import StylePredictorFlow

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

        
        # neu_path = os.path.join(preprocess_config["path"]["curr_path"], "emotion_style_vectors", "neu_style.npy")
        # neu_vec = np.load(neu_path).astype("float32").squeeze()          # [256]

        
        dim = model_config["residual_vq"]["vq_hidden"] # 256 # * model_config["residual_vq"]["num_rvq"]
        neu_vec = torch.randn(dim) * 0.02                      # 표준편차는 필요시 조정

        # self.neu_base = nn.Parameter(neu_vec, requires_grad=True)  

        # assert neu_vec.ndim == 1
        # assert neu_vec.size == dim
        # 1×256 버퍼로 보관
        self.neu_base = nn.Parameter(neu_vec.unsqueeze(0), requires_grad=True)  # [1, 256*n_rvq]    

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

        # __init__ 안
        sp_cfg = model_config.get("style_predictor", {})
        self.style_predictor = StylePredictorFlow(
            dim_text=model_config["transformer"]["encoder_hidden"],
            dim_tag=(self.emotion_emb.embedding_dim if self.emotion_emb is not None else model_config["transformer"]["encoder_hidden"]),
            dim_neu=model_config["residual_vq"]["vq_hidden"],   # 256
            dim_style=model_config["residual_vq"]["vq_hidden"], # 256
            hidden=sp_cfg.get("hidden", model_config["style_predictor"]["hidden"]),
            dropout=0.1,
            use_transformer_block=False,
            nhead=2,
            nlayers=1,
            noise_k_train=sp_cfg.get("noise_k_train", 0.1),
            noise_k_infer=sp_cfg.get("noise_k_infer", 0.0),
        )


        self.padding_idx = len(symbols) + 1

        self.max_source_positions = 2000
        self.embed_positions = SinusoidalPositionalEmbedding(
            model_config["transformer"]["encoder_hidden"],
            self.padding_idx,
            init_size=self.max_source_positions + self.padding_idx + 1,
        )
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json"), "r") as f:
            emo_map = json.load(f)

        self.neutral_id = int(emo_map["neu"])   # 여기서 4로 확정


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
        ser_embs=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None,
        inference=False,
        intensity=1.0,
        # did_x0_init=False,
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

        # ser_embs: [B,256] expected
        if ser_embs is not None:
            if isinstance(ser_embs, np.ndarray):
                ser_embs = torch.from_numpy(ser_embs)
            ser_embs = ser_embs.to(device=output.device, dtype=output.dtype)
            if ser_embs.dim() == 1:
                ser_embs = ser_embs.unsqueeze(0)  # [1,256]

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

        # B = output.size(0)

        B = output.size(0)
        neu_cond = self.neu_base.detach().expand(B, -1).contiguous()   # predictor/decoder 조건
        neu_base_for_loss = self.neu_base.expand(B, -1).contiguous()   # 손실용(grad 유지)

        # neu_emb = self.neu_base.expand(B, -1).contiguous()  # [B,256]    
        # # neu_emb = self.neu_base.expand(B, -1).to(dtype=output.dtype)  # device는 생략 가능

        # Training mode
        if not inference:
            if ser_embs is None:
                 raise ValueError("ser_embs is required for training (SER target).")
                 
            t_end_train  = float(self.model_config["style_predictor"].get("t_end_train", 1.0))
            steps_train  = int(self.model_config["style_predictor"].get("steps_train", 2))

            # --------------------------
            # (B) Rectified Flow predictor  →  style_pred_embs & flow_loss
            # --------------------------
            neutral_mask = None
            if self.neutral_id is not None:
                neutral_mask = (emotions == self.neutral_id)  # [B]

            style_ref_embs = ser_embs  # [B,256]


            style_pred_embs, flow_loss, soft_zero_loss = self.style_predictor(
                text_enc=output,
                style_tag_emb=style_tag_emb,
                neu_emb=neu_cond,            # [B,256]
                text_mask=text_mask,
                target_style=style_ref_embs.detach(),
                return_loss=True,
                t_end=t_end_train,
                steps=steps_train,
                neutral_mask=neutral_mask,
            )
            orig_style_pred_embs = style_pred_embs

            output = output + style_ref_embs.unsqueeze(1)

            vq_loss, min_encoding_indices, orig_style_ref_embs = None, None, None
            
        # Inference mode
        else:
            style_ref_embs, vq_loss, min_encoding_indices, orig_style_ref_embs = None, None, None, None



            t_end_infer = float(intensity)
            t_end_infer = max(0.0, min(1.0, t_end_infer))  # clamp
            # neu_emb: [B,256] (forward 초반에 만든 것 그대로)
            style_pred_embs = self.style_predictor(
                text_enc=output,
                style_tag_emb=style_tag_emb,
                neu_emb=neu_cond,
                text_mask=text_mask,
                t_end=t_end_infer,
                steps=self.model_config["style_predictor"].get("steps_infer", 1),
            )  # [B,256]

            orig_style_pred_embs = style_pred_embs
            output = output + style_pred_embs.unsqueeze(1)

            flow_loss = torch.tensor(0.0, device=device)
            soft_zero_loss = torch.tensor(0.0, device=device)

        # =========================
    
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
