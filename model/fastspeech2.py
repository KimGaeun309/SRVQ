import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .text2style_aligner import Text2Style_Aligner

# # Flow style predictor
# from .style_predictor import StylePredictor, LinearNorm
from .style_predictor_flow import StylePredictorFlowMultiStage
from .style_predictor import LinearNorm
from .transformers.transformer import Encoder, Decoder, MelDecoder, LightMelDecoder
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
        if model_config["residual_vq"]["num_rvq"] == 3:
            self.decoder = MelDecoder(model_config) # vq3
        else:
            # self.decoder = Decoder(model_config) # vq2, vq4
            pass
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

        
        dim = model_config["residual_vq"]["vq_hidden"] * model_config["residual_vq"]["num_rvq"]
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

        # Style module
        self.ref_enc = ReferenceEncoder_cls(
            idim=model_config["residual_vq"]["n_mel_channels"],
            conv_layers=model_config["residual_vq"]["rvq_conv_layers"],
            conv_chans_list=model_config["residual_vq"]["rvq_conv_chans_list"],
            conv_kernel_size=model_config["residual_vq"]["rvq_conv_kernel_size"],
            conv_stride=model_config["residual_vq"]["rvq_conv_stride"],
            gru_layers=model_config["residual_vq"]["rvq_gru_layers"],
            gru_units=model_config["residual_vq"]["rvq_gru_units"],
            e_dim=model_config["residual_vq"]["vq_hidden"],
        )
        self.style_extractor = ResidualVQ_kmeans(
            n_e=n_emotion,
            e_dim=model_config["residual_vq"]["vq_hidden"],
            num_vq=model_config["residual_vq"]["num_rvq"],
        )

        self.style_extract_fc = LinearNorm(
            model_config["residual_vq"]["vq_hidden"]*model_config["residual_vq"]["num_rvq"],
            model_config["residual_vq"]["vq_hidden"]
        )
        self.style_pred_fc = LinearNorm(
            model_config["residual_vq"]["vq_hidden"]*model_config["residual_vq"]["num_rvq"],
            model_config["residual_vq"]["vq_hidden"]
        )

        # __init__ 안
        sp_cfg = model_config.get("style_predictor", {})
        self.style_predictor = StylePredictorFlowMultiStage(
            n_stages=model_config["residual_vq"]["num_rvq"],
            dim_text=model_config["transformer"]["encoder_hidden"],
            dim_tag=(self.emotion_emb.embedding_dim if self.emotion_emb is not None else model_config["transformer"]["encoder_hidden"]),
            dim_neu=(model_config["residual_vq"]["vq_hidden"]),
            dim_style=model_config["residual_vq"]["vq_hidden"],
            hidden=sp_cfg.get("hidden", model_config["style_predictor"]["hidden"]),
            dropout=0.1,
            use_transformer_block=False,
            nhead=2,
            nlayers=1,
            # ▼ 새 파라미터(없으면 기본값)
            noise_k_train=sp_cfg.get("noise_k_train", 0.1),
            noise_k_infer=sp_cfg.get("noise_k_infer", 0.0),
        )

        self.cross_attn = Text2Style_Aligner(
            num_layers=2,
            hidden_size=256,
        )

        self.text2style_alignment = Text2Style_Aligner(
            num_layers=2,
            hidden_size=256,
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

        # B = output.size(0)

        B = output.size(0)
        neu_cond = self.neu_base.detach().expand(B, -1).contiguous()   # predictor/decoder 조건
        neu_base_for_loss = self.neu_base.expand(B, -1).contiguous()   # 손실용(grad 유지)

        # neu_emb = self.neu_base.expand(B, -1).contiguous()  # [B,256]    
        # # neu_emb = self.neu_base.expand(B, -1).to(dtype=output.dtype)  # device는 생략 가능


        if not inference:
            # --------------------------
            # Style extractor (레퍼런스 기반 RVQ) - 기존 그대로
            # --------------------------
            if self.model_config["gst"]["use_gst"]:
                ref_embs = self.gst(mels)
                style_ref_embs, vq_loss, min_encoding_indices, codebooks = self.style_extractor(ref_embs, p_targets=p_targets, d_targets=d_targets, e_targets=e_targets)
            else:
                
                # style_ref_embs, vq_loss, min_encoding_indices, codebooks = self.style_extractor(mels, emotions=emotions)
                ref_embs, cls_loss = self.ref_enc(mels, emotions=emotions)
                # if init_flag:
                #     # kmeans_init !!!!
                #     # self.style_extractor.vq_layers[0].init_codebook_kmeans(ref_embs)
                #     cls_loss = None
                style_ref_embs, vq_loss, min_encoding_indices, codebooks = self.style_extractor(ref_embs, cls_loss) 
                # style_ref_embs, vq_loss, min_encoding_indices, codebooks = self.style_extractor(mels, p_targets=p_targets, d_targets=d_targets, e_targets=e_targets)

                # if init_flag:
                #     self.style_extractor.vq_layers[1].init_codebook_kmeans(ref_embs - style_ref_embs[:, :256])
                #     self.style_extractor.vq_layers[2].init_codebook_kmeans(ref_embs - style_ref_embs[:, :256] - style_ref_embs[:, 256:512])
                    

            orig_style_ref_embs = style_ref_embs


            t_end_train  = float(self.model_config["style_predictor"].get("t_end_train", 1.0))
            steps_train  = int(self.model_config["style_predictor"].get("steps_train", 2))

            # # 기본 타깃은 style_ref_embs
            # target_style_for_flow = style_ref_embs.detach().to(output.dtype)

            # # neutral이면 neu_emb로 치환
            # if self.neutral_id is not None:
            #     neutral_mask = (emotions == self.neutral_id)  # [B]
            #     if neutral_mask.any():
            #         # float dtype 일치
            #         neu = neu_emb.to(style_ref_embs.dtype)
            #         # style_ref_embs: [B, 256 * n_stages] 에 대해 복사
            #         style_ref_embs = style_ref_embs.clone()
            #         style_ref_embs[neutral_mask] = neu[neutral_mask]



            # --------------------------
            # (B) Rectified Flow predictor  →  style_pred_embs & flow_loss
            # --------------------------
            neutral_mask = None
            if self.neutral_id is not None:
                neutral_mask = (emotions == self.neutral_id)  # [B]

            
            style_pred_embs, flow_loss, soft_zero_loss = self.style_predictor(
                text_enc=output,                 # [B,T,256]
                style_tag_emb=style_tag_emb,     # [B,256]
#                spk_emb=spk_emb,                 # [B,256] or None
                neu_emb=neu_cond,            # [D]  ← 추가: x0로 사용
                text_mask=text_mask,             # [B,T] bool
                target_style=style_ref_embs.detach(),  # x1 supervision: [B,256]
                return_loss=True,
                t_end=t_end_train,
                steps=steps_train,
                neutral_mask=neutral_mask,
            )

            orig_style_pred_embs = style_pred_embs

            

            # # RVQ stage 수에 맞게 복제 
            # if self.model_config["residual_vq"]["num_rvq"] == 4:
            #     style_pred_embs = torch.cat([style_pred_embs, style_pred_embs, style_pred_embs, style_pred_embs], dim=1) # vq4
            # elif self.model_config["residual_vq"]["num_rvq"] == 3:
            #     style_pred_embs = torch.cat([style_pred_embs, style_pred_embs, style_pred_embs], dim=1) # vq3
            # elif self.model_config["residual_vq"]["num_rvq"] == 2:
            #     style_pred_embs = torch.cat([style_pred_embs, style_pred_embs], dim=1) # vq2
            
            style_pred_embs = self.style_pred_fc(style_pred_embs) # [16, 256*3] -> [16 ,256]

            # self.style_extract_fc : 256*3 -> 256 Linear Layer

            # output shape : [16, 86, 256] / style_ref_embs shape : [16, 256]

            style_ref_embs = self.style_extract_fc(style_ref_embs) 
            # [B, 256 * n_stages] -> [B, 256]

            # if did_x0_init and step >= 350000:
            #     output = output + style_pred_embs.unsqueeze(1)
            #     # codebooks 구성 (num_rvq == 3 가정)
            #     z1, z2, z3 = torch.split(orig_style_pred_embs, 256, dim=1)
            #     codebooks = [z1, z2, z3, z1+z2+z3]
            # else:
            output = output + style_ref_embs.unsqueeze(1)

            positions = self.embed_positions(style_ref_embs.unsqueeze(1)[:, :, 0])
            prosody_embedding = style_ref_embs.unsqueeze(1) + positions

        else:
            style_ref_embs, vq_loss, min_encoding_indices, orig_style_ref_embs = None, None, None, None



            t_end_infer = float(intensity)
            t_end_infer = max(0.0, min(1.0, t_end_infer))  # clamp
            # neu_emb: [B,256] (forward 초반에 만든 것 그대로)
            style_pred_embs = self.style_predictor(
                text_enc=output,
                style_tag_emb=style_tag_emb,
                neu_emb=neu_cond,                     # [B,256]
                text_mask=text_mask,
                t_end=t_end_infer,
                steps=self.model_config["style_predictor"].get("steps_infer", 1),
            )  # [B,256]

            orig_style_pred_embs = style_pred_embs


            # codebooks 구성 (num_rvq == 3 가정)
            z1, z2, z3 = torch.split(style_pred_embs, 256, dim=1)
            codebooks = [z1, z2, z3, z1+z2+z3]

            # # neutral이면 강제로 neu_emb 사용
            # if self.neutral_id is not None:
            #     neutral_mask = (emotions == self.neutral_id)    # [B]
            #     if neutral_mask.any():
            #         style_pred_embs = style_pred_embs.clone()
            #         style_pred_embs[neutral_mask] = neu_emb[neutral_mask]  

            
            style_pred_embs = self.style_pred_fc(style_pred_embs)

            output = output + style_pred_embs.unsqueeze(1)
            positions = self.embed_positions(style_pred_embs.unsqueeze(1)[:, :, 0])
            prosody_embedding = style_pred_embs.unsqueeze(1) + positions

            flow_loss = torch.tensor(0.0, device=device)
            soft_zero_loss = torch.tensor(0.0, device=device)

        src_key_padding_mask = output[:, :, 0].eq(self.padding_idx).data
        prosody_key_padding_mask = prosody_embedding[:, :, 0].eq(self.padding_idx).data

        # Text2style_alignment
        t2s_align, guided_loss_2, attn_emo_list = self.text2style_alignment(
            output.transpose(0, 1),
            prosody_embedding.transpose(0, 1),
            src_key_padding_mask,
            prosody_key_padding_mask
        )
        output = output + t2s_align.transpose(0, 1)

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

        # Decoder
        if self.model_config["residual_vq"]["num_rvq"] == 3:
            output, mel_masks = self.decoder(output, mel_masks, codebooks) # vq3
        else:
            output, mel_masks = self.decoder(output, mel_masks) # vq2, vq4
        output = self.mel_linear(output)

        # Post-net
        postnet_output = self.postnet(output) + output

        # Loss
        guided_loss = guided_loss_1 + guided_loss_2

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
