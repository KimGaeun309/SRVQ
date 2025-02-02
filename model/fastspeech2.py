import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text2style_aligner import Text2Style_Aligner
from .style_predictor import StylePredictor, LinearNorm
from .transformers.transformer import Encoder, Decoder, MelDecoder, LightMelDecoder
from .transformers.layers import PostNet
from .modules import VarianceAdaptor, SinusoidalPositionalEmbedding
from utils.tools import get_mask_from_lengths
from text.symbols import symbols
from .residual_vq_gaeun import ReferenceEncoderSRVQ3, SRVQ3WithNeutralization # 수정
# from .residuaal_vq import SRVQPyworld, ResidualVQ

from .gst.style_encoder import StyleEncoder, GST_VQ

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
        self.ref_enc = ReferenceEncoderSRVQ3(
            e_dim=model_config["residual_vq"]["vq_hidden"],
        )
        self.style_extractor = SRVQ3WithNeutralization(
            idim=model_config["residual_vq"]["n_mel_channels"],
            conv_layers=model_config["residual_vq"]["rvq_conv_layers"],
            conv_chans_list=model_config["residual_vq"]["rvq_conv_chans_list"],
            conv_kernel_size=model_config["residual_vq"]["rvq_conv_kernel_size"],
            conv_stride=model_config["residual_vq"]["rvq_conv_stride"],
            gru_layers=model_config["residual_vq"]["rvq_gru_layers"],
            gru_units=model_config["residual_vq"]["rvq_gru_units"],
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

        self.style_predictor = StylePredictor()

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
        pitch_mel=None,
        energy_mel=None,
        init_flag=False,
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

        # Style module
        ## Style predictor

        
        emo_emb = self.emotion_emb(emotions).unsqueeze(1)
        text_key_padding_mask = output[:, :, 0].eq(self.padding_idx).data
        emo_key_padding_mask = emo_emb[:, :, 0].eq(self.padding_idx).data
        phn_style_emb, guided_loss_1, _ = self.cross_attn(
            emo_emb.transpose(0, 1),
            output.transpose(0, 1).detach(),
            emo_key_padding_mask,
            text_key_padding_mask
        )

        if not inference:
            ## Style extractor
            style_pred_embs = self.style_predictor(phn_style_emb.transpose(0, 1))

            if self.model_config["residual_vq"]["num_rvq"] == 4:
                style_pred_embs = torch.cat([style_pred_embs, style_pred_embs, style_pred_embs, style_pred_embs], dim=1) # vq4
            elif self.model_config["residual_vq"]["num_rvq"] == 3:
                style_pred_embs = torch.cat([style_pred_embs, style_pred_embs, style_pred_embs], dim=1) # vq3
            elif self.model_config["residual_vq"]["num_rvq"] == 3:
                style_pred_embs = torch.cat([style_pred_embs, style_pred_embs], dim=1) # vq2
            
            style_pred_embs = self.style_pred_fc(style_pred_embs) # [16, 256*3] -> [16 ,256]
            

            if self.model_config["gst"]["use_gst"]:
                ref_embs = self.gst(mels)
                style_ref_embs, vq_loss, min_encoding_indices, codebooks = self.style_extractor(ref_embs, p_targets=p_targets, d_targets=d_targets, e_targets=e_targets)
            else:
                print("emotions", emotions)
                # style_ref_embs, vq_loss, min_encoding_indices, codebooks = self.style_extractor(mels, emotions=emotions)
                z_mel, z_pitch, z_energy, cls_loss = self.ref_enc(mels, emotions=emotions, p_mel=pitch_mel, e_mel=energy_mel)
                if init_flag:
                    # kmeans_init !!!!
                    self.style_extractor.RVQ1.vq_layers[0].init_codebook_kmeans(z_mel)
                    self.style_extractor.RVQ2.vq_layers[0].init_codebook_kmeans(z_pitch)
                    self.style_extractor.RVQ3.vq_layers[0].init_codebook_kmeans(z_energy)

                style_ref_embs, vq_loss, min_encoding_indices, codebooks = self.style_extractor(z_mel, z_pitch, z_energy, cls_loss) 
                # style_ref_embs, vq_loss, min_encoding_indices, codebooks = self.style_extractor(mels, p_targets=p_targets, d_targets=d_targets, e_targets=e_targets)
                print('style_ref_embs', style_ref_embs.shape)

                if init_flag:
                    self.style_extractor.RVQ1.vq_layers[1].init_codebook_kmeans(style_ref_embs[:, :128])
                    self.style_extractor.RVQ2.vq_layers[1].init_codebook_kmeans(style_ref_embs[:, 256:384])
                    self.style_extractor.RVQ3.vq_layers[1].init_codebook_kmeans(style_ref_embs[:, 512:640])

            # style_ref_embs shape : [16, 256*3]

            style_ref_embs = self.style_extract_fc(style_ref_embs) 
            # self.style_extract_fc : 256*3 -> 256 Linear Layer

            # output shape : [16, 86, 256] / style_ref_embs shape : [16, 256]

            output = output + style_ref_embs.unsqueeze(1)

            positions = self.embed_positions(style_ref_embs.unsqueeze(1)[:, :, 0])
            prosody_embedding = style_ref_embs.unsqueeze(1) + positions

        else:
            style_ref_embs, vq_loss, min_encoding_indices = None, None, None
            style_pred_embs = self.style_predictor(phn_style_emb.transpose(0, 1))
            if self.model_config["residual_vq"]["num_rvq"] == 4:
                style_pred_embs = torch.cat([style_pred_embs, style_pred_embs, style_pred_embs, style_pred_embs], dim=1) # vq4
            elif self.model_config["residual_vq"]["num_rvq"] == 3:
                style_pred_embs = torch.cat([style_pred_embs, style_pred_embs, style_pred_embs], dim=1) # vq3
                codebook = torch.split(style_pred_embs, 256, dim=1) # vq3
                codebooks = [codebook[0], codebook[1], codebook[2], codebook[0] + codebook[1] + codebook[2]] # vq3
            elif self.model_config["residual_vq"]["num_rvq"] == 2:
                style_pred_embs = torch.cat([style_pred_embs, style_pred_embs], dim=1) # vq2
            style_pred_embs = self.style_pred_fc(style_pred_embs)

            output = output + style_pred_embs.unsqueeze(1)
            positions = self.embed_positions(style_pred_embs.unsqueeze(1)[:, :, 0])
            prosody_embedding = style_pred_embs.unsqueeze(1) + positions

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
            min_encoding_indices
        )
