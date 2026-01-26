import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FastSpeech2Loss(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super().__init__()

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        sp_cfg = model_config.get("style_predictor", {})
        self.neutral_id: Optional[int] = sp_cfg.get("neutral_id", None)

        # weight (원하는대로 조절)
        self.lambda_style = float(sp_cfg.get("lambda_style", 0.1))        # style MAE
        self.lambda_flow = float(sp_cfg.get("lambda_flow", 30.0))         # flow loss
        self.lambda_soft0 = float(sp_cfg.get("lambda_soft0", 10.0))       # soft-zero loss

    def forward(self, inputs, predictions, step):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[7:]

        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            style_ref_embs,       # SER target
            style_pred_embs,      # predictor output
            guided_loss,          # (unused now)
            vq_loss,              # (unused now)
            flow_loss,            # 핵심
            soft_zero_loss,       # neutral regularizer
            min_encoding_indices, # (unused)
            orig_style_ref_embs,  # (unused)
            x0_raw,               # (unused)
            orig_style_pred_embs, # (unused)
        ) = predictions

        # ====== mask 정리 ======
        src_masks = ~src_masks
        mel_masks = ~mel_masks

        log_duration_targets = torch.log(duration_targets.float() + 1)

        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, : mel_masks.shape[1]]

        # stop grad
        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        # ====== pitch/energy masking ======
        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        elif self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        # duration
        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        # mel
        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        # ====== 기본 loss ======
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        # ====== Style loss (SER target) ======
        # predictor output이 SER로 잘 회귀되도록 보조 loss
        # (flow_loss만 써도 되는데 초기 안정화에 도움 됨)
        if (style_ref_embs is None) or (style_pred_embs is None):
            style_loss = torch.tensor(0.0, device=mel_loss.device)
        else:
            style_loss = self.mae_loss(style_pred_embs, style_ref_embs)

        # flow term (이미 scalar로 들어온다고 가정)
        style_flow_term = flow_loss
        soft_zero_term = soft_zero_loss

        total_loss = (
            mel_loss
            + postnet_mel_loss
            + pitch_loss
            + energy_loss
            + duration_loss
            + self.lambda_style * style_loss
            + self.lambda_flow * style_flow_term
            + self.lambda_soft0 * soft_zero_term
        )

        return (
            total_loss,         # 0
            mel_loss,           # 1
            postnet_mel_loss,   # 2
            pitch_loss,         # 3
            energy_loss,        # 4
            duration_loss,      # 5
            style_loss,         # 6
            style_flow_term,    # 7
            soft_zero_term,     # 8
        )