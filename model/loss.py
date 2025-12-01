import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.criterion = nn.CrossEntropyLoss()

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
            style_ref_embs,
            style_pred_embs,
            guided_loss,
            vq_loss,
            min_encoding_indices
        ) = predictions

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        # Style loss
        style_loss = self.mae_loss(style_pred_embs, style_ref_embs) * 10 # lamda scale
        total_style_loss = style_loss + guided_loss
    

        


        # Clssifier
        # emotions_pred = torch.softmax(min_encoding_indices.float(), dim=1)
        emotions = inputs[3]
        num_classes = 5

        # 1. one-hot hard assignment
        onehot = F.one_hot(min_encoding_indices.squeeze(), num_classes).float()  # (B, num_classes)

        # 2. soft version for gradient
        soft = F.softmax(onehot / 1.0, dim=-1)  # dummy continuous proxy
        # 또는 더 자연스럽게
        # soft = onehot.clone() + 0.0  # shape 유지용

        # 3. straight-through trick
        # forward는 one-hot, backward는 soft
        onehot_st = onehot + soft - soft.detach()

        # 4. logit처럼 사용
        # small linear projection to match logit space
        logits = onehot_st @ torch.eye(num_classes, device=onehot.device)  # identity mapping
        # (optionally add trainable weights)
        # logits = self.classifier(onehot_st)

        classifier_loss = self.criterion(logits, emotions)

        # classifier_loss = self.criterion(min_encoding_indices.float(), emotions)
        
        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + total_style_loss + vq_loss + classifier_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            style_loss,
            guided_loss,
            vq_loss,
            classifier_loss,
        )