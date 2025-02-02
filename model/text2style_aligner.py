import torch


class CrossAttenLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CrossAttenLayer, self).__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.activation = torch.nn.ReLU()

    def forward(self, src, local_emotion, emotion_key_padding_mask=None, forcing=False):
        # src: (Tph, B, 256) local_emotion: (Temo, B, 256) emotion_key_padding_mask: (B, Temo)
        if forcing:
            maxlength = src.shape[0]
            k = local_emotion.shape[0] / src.shape[0]
            lengths1 = torch.ceil(torch.tensor([i for i in range(maxlength)]).to(src.device) * k) + 1
            lengths2 = torch.floor(torch.tensor([i for i in range(maxlength)]).to(src.device) * k) - 1
            mask1 = sequence_mask(lengths1, local_emotion.shape[0])
            mask2 = sequence_mask(lengths2, local_emotion.shape[0])
            mask = mask1.float() - mask2.float()
            attn_emo = mask.repeat(src.shape[1], 1, 1) # (B, Tph, Temo)
            src2 = torch.matmul(local_emotion.permute(1, 2, 0), attn_emo.float().transpose(1, 2)).permute(2, 0, 1)
        else:
            src2, attn_emo = self.multihead_attn(src, local_emotion, local_emotion, key_padding_mask=emotion_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_emo


class Text2Style_Aligner(torch.nn.Module):
    def __init__(
            self,
            num_layers,
            hidden_size,
            guided_sigma=0.3,
            guided_layers=None,
            norm=None
        ):
        super(Text2Style_Aligner, self).__init__()
        self.layers = torch.nn.ModuleList(
            [CrossAttenLayer(d_model=hidden_size, nhead=2) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm
        self.guided_sigma = guided_sigma
        self.guided_layers = guided_layers if guided_layers is not None else num_layers

    def forward(self, src, local_emotion, src_key_padding_mask=None, emotion_key_padding_mask=None, forcing=False):
            output = src
            guided_loss = 0
            attn_emo_list = []
            for i, mod in enumerate(self.layers):
                # output: (Tph, B, 256), global_emotion: (1, B, 256), local_emotion: (Temo, B, 256) mask: None, src_key_padding_mask: (B, Tph),
                # emotion_key_padding_mask: (B, Temo)
                output, attn_emo = mod(output, local_emotion, emotion_key_padding_mask=emotion_key_padding_mask, forcing=forcing)
                attn_emo_list.append(attn_emo.unsqueeze(1))
                # attn_emo: (B, Tph, Temo) attn: (B, Tph, Tph)
                if i < self.guided_layers and src_key_padding_mask is not None:
                    s_length = (~src_key_padding_mask).float().sum(-1) # B
                    emo_length = (~emotion_key_padding_mask).float().sum(-1)
                    attn_w_emo = _make_guided_attention_mask(src_key_padding_mask.size(-1), s_length, emotion_key_padding_mask.size(-1), emo_length, self.guided_sigma)

                    g_loss_emo = attn_emo * attn_w_emo  # N, L, S
                    non_padding_mask = (~src_key_padding_mask).unsqueeze(-1) & (~emotion_key_padding_mask).unsqueeze(1)
                    guided_loss = g_loss_emo[non_padding_mask].mean() + guided_loss

            if self.norm is not None:
                output = self.norm(output)

            return output, guided_loss, attn_emo_list

def sequence_mask(lengths, maxlen, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).to(lengths.device).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask


def _make_guided_attention_mask(ilen, rilen, olen, rolen, sigma):
    grid_x, grid_y = torch.meshgrid(torch.arange(ilen, device=rilen.device), torch.arange(olen, device=rolen.device))
    grid_x = grid_x.unsqueeze(0).expand(rilen.size(0), -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(rolen.size(0), -1, -1)
    rilen = rilen.unsqueeze(1).unsqueeze(1)
    rolen = rolen.unsqueeze(1).unsqueeze(1)
    return 1.0 - torch.exp(
        -((grid_y.float() / rolen - grid_x.float() / rilen) ** 2) / (2 * (sigma ** 2))
    )
