import os
import json

import yaml
import torch
import matplotlib
import numpy as np
import torch.nn.functional as F

from scipy.io import wavfile
from matplotlib import pyplot as plt

import torchaudio
from speechbrain.inference.vocoders import HIFIGAN

matplotlib.use("Agg")


def load_emotion_id2name(preprocessed_path):
    emo_path = os.path.join(preprocessed_path, "emotions.json")
    with open(emo_path, "r", encoding="utf-8") as f:
        emo2id = json.load(f)  # {"Angry":0, ...}

    id2emo = {int(v): k for k, v in emo2id.items()}  # {0:"Angry", ...}
    return id2emo


def get_speechbrain_hifigan():
    # 16kHz HiFiGAN (SpeechBrain pretrained)
    vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="pretrained_models/tts-hifigan-libritts-16kHz")
    return vocoder


def speechbrain_vocode(vocoder, mel):
    """
    mel: torch.Tensor
      - (80, T) or (B, 80, T)
    return:
      - wav int16 (T_wav,)
    """
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)  # (1, 80, T)

    # ✅ SpeechBrain expects (B, 80, T) 그대로
    with torch.no_grad():
        wav = vocoder.decode_batch(mel)  # (B, T_wav)


    # 혹시 (B, 1, T)로 나오면 정리
    if wav.dim() == 3:
        wav = wav.squeeze(1)  # (B, T)

    # wav = wav[0].unsqueeze(0)   # (1, T)

    # ✅ torchaudio.save는 CPU tensor가 안전
    wav = wav.detach().cpu()

    # ✅ 클리핑 (optional)
    wav = torch.clamp(wav, -1.0, 1.0)

    return wav


def load_gt_mel_from_preprocessed(preprocessed_path, basename):
    """
    basename 예: 0012_000356
    preprocessed mel 파일은 보통:
      {preprocessed_path}/mel/{speaker}-mel-{basename}.npy
    그리고 저장된 mel shape은 (T, 80) 인 경우가 많음.
    vocoder_infer는 (B, 80, T) 를 기대하므로 transpose 필요.
    """
    speaker = basename.split("_")[0]
    mel_path = os.path.join(preprocessed_path, "mel", f"{speaker}-mel-{basename}.npy")
    if not os.path.exists(mel_path):
        return None

    mel = np.load(mel_path)  # (T, 80)
    if mel.ndim != 2:
        return None

    # (T, 80) -> (80, T)
    if mel.shape[1] == 80:
        mel = mel.T
    return mel

def load_gt_pitch_from_preprocessed(preprocessed_path, basename):
    speaker = basename.split("_")[0]
    p_path = os.path.join(preprocessed_path, "pitch", f"{speaker}-pitch-{basename}.npy")
    if not os.path.exists(p_path):
        return None
    pitch = np.load(p_path)  # (T,) or (T,1)
    pitch = np.squeeze(pitch)
    return pitch


def load_gt_energy_from_preprocessed(preprocessed_path, basename):
    speaker = basename.split("_")[0]
    e_path = os.path.join(preprocessed_path, "energy", f"{speaker}-energy-{basename}.npy")
    if not os.path.exists(e_path):
        return None
    energy = np.load(e_path)  # (T,) or (T,1)
    energy = np.squeeze(energy)
    return energy


def _match_length_1d(x, T):
    """x: (Tx,) -> (T,) 로 자르거나 0으로 pad"""
    if x is None:
        return None
    x = np.asarray(x).reshape(-1)
    if len(x) >= T:
        return x[:T]
    pad = np.zeros((T - len(x),), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)
def save_mel_compare_with_pe_png(
    gt_mel,
    pred_mel,
    gt_pitch,
    gt_energy,
    pred_pitch,
    pred_energy,
    stats,
    out_path,
    title_left="GT",
    title_right="Pred",
):
    """
    gt_mel, pred_mel: numpy (80, T)
    gt_pitch/energy: numpy (Tgt,)
    pred_pitch/energy: numpy (Tpred,)
    stats: [pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max]
    """

    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats

    gt_mel = np.asarray(gt_mel)
    pred_mel = np.asarray(pred_mel)

    Tgt = gt_mel.shape[1]
    Tpred = pred_mel.shape[1]

    gt_pitch = _match_length_1d(gt_pitch, Tgt)
    gt_energy = _match_length_1d(gt_energy, Tgt)
    pred_pitch = _match_length_1d(pred_pitch, Tpred)
    pred_energy = _match_length_1d(pred_energy, Tpred)

    # de-norm pitch only (energy는 stats에서 min/max만 쓰는 형태라 그대로)
    gt_pitch_plot = gt_pitch * pitch_std + pitch_mean
    pred_pitch_plot = pred_pitch * pitch_std + pitch_mean

    vmin = min(gt_mel.min(), pred_mel.min())
    vmax = max(gt_mel.max(), pred_mel.max())

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    # ---- GT ----
    axes[0].imshow(gt_mel, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title(title_left, fontsize="medium")
    axes[0].set_ylim(0, gt_mel.shape[0])
    axes[0].tick_params(labelsize="x-small", left=False, labelleft=False)
    axes[0].set_anchor("W")

    ax1 = add_axis(fig, axes[0])
    ax1.plot(gt_pitch_plot, color="tomato")
    ax1.set_xlim(0, Tgt)
    ax1.set_ylim(0, pitch_max * pitch_std + pitch_mean)
    ax1.set_ylabel("F0", color="tomato")
    ax1.tick_params(labelsize="x-small", colors="tomato", bottom=False, labelbottom=False)

    ax2 = add_axis(fig, axes[0])
    ax2.plot(gt_energy, color="darkviolet")
    ax2.set_xlim(0, Tgt)
    ax2.set_ylim(energy_min, energy_max)
    ax2.set_ylabel("Energy", color="darkviolet")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(
        labelsize="x-small",
        colors="darkviolet",
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
    )

    # ---- Pred ----
    axes[1].imshow(pred_mel, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title(title_right, fontsize="medium")
    axes[1].set_ylim(0, pred_mel.shape[0])
    axes[1].tick_params(labelsize="x-small", left=False, labelleft=False)
    axes[1].set_anchor("W")

    ax1 = add_axis(fig, axes[1])
    ax1.plot(pred_pitch_plot, color="tomato")
    ax1.set_xlim(0, Tpred)
    ax1.set_ylim(0, pitch_max * pitch_std + pitch_mean)
    ax1.set_ylabel("F0", color="tomato")
    ax1.tick_params(labelsize="x-small", colors="tomato", bottom=False, labelbottom=False)

    ax2 = add_axis(fig, axes[1])
    ax2.plot(pred_energy, color="darkviolet")
    ax2.set_xlim(0, Tpred)
    ax2.set_ylim(energy_min, energy_max)
    ax2.set_ylabel("Energy", color="darkviolet")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(
        labelsize="x-small",
        colors="darkviolet",
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
    )

    # colorbar 하나만
    fig.colorbar(axes[1].images[0], ax=axes, fraction=0.02, pad=0.02)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def get_configs_of(dataset):
    config_dir = os.path.join("./config", dataset)
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        os.path.join(config_dir, "model.yaml"), "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        os.path.join(config_dir, "train.yaml"), "r"), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config

def get_decode_config(dataset):
    config_dir = os.path.join("./config", dataset)
    decode_config = yaml.load(open(
        os.path.join(config_dir, "decode.yaml"), "r"), Loader=yaml.FullLoader)
    return decode_config

def to_device(data, device):
    if len(data) == 13:
        (
            ids,
            raw_texts,
            speakers,
            emotions,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        emotions = torch.from_numpy(emotions).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            emotions,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    if len(data) == 7:
        (ids, raw_texts, speakers, emotions, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        emotions = torch.from_numpy(emotions).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, emotions, texts, src_lens, max_src_len)

    if len(data) == 10:
        (ids, raw_texts, speakers, emotions, texts, src_lens, max_src_len, mel, mel_len, max_mel_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        emotions = torch.from_numpy(emotions).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mel = torch.from_numpy(mel).float().to(device)
        mel_len = torch.from_numpy(mel_len).to(device)

        return (ids, raw_texts, speakers, emotions, texts, src_lens, max_src_len, mel, mel_len, max_mel_len)


def log(
    logger, step=None, losses=None, fig=None, audio=None, alignment=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)
        logger.add_scalar("Loss/style_loss", losses[6], step) 
        logger.add_scalar("Loss/guide_loss", losses[7], step)
        logger.add_scalar("Loss/vq_loss", losses[8], step)
        logger.add_scalar("Loss/cls_loss(indices)", losses[9], step)
        logger.add_scalar("Loss/flow_loss", losses[10], step)
        logger.add_scalar("Loss/neu_align_loss", losses[11], step)
        logger.add_scalar("Loss/soft_zero_loss", losses[12], step) 

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )

    if alignment is not None:
        logger.add_image(
            tag,
            plot_alignment_to_numpy(alignment.data.cpu().numpy().T),
            step,
            dataformats='HWC'
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(batch, model, vocoder, model_config, preprocess_config):

    with torch.no_grad():
        test_output = model(*(batch[2:]), inference=True) # Inference

    (
        ids,
        raw_texts,
        speakers,
        emotions,
        texts,
        text_lens,
        max_text_lens,
        mels,
        mel_lens,
        max_mel_lens,
        pitches,
        energies,
        durations
    ) = batch

    (
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
        _, 
        _,
        _,
        _,
        _,
        _,
        *rest,
    ) = test_output

    basename = ids[0]
    src_len = src_lens[0].item()
    mel_len = mel_lens[0].item()
    mel_target = mels[0, :mel_len].detach().transpose(0, 1)
    mel_prediction = postnet_output[0, :mel_len].detach().transpose(0, 1)
    duration = durations[0, :src_len].detach().cpu().numpy()
    style_attn = None

    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = pitches[0, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = pitches[0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = energies[0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = energies[0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )
    vocoder = get_speechbrain_hifigan()
    wav_reconstruction = speechbrain_vocode(vocoder,  mel_target)
    wav_prediction = speechbrain_vocode(vocoder, mel_prediction)

    # if vocoder is not None:
    #     from .model import vocoder_infer

    #     wav_reconstruction = vocoder_infer(
    #         mel_target.unsqueeze(0),
    #         vocoder,
    #         model_config,
    #         preprocess_config,
    #     )[0]
    #     wav_prediction = vocoder_infer(
    #         mel_prediction.unsqueeze(0),
    #         vocoder,
    #         model_config,
    #         preprocess_config,
    #     )[0]
    # else:
    #     wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename, style_attn


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path, args):

    basenames = targets[0]
    emotion_ids = targets[3]

    (
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
        style_embs,
        style_pred_embs,
        guided_loss,
        vq_loss,
        _, 
        *rest,
    ) = predictions

    # ✅ GT mel 로딩 위한 preprocessed_path
    preprocessed_path = preprocess_config["path"]["preprocessed_path"]
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    id2emo = load_emotion_id2name(preprocessed_path)
    
    os.makedirs(path, exist_ok=True)

    vocoder = get_speechbrain_hifigan()

    for i in range(len(predictions[0])):
        basename = basenames[i]
        emotion = id2emo[emotion_ids[i].item()]
        src_len = src_lens[i].item()
        mel_len = mel_lens[i].item()

        # pred mel (80, Tpred)
        mel_prediction = postnet_output[i, :mel_len].detach().transpose(0, 1)
        duration = d_rounded[i, :src_len].detach().cpu().numpy()

        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = p_predictions[i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = p_predictions[i, :mel_len].detach().cpu().numpy()

        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = e_predictions[i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = e_predictions[i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocessed_path, "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        # # =========================
        # # (1) 기존: pred mel png 저장
        # # =========================
        # fig = plot_mel(
        #     [
        #         (mel_prediction.cpu().numpy(), pitch, energy),
        #     ],
        #     stats,
        #     ["Synthetized Spectrogram"],
        # )
        # plt.savefig(os.path.join(path, "{}.png".format(basename)))
        # plt.close()

        # =========================
        # ✅ 추가: GT mel 로드 + 비교 PNG 저장
        # =========================
        gt_mel = load_gt_mel_from_preprocessed(preprocessed_path, basename)  # (80,Tgt)
        gt_pitch = load_gt_pitch_from_preprocessed(preprocessed_path, basename)  # (Tgt,)
        gt_energy = load_gt_energy_from_preprocessed(preprocessed_path, basename)  # (Tgt,)


        if gt_mel is not None and gt_pitch is not None and gt_energy is not None:
            pred_mel_np = mel_prediction.cpu().numpy()  # (80, Tpred)

            # pred pitch/energy는 이미 만들어둔거 사용
            pred_pitch = pitch
            pred_energy = energy

            compare_png_path = os.path.join(path, f"{emotion}-{basename}.png")

            save_mel_compare_with_pe_png(
                gt_mel=gt_mel,
                pred_mel=pred_mel_np,
                gt_pitch=gt_pitch,
                gt_energy=gt_energy,
                pred_pitch=pred_pitch,
                pred_energy=pred_energy,
                stats=stats,
                out_path=compare_png_path,
                title_left=f"GT mel ({basename})",
                title_right=f"Pred mel ({basename})",
            )
        else:
            fig = plot_mel(
                [
                    (mel_prediction.cpu().numpy(), pitch, energy),
                ],
                stats,
                ["Synthetized Spectrogram"],
            )
            plt.savefig(os.path.join(path, "{}.png".format(basename)))
            plt.close()

        pred_wav = speechbrain_vocode(vocoder, mel_prediction)
        torchaudio.save(os.path.join(path, f"{emotion}-{basename}-pred.wav"), pred_wav, sampling_rate)

        if gt_mel is not None:
            gt_wav = speechbrain_vocode(vocoder, torch.from_numpy(gt_mel))
            torchaudio.save(os.path.join(path, f"{emotion}-{basename}-gt.wav"), gt_wav, sampling_rate)

        np.save(os.path.join(path, f"{emotion}-{basename}"), mel_prediction.detach().cpu().numpy())

        print(f"[OK] {basename} -> png / gt.wav / pred.wav")
    
    # from .model import vocoder_infer

    # print("postnet_output shape", postnet_output.shape)

    # mel_predictions = postnet_output.transpose(1, 2)  # (B, 80, Tpred)

    # print("mel_predictions shape", mel_predictions.shape)
    # lengths = mel_lens * preprocess_config["preprocessing"]["stft"]["hop_length"]

    # # =========================
    # # (2) 기존: pred wav 생성
    # # =========================
    # wav_predictions = vocoder_infer(
    #     mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    # )

    # # (3) 기존: pred mel npy 저장
    # for mel, basename in zip(mel_predictions, basenames):
    #     np.save(os.path.join(path, basename), mel.cpu())

    # sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    # # =========================
    # # (4) 기존: pred wav 저장
    # # =========================
    # for wav, basename in zip(wav_predictions, basenames):
    #     wavfile.write(os.path.join(path, "{}-pred.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
