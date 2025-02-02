import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import torch
from scipy.io import wavfile
import os

from utils.model import get_vocoder
from utils.tools import get_configs_of


def plot_mel(data, titles=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for _ in range(len(data))]

    for i in range(len(data)):
        mel = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig

def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs

"""
mel_basename = "JCH_0003-mel-0003_G1A4E5S0C0_JCH_001396"
# mel_basename = "CHY_0012-mel-0012_G1A2E4S0C0_CHY_000103"
mel_basename = "CHY_0012-mel-0012_G1A2E4S0C0_CHY_000001"
mel_basename = "CHY_0012-mel-0012_G1A2E4S0C0_CHY_000932"
mel_basename = "CHY_0012-mel-0012_G1A2E4S0C0_CHY_000002"
mel_basename = "JCH_0003-mel-0003_G1A4E1S0C0_JCH_00101"
mel_basename = "JCH_0003-mel-0003_G1A4E1S0C0_JCH_00044"
mel_basename = "CHY_0012-mel-0012_G1A2E6S0C0_CHY_000402"
basename = mel_basename[13:]
speaker  = mel_basename[:8]
print("basename:", basename) 
print("speaker:", speaker)


# mel_basename = "JCH_0003-mel-0003_G1A4E5S0C0_JCH_00246"
# 0003_G1A4E5S0C0_JCH_001396


# Load the mel spectrogram data
mel_data = np.load('./preprocessed_data/emo_kr_22050/mel/{}.npy'.format(mel_basename))

# mel_data = np.load('/home/gaeun/Documents/DL/Codes/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/output/result/icassp_2024/740000/0012_G1A2E6S0C0_CHY_000402.npy')
# mel_basename = "0012_G1A2E6S0C0_CHY_000402_Synthesized"

"""

mel_data = np.load('/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/raw_norm_wavs/pitch_norm/CHY/CHY_ang_000001.npy')

mel_data = np.transpose(mel_data)

fig = plot_mel([mel_data], ["CHY_ang_000001_p"])
plt.savefig('./preprocessed_mels/{}.png'.format("CHY_ang_000001_p"), format='png')
plt.close()

print("mel_data shape", mel_data.shape)



# postnet_output shape torch.Size([1, 77, 80])s


"""
# Read Config
preprocess_config, model_config, train_config = get_configs_of("icassp_2024")

torch.manual_seed(train_config["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(train_config["seed"])
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device of TTS: {device}")

# Load vocoder
vocoder = get_vocoder(model_config, device)

mel_data = torch.from_numpy(mel_data).float().unsqueeze(0).to(device)


print("mel_data shape", mel_data.shape)

wav_data = vocoder_infer(
    mel_data, vocoder, model_config, preprocess_config
)

sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
wavfile.write('./preprocessed_mels/{}.wav'.format(mel_basename), sampling_rate, wav_data[0])

lab_path = os.path.join("./raw_data/emo_kr_22050", speaker, "{}.lab".format(basename))

print("lab path:", lab_path)

with open(lab_path, 'r') as f:
    print(f.readline())





# # Create the plot
# plt.figure()
# librosa.display.specshow(mel_data)
# plt.colorbar()

# # Save the plot as a PNG file
# plt.savefig('./preprocessed_mels/{}.png'.format(mel_basename))
# plt.close()
"""