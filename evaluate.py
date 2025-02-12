import torch
from torch.utils.data import DataLoader

from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

import numpy as np
from utils.tools import pad_1D, pad_2D

def evaluate(device, model, step, configs, logger=None, vocoder=None, losses=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [{k:0 for k in loss.keys()} if isinstance(loss, dict) else 0 for loss in losses]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)

                
            basenames = batch[0]
            
            pitch_mel, energy_mel = [], []

            for basename in basenames:
                pitch_path = f"/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/normalized_data/pitch_only/{basename}_pitch.npy"
                energy_path = f"/root/mydir/ICASSP2024_FS2-develop/ICASSP2024_FS2-develop/normalized_data/energy_only/{basename}_energy.npy"
                pitch_mel.append(np.load(pitch_path).T)
                energy_mel.append(np.load(energy_path).T)
            
            pitch_mel = pad_2D(pitch_mel)
            pitch_mel = torch.from_numpy(pitch_mel).to('cuda')
            energy_mel = pad_2D(energy_mel)
            energy_mel = torch.from_numpy(energy_mel).to('cuda')
            


            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]), step=step, inference=False,  pitch_mel=pitch_mel, energy_mel=energy_mel,) # To do Step

                # Cal Loss
                losses = Loss(batch, output, step=step)

                for i in range(len(losses)):
                    if isinstance(losses[i], dict):
                        for k in loss_sums[i].keys():
                            loss_sums[i][k] += losses[i][k].item() * len(batch[0])
                    else:
                        loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = []
    loss_means_ = []
    for loss_sum in loss_sums:
        if isinstance(loss_sum, dict):
            loss_mean = {k:v / len(dataset) for k, v in loss_sum.items()}
            loss_means.append(loss_mean)
            loss_means_.append(sum(loss_mean.values()))
        else:
            loss_means.append(loss_sum / len(dataset))
            loss_means_.append(loss_sum / len(dataset))

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Style Loss: {:.4f}, guided_loss: {:.4f}, vq_loss: {:.4f}".format( 
        *([step] + [l for l in loss_means_])
    )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag, style_attn = synth_one_sample(
            batch,
            model,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message
