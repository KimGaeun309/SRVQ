import torch
from torch.utils.data import DataLoader

from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset


def evaluate(device, model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    dataset = Dataset(
        "val_ra.txt", preprocess_config, train_config,
        sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    loss_sums = [0.0] * 6  # total, mel, postnet, pitch, energy, duration

    model.eval()
    with torch.no_grad():
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                output = model(*(batch[2:]), inference=False)
                losses = Loss(batch, output)  # tuple of 6 scalars

                B = len(batch[0])
                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * B

    loss_means = [l / len(dataset) for l in loss_sums]

    message = (
        "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, "
        "Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, "
        "Duration Loss: {:.4f}"
    ).format(step, *loss_means)

    if logger is not None and vocoder is not None:
        # 마지막 batch 하나로 샘플 로그
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch, model, vocoder, model_config, preprocess_config
        )

        log(logger, step, losses=loss_means)
        log(logger, fig=fig, tag="Validation/step_{}_{}".format(step, tag))

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