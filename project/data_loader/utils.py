"""
    sound and images transformations files
"""

import struct
from typing import Iterable

import numpy as np
import webrtcvad
from scipy.io import wavfile


def file_2_vad_struct(filepath: str, aggressiveness: int = 3, window_duration: float = 0.03,
                      bytes_per_sample: int = 2) -> Iterable[dict]:
    """
    convert sample to raw 16 bit per sample stream and test for voice
    :param filepath:
    :param aggressiveness: how aggressively do you want to detect voice (from 0 to 3)
    :param window_duration: duration in seconds
    :param bytes_per_sample

    :return:
    """

    sample_rate, samples = wavfile.read(filepath)
    vad = webrtcvad.Vad()
    vad.set_mode(aggressiveness)
    raw_samples = struct.pack("%dh" % len(samples), *samples.astype(int))
    samples_per_window = int(window_duration * sample_rate + 0.5)
    segments = []
    for start in np.arange(0, len(samples), samples_per_window):
        stop = min(start + samples_per_window, len(samples))
        is_speech = vad.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample],
                                  sample_rate=sample_rate)
        segments.append(dict(start=start, stop=stop, is_speech=is_speech))
    return segments


def file_2_vad_ts(filepath: str, time_space: bool = True) -> Iterable[dict]:
    """
    return an array of dict giving start and end of voice segments.
    :param filepath:
    :param time_space: are we in time space or frequency space
    :return:
    """
    import torch
    if time_space:
        SAMPLE_RATE = 16000
    else:
        SAMPLE_RATE = 1

    torch.set_num_threads(1)
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_ts,
     _,
     _,
     read_audio,
     _,
     _,
     _) = utils
    wav = read_audio(filepath)
    output = get_speech_ts(wav, model, num_steps=4)
    return list(map(lambda mydict: mydict.update(
        {"start": mydict["start"] / SAMPLE_RATE, "end": mydict["end"] / SAMPLE_RATE}) or mydict, output))


def transformations(inpath: str, outpath: str):
    import torchaudio
    from data_loader.data_loaders import MySoundFolder
    from torchaudio.transforms import MelSpectrogram
    from typing import Iterable
    from torchvision import transforms
    from pathlib import Path
    torchaudio.set_audio_backend("sox_io")

    items = MySoundFolder(root=inpath, loader=torchaudio.load)

    for item in items:
        sample, target, abs_path, case = item
        sentiment: str = Path(abs_path).parent.name
        file_name: str = Path(abs_path).stem
        waveform, sample_rate = sample
        # voice activity detection calculus
        vads: Iterable[dict] = file_2_vad_ts(abs_path, time_space=False)
        if len(vads):  # there's a voice
            # trim at start and end
            start = int(vads[0]["start"])
            end = int(vads[-1]["end"])
            waveform = waveform[:, start:end]
            # create and store mfcc
            image_tensor = MelSpectrogram(sample_rate=sample_rate, n_mels=52)(waveform)
            im = transforms.ToPILImage()(image_tensor).convert("RGB")
            Path(Path(outpath), sentiment).mkdir(parents=True, exist_ok=True)
            im.save(Path(Path(outpath), sentiment, f"{case}_{file_name}.jpg"), "JPEG")
