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


def file_2_vad_ts(filepath: str) -> Iterable[dict]:
    """
    return an array of dict giving start and end of voice segments.
    :param filepath:
    :return:
    """
    import torch
    SAMPLE_RATE = 16000
    torch.set_num_threads(1)
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
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