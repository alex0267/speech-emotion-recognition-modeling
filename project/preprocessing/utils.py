"""
    sound and images transformations files
"""

import logging
import struct
import sys
from typing import Iterable

import numpy as np
import webrtcvad
from scipy.io import wavfile
import librosa

from preprocessing.remove_silence import remove_silence_start_end, remove_silence_intervals

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


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
    from pathlib import Path
    from data_loader.voice_activity_detection.snakers4_silero_vad_master.utils_vad import get_speech_ts,read_audio,init_jit_model
    if time_space:
        SAMPLE_RATE = 16000
    else:
        SAMPLE_RATE = 1

    torch.set_num_threads(1)
    model = init_jit_model(str(Path(Path(__file__).parent, "../data_loader/voice_activity_detection", "snakers4_silero_vad_master", "files", "model.jit")))
    wav = read_audio(filepath)
    output = get_speech_ts(wav, model, num_steps=4)
    return list(map(lambda mydict: mydict.update(
        {"start": mydict["start"] / SAMPLE_RATE, "end": mydict["end"] / SAMPLE_RATE}) or mydict, output))


def transformations(inpath: str, outpath: str, debug: bool = False, limit=None,min_size=53,n_mels=56,length=102, mode=None):
    """

    :param inpath:
    :param outpath:
    :param debug:
    :param limit:
    :param min_size: mininimum size in pixel
    :param n_mels: mel spec param
    :param length: Split into patches param
    :return:
    """
    from pathlib import Path
    import torchaudio
    import soundfile
    import torch
    from torchvision import transforms
    from torchaudio.transforms import MelSpectrogram
    from data_loader.data_loaders import MySoundFolder
    from preprocessing.transforms import (ToMelSpectogram)
    torchaudio.set_audio_backend("sox_io")
    if debug:
        logging.info(f" exporting directory {inpath} ")
    if mode is None:
        mode = "start_end"

    items = MySoundFolder(root=inpath, loader=torchaudio.load)

    out = [ToMelSpectogram(n_mels=n_mels)(item[0]).max() for item in items] #mel spec max
    _max = max(out) #store max

    if limit:
        items = list(items)[0:int(limit) + 1] #given a limit add data
    for item in items: #loop on items
        try:
            sample, target, abs_path, case = item #sample data, target, source filepath, case of data augmentation
            sentiment: str = Path(abs_path).parent.name
            file_name: str = Path(abs_path).stem
            if debug:
                logging.info(f" exporting file {abs_path} ")

            waveform, sample_rate = sample
            # voice activity detection calculus

            if mode == "neural_vad":
                waveform = vad_extraction(abs_path, debug, sample_rate, waveform)
            elif mode == "start_end":
                waveform = remove_silence_start_end(waveform, 0.01)
            elif mode == "any":
                waveform = remove_silence_intervals(waveform, 0.01, 3600)
            else:
                raise ValueError("Unknwon mode")
            image_tensor = MelSpectrogram(sample_rate=sample_rate, n_mels=52)(waveform) # create and store mfcc
            normalized_image_tensor = image_tensor/_max #normalize by max

            im = transforms.ToPILImage("L")(normalized_image_tensor) #create a PIL image represented internally as a tensor of 8bits
            Path(Path(outpath), sentiment).mkdir(parents=True, exist_ok=True)
            stem = str(Path(Path(outpath), sentiment, f"{case}_{file_name}"))
            width, height = im.size

            im = im.resize((width,max([height,min_size])))
            im.save(f"{stem}.jpg", "JPEG")

            with open(Path(outpath,"stats.txt"), "w") as f:
                f.write(f"max:{_max}")

            if debug:
                logging.info(f" saving {stem}.wav ")
                soundfile.write(f"{stem}.wav", torch.transpose(waveform, 0, 1), sample_rate)
        except RuntimeError as e:
            logging.info(f"Error on {abs_path} : {str(e)}")


def vad_extraction(abs_path, debug, sample_rate, waveform):
    vads: Iterable[dict] = file_2_vad_ts(abs_path, time_space=True)
    if len(vads):  # there's a voice
        # trim at start and end
        start = int(vads[0]["start"])  # seconds
        end = int(vads[-1]["end"])  # seconds
        if debug:
            logging.info(f" start {start} seconds")
            logging.info(f" end {end} seconds")
    waveform = waveform[:,
               librosa.time_to_samples(start, sr=sample_rate):librosa.time_to_samples(end, sr=sample_rate)]
    return waveform
