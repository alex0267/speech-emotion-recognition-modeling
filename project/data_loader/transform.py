import librosa

class Spectrogram:
    """sound file to mel spectrogram.

    Args:
        pathfile (str): path to input file.
    """

    def __init__(self, pathfile):
        assert isinstance(pathfile, str)
        self.pathfile = pathfile

    def __call__(self, n_mels=56):
        samples, sample_rate = librosa.load(self.pathfile)
        img = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=n_mels)
        return {'image': img}