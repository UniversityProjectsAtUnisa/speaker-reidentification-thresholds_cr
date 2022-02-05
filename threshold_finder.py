# noqa
import os  # nopep8
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # nopep8

from pathlib import Path
import speech_recognition as sr
from identification.deep_speaker.audio import get_mfcc
from identification.deep_speaker.model import get_deep_speaker
from identification.utils import batch_cosine_similarity, dist2id
import numpy as np


SAMPLE_RATE = 16000
STRICT_TH = 0.75
PERMESSIVE_TH = 0.50

PROJ_PATH = Path(__file__).absolute().parent
MODEL_PATH = PROJ_PATH.joinpath("deep_speaker.h5")
SAMPLES_PATH = PROJ_PATH.joinpath("samples")

ERROR_TEMPLATE = "Audio file \"{}\" with wrong samplerate: should be {} but got {}"


class AudioIdentifier:
    def __init__(self, model_path):
        self._model = get_deep_speaker(model_path)

    def extract_embeddings(self, audio_data) -> np.ndarray:
        # Processing
        ukn = get_mfcc(audio_data, SAMPLE_RATE)
        # Prediction
        return self._model.predict(np.expand_dims(ukn, 0))


def load_audio(filename, r):
    with sr.AudioFile(filename) as s:
        rate = s.audio_reader.getframerate()
        assert rate == SAMPLE_RATE, ERROR_TEMPLATE.format(filename, SAMPLE_RATE, rate)
        audio = r.record(s)
    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
    return audio_data.astype(np.float32, order="C") / 32768.0


def find_id(ukn, X, y, threshold, mode="avg"):
    if len(X) > 0:
        # Distance between the sample and the support set
        emb_voice = np.repeat(ukn, len(X), 0)
        cos_dist = batch_cosine_similarity(np.array(X), emb_voice)
        # Matching
        return dist2id(cos_dist, y, threshold, mode=mode)
    return None


def main():
    # LOAD IDENTITIES AND WAV SAMPLES PATH
    identities = [child for child in SAMPLES_PATH.iterdir() if child.is_dir()]
    identity_to_files = {child: [s for s in child.iterdir() if s.suffix == '.wav'] for child in identities}
    identity_to_files = {k.stem: v for k, v in identity_to_files.items()}
    print(identity_to_files)

    # LOAD AUDIO DATA FROM WAV SAMPLES
    r = sr.Recognizer()
    identity_to_audio = {k: [load_audio(str(s_path), r) for s_path in v] for k, v in identity_to_files.items()}
    print({k: [e.shape for e in v] for k, v in identity_to_audio.items()})

    # LOAD EMBEDDINGS DATA FROM AUDIO DATA
    ai = AudioIdentifier(MODEL_PATH)
    identity_to_embeddings = {k: [ai.extract_embeddings(e) for e in v] for k, v in identity_to_audio.items()}
    print({k: [e.shape for e in v] for k, v in identity_to_embeddings.items()})

    # CREATE TRAINING DATA
    X, y = [], []
    for identity, embeddings_list in identity_to_embeddings.items():
        X.extend(embeddings_list)
        y.extend([identity] * len(embeddings_list))
    
    # TODO: la parte importante lol


if __name__ == '__main__':
    main()
