# noqa
import os  # nopep8
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # nopep8

from pathlib import Path
import speech_recognition as sr
from identification.deep_speaker.audio import get_mfcc
from identification.deep_speaker.model import get_deep_speaker
from identification.utils import batch_cosine_similarity, dist2id
import numpy as np
from tqdm import tqdm
import pickle

TEST_SIZE = 6

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


def create_datasets(i2e, test_size):
    X_train, y_train = [], []
    X_test, y_test = [], []
    for identity, embeddings_list in i2e.items():
        X_train.extend(embeddings_list[test_size:])
        X_test.extend(embeddings_list[:test_size])
        y_train.extend([identity] * (len(embeddings_list)-test_size))
        y_test.extend([identity] * test_size)

    return X_train, y_train, X_test, y_test


def predict_single(sample, X_train, y_train, th, mode):
    return find_id([sample], X_train, y_train, th, mode)


def predict(samples, X_train, y_train, th, mode):
    return [predict_single(sample, X_train, y_train, th, mode) for sample in samples]


def score(preds, y_test):
    assert len(preds) == len(y_test)
    right = sum([int(p == y) for p, y in zip(preds, y_test)])
    return right/len(y_test)


def i2e_to_dataset(i2e, type="strict"):
    if type == "strict":
        X, y = [], []
        for k, embeddings_list in i2e.items():
            X.extend(embeddings_list)
            y.extend([k] * len(embeddings_list))
        return X, y
    elif type == "permissive":
        pass
    return [], []


def train_strict_th(X_train, y_train, target_score=1.0, mode="avg", rounds=10):
    def try_threshold(th):
        preds = []
        for i in range(len(X_train)):
            sample, _y = X_train.pop(i), y_train.pop(i)
            ans = predict_single(sample, X_train, y_train, th, mode)
            preds.append(ans)
            y_train.insert(i, _y)
            X_train.insert(i, sample)
        ans = score(preds, y_train)
        print(f"{th=} score={ans}")
        return ans

    left = 0.0
    right = 1.0
    best_th = 0.0
    for _ in tqdm(range(rounds), desc="Optimizing threshold"):
        middle = (left+right)/2
        middle_score = try_threshold(middle)

        if middle_score >= target_score:
            left = middle
            if middle > best_th:
                best_th = middle
        else:
            right = middle
    return best_th

    ###
    for th in ths:
        cur = try_threshold(th)
        if cur > best_score:
            best = th
        print(f"{th=} score={cur}")
    return best


def test_strict_th(i2e_test, X_train, y_train, strict_th, mode):
    X_test, y_test = i2e_to_dataset(i2e_test, "strict")
    preds = predict(X_test, X_train, y_train, strict_th, mode)
    return score(preds, y_test)


def split_i2e(i2e, test_size):
    i2e_train, i2e_test = {}, {}
    for k, embeddings in i2e.items():
        i2e_train[k] = embeddings[test_size:]
        i2e_test[k] = embeddings[:test_size]
    return i2e_train, i2e_test


def main():
    cache_file = 'i2e.pkl'
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            identity_to_embeddings = pickle.load(f)
    else:
        # LOAD IDENTITIES AND WAV SAMPLES PATH
        identities = [child for child in SAMPLES_PATH.iterdir() if child.is_dir()]
        identity_to_files = {child: [s for s in child.iterdir() if s.suffix == '.wav'] for child in identities}
        identity_to_files = {k.stem: v for k, v in identity_to_files.items()}

        # LOAD AUDIO DATA FROM WAV SAMPLES
        r = sr.Recognizer()
        identity_to_audio = {k: [load_audio(str(s_path), r) for s_path in v] for k, v in identity_to_files.items()}
        # print({k: [e.shape for e in v] for k, v in identity_to_audio.items()})

        # LOAD EMBEDDINGS DATA FROM AUDIO DATA
        ai = AudioIdentifier(MODEL_PATH)
        identity_to_embeddings = {k: [ai.extract_embeddings(e)[0] for e in v] for k, v in identity_to_audio.items()}
        # print({k: [e.shape for e in v] for k, v in identity_to_embeddings.items()})

        with open(cache_file, "wb") as f:
            pickle.dump(identity_to_embeddings, f)

    i2e_train, i2e_test = split_i2e(identity_to_embeddings, TEST_SIZE)

    X_train, y_train = i2e_to_dataset(i2e_train, "strict")

    modes = ['avg', 'max', 'min']
    for mode in modes:
        strict_th = train_strict_th(X_train, y_train, 0.9, mode)
        s = test_strict_th(i2e_test, X_train, y_train, strict_th, mode)
        print(f"{mode=} {strict_th=} scored {s}")
    # permissive_th = train_permissive_th(i2e_train)
    # test_permissive_th(i2e_test)

    # CREATE TRAINING DATA
    X_train, y_train, X_test, y_test = create_datasets(identity_to_embeddings, TEST_SIZE)


if __name__ == '__main__':
    main()
