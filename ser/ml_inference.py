import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pathlib
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # no GUI — must be before pyplot import
import matplotlib.pyplot as plt
import tensorflow as tf

_BASE = pathlib.Path(__file__).resolve().parent.parent  # mood_detection/
MODEL_PATH   = _BASE / 'emotion_model.h5'
CLASSES_PATH = _BASE / 'classes.npy'

SAMPLE_RATE = 22050
MAX_SAMPLES = SAMPLE_RATE * 3  # 3 seconds
N_MELS = 128

def _load_model(num_classes: int):
    # Rebuild the exact architecture from Emotions.py to avoid Keras 3 / h5 config
    # parsing issues ('quantization_config' baked into older saved models).
    # Then load only the weight tensors from the h5 file — no config parsing needed.
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                               input_shape=(128, 130, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.load_weights(str(MODEL_PATH))
    return model


# Load once at module level — not per request
_classes = np.load(str(CLASSES_PATH), allow_pickle=True)
_model = _load_model(num_classes=len(_classes))


def _preprocess(audio_path: str) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    if len(y) == 0:
        raise ValueError('Audio file is empty.')
    if len(y) < SAMPLE_RATE * 0.5:
        raise ValueError('Audio too short — minimum 0.5 seconds required.')
    if len(y) > MAX_SAMPLES:
        y = y[:MAX_SAMPLES]
    else:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)), 'constant')
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, fmax=8000)
    return librosa.power_to_db(mel, ref=np.max)


def predict_emotion(audio_path: str) -> dict:
    """
    Returns:
        {
            'emotion': str,
            'confidence': float (0-1),
            'all_probs': {label: prob, ...},
            'mel_spec_db': np.ndarray
        }
    Raises ValueError for bad audio.
    """
    mel_spec_db = _preprocess(audio_path)
    X = mel_spec_db[np.newaxis, ..., np.newaxis]
    probs = _model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        'emotion': str(_classes[idx]),
        'confidence': float(probs[idx]),
        'all_probs': {str(c): round(float(p), 4) for c, p in zip(_classes, probs)},
        'mel_spec_db': mel_spec_db,
    }


def generate_spectrogram_image(mel_spec_db: np.ndarray, save_path: str) -> None:
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time',
                             sr=SAMPLE_RATE, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(save_path, dpi=80)
    plt.close()
