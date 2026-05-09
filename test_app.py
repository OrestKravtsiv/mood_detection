import os
import time
import datetime
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import keyboard
import tensorflow as tf
from scipy.io import wavfile

# Вимикаємо зайві попередження TensorFlow у консолі
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# --- НАЛАШТУВАННЯ (мають збігатися з тренуванням) ---
SAMPLE_RATE = 22050                                                                                                                     
DURATION = 3
MAX_SAMPLES = SAMPLE_RATE * DURATION
N_MELS = 128
SAVE_DIR = "collected_data" # Папка для збереження нових записів

# Створюємо папку, якщо її немає
os.makedirs(SAVE_DIR, exist_ok=True)

# --- СЛОВНИК ПЕРЕКЛАДУ ЕМОЦІЙ ---
EMOTIONS_UK = {
    'neutral': 'НЕЙТРАЛЬНИЙ 😐',
    'calm': 'СПОКІЙНИЙ 😌',
    'happy': 'РАДІСНИЙ 😄',
    'sad': 'СУМНИЙ 😢',
    'angry': 'ЗЛИЙ 😡',
    'fearful': 'НАЛЯКАНИЙ 😨',
    'disgust': 'ОГИДА 🤢',
    'surprised': 'ЗДИВОВАНИЙ 😲'
}

def preprocess_live_audio(audio_data):
    """
    Обробляє записаний шматок аудіо так само, як ми робили це при створенні датасету.
    """
    if len(audio_data) > MAX_SAMPLES:
        audio_data = audio_data[:MAX_SAMPLES]
    else:
        padding_size = MAX_SAMPLES - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding_size), 'constant')
    
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=SAMPLE_RATE, n_mels=N_MELS, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    X_input = mel_spec_db[np.newaxis, ..., np.newaxis]
    return X_input

def main():
    print("[INFO] Завантаження моделі... Зачекайте.")
    try:
        model = tf.keras.models.load_model('emotion_model.h5')
        classes = np.load('classes.npy')
    except Exception as e:
        print("Помилка! Не знайдено файли моделі. Спочатку запустіть скрипт навчання.")
        return

    print("\n" + "="*50)
    print(f"МОДЕЛЬ ГОТОВА! 🎤 (Нові записи зберігатимуться у папку '{SAVE_DIR}')")
    print("Натисніть та УТРИМУЙТЕ клавішу ПРОБІЛ, щоб сказати щось.")
    print("Натисніть 'ESC' для виходу з програми.")
    print("="*50 + "\n")

    while True:
        try:
            event = keyboard.read_event()
            
            if event.name == 'esc':
                print("Вихід з програми...")
                break
                
            if event.name == 'space' and event.event_type == 'down':
                print("🔴 ЙДЕ ЗАПИС... (Говоріть, поки тримаєте пробіл)")
                
                recorded_frames = []
                
                def audio_callback(indata, frames, time, status):
                    if status:
                        print(status)
                    recorded_frames.append(indata.copy())
                
                with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
                    while keyboard.is_pressed('space'):
                        time.sleep(0.01)
                
                print("🟢 ЗАПИС ЗУПИНЕНО. Обробка...")
                
                if len(recorded_frames) > 0:
                    audio_data = np.concatenate(recorded_frames, axis=0).flatten()
                    
                    if len(audio_data) < SAMPLE_RATE * 0.5:
                        print("Запис був занадто коротким! Спробуйте ще раз.\n")
                        continue
                    
                    # 1. Передбачення
                    X_input = preprocess_live_audio(audio_data)
                    predictions = model.predict(X_input, verbose=0) 
                    
                    predicted_index = np.argmax(predictions[0])
                    emotion_english = classes[predicted_index]
                    emotion_ukrainian = EMOTIONS_UK.get(emotion_english, emotion_english.upper())
                    confidence = predictions[0][predicted_index] * 100
                    
                    print(f"👉 Ваш настрій: ** {emotion_ukrainian} ** (Впевненість: {confidence:.1f}%)")

                    # ==========================================
                    # 2. ЗБЕРЕЖЕННЯ ДАНИХ ДЛЯ ПЕРЕТРЕНУВАННЯ
                    # ==========================================
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_filename = f"{emotion_english}_{timestamp}"
                    
                    # Зберігаємо WAV файл
                    wav_path = os.path.join(SAVE_DIR, f"{base_filename}.wav")
                    wavfile.write(wav_path, SAMPLE_RATE, audio_data)
                    
                    # Зберігаємо PNG картинку спектрограми
                    png_path = os.path.join(SAVE_DIR, f"{base_filename}.png")
                    plt.figure(figsize=(10, 4))
                    # Витягуємо 2D матрицю з X_input (який має форму 1, 128, 130, 1)
                    mel_2d = X_input[0, :, :, 0] 
                    librosa.display.specshow(mel_2d, y_axis='mel', x_axis='time', sr=SAMPLE_RATE, fmax=8000)
                    plt.colorbar(format='%+2.0f dB')
                    plt.title(f'Настрій: {emotion_ukrainian} ({confidence:.1f}%)')
                    plt.tight_layout()
                    plt.savefig(png_path)
                    plt.close() # Обов'язково закриваємо, щоб не забити пам'ять

                    print(f"💾 Файли збережено: {wav_path} та {png_path}\n")

                else:
                    print("Нічого не записалось.\n")
                    
                time.sleep(0.5)
                print("Натисніть та утримуйте ПРОБІЛ для нового запису...")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()