import os
import librosa
import numpy as np
import matplotlib.pyplot as plt # ДОДАНО: бібліотека для малювання
import librosa.display       # ДОДАНО: спеціально для відображення аудіо
from tqdm import tqdm

# --- Налаштування (ті самі, що й раніше) ---
DATASET_PATH = "./samples"  
OUTPUT_X_PATH = "X_mel_specs.npy" 
OUTPUT_Y_PATH = "y_labels.npy"    
SAMPLE_RATE = 22050     
DURATION = 3            
MAX_SAMPLES = SAMPLE_RATE * DURATION 
N_MELS = 128            
EMOTIONS = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}

def process_audio(file_path):
    # (Функція залишається без змін)
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(y) > MAX_SAMPLES:
        y = y[:MAX_SAMPLES]
    else:
        padding_size = MAX_SAMPLES - len(y)
        y = np.pad(y, (0, padding_size), 'constant')
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def create_dataset(dataset_path):
    X, y = [], []
    file_list = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_list.append(os.path.join(root, file))
    
    print(f"Знайдено {len(file_list)} аудіофайлів. Починаємо обробку...")
    
    # Прапорець, щоб зберегти лише першу спектрограму
    saved_first_image = False 
    
    for file_path in tqdm(file_list):
        file_name = os.path.basename(file_path)
        parts = file_name.split('-')
        emotion_code = parts[2]
        emotion_label = EMOTIONS.get(emotion_code)
        
        mel_spectrogram = process_audio(file_path)
        
        # --- НОВИЙ БЛОК: Збереження першої спектрограми як PNG ---
        if not saved_first_image:
            plt.figure(figsize=(10, 4)) # Створюємо фігуру певного розміру
            # Спеціальна функція librosa для малювання спектрограм
            librosa.display.specshow(mel_spectrogram, y_axis='mel', x_axis='time', sr=SAMPLE_RATE, fmax=8000)
            plt.colorbar(format='%+2.0f dB') # Додаємо кольорову шкалу
            plt.title(f'Mel-спектрограма: {emotion_label}') # Заголовок з емоцією
            plt.tight_layout() # Оптимізуємо відступи
            
            # Зберігаємо як PNG файл
            plt.savefig('first_spectrogram.png') 
            print(f"\n[INFO] Перша спектрограма збережена як 'first_spectrogram.png' для перевірки.")
            
            plt.close() # Обов'язково закриваємо фігуру, щоб не переповнити пам'ять!
            saved_first_image = True # Опускаємо прапорець
        # -----------------------------------------------------

        X.append(mel_spectrogram)
        y.append(emotion_label)
        
    return np.array(X), np.array(y)

# --- Запуск пайплайну (без змін) ---
if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"Помилка: Папку {DATASET_PATH} не знайдено!")
    else:
        X, y = create_dataset(DATASET_PATH)
        np.save(OUTPUT_X_PATH, X)
        np.save(OUTPUT_Y_PATH, y)
        print("\nГотово! Датасет успішно створено.")
        print(f"Розмір матриці ознак (X): {X.shape}") 
        print(f"Розмір матриці міток (y): {y.shape}")