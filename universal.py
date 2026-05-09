import os
import librosa
import numpy as np
from tqdm import tqdm
import re # Додано для розумного пошуку в тексті

# --- НАЛАШТУВАННЯ ---
DATASETS_DIR = "./datasets"  # Головна папка з усіма датасетами
OUTPUT_X_PATH = "X_mel_specs.npy" 
OUTPUT_Y_PATH = "y_labels.npy"    
SAMPLE_RATE = 22050     
DURATION = 3            
MAX_SAMPLES = SAMPLE_RATE * DURATION 
N_MELS = 128            

# Словник для об'єднання всіх емоцій до єдиного стандарту
VALID_EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def process_audio(file_path):
    """Обробляє один аудіофайл і повертає спектрограму."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(y) == 0:
            return None 
            
        if len(y) > MAX_SAMPLES:
            y = y[:MAX_SAMPLES]
        else:
            padding_size = MAX_SAMPLES - len(y)
            y = np.pad(y, (0, padding_size), 'constant')
            
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    except Exception:
        return None

def extract_emotion(file_name):
    """
    НОВА РОЗУМНА ЛОГІКА: Аналізує патерн самої назви файлу.
    """
    file_name = file_name.lower()

    # 1. RAVDESS (Формат: 03-01-05-01-01-01-01.wav - має 6 дефісів)
    if '-' in file_name:
        parts = file_name.split('-')
        if len(parts) == 7:
            ravdess_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
            return ravdess_map.get(parts[2], None)

    # 2. CREMA-D (Формат: 1001_DFA_ANG_XX.wav - має 3 підкреслення, перше слово - цифри актора)
    parts = file_name.split('_')
    if len(parts) == 4 and parts[0].isdigit():
        crema_map = {'neu': 'neutral', 'hap': 'happy', 'sad': 'sad', 'ang': 'angry', 'fea': 'fearful', 'dis': 'disgust'}
        return crema_map.get(parts[2], None)

    # 3. SAVEE (Формат: DC_a01.wav або JE_su15.wav - Ініціали_ЕмоціяЦифри)
    match = re.match(r'^[a-z]+_([a-z]+)[0-9]+\.wav', file_name)
    if match:
        savee_map = {'a': 'angry', 'd': 'disgust', 'f': 'fearful', 'h': 'happy', 'n': 'neutral', 'sa': 'sad', 'su': 'surprised'}
        return savee_map.get(match.group(1), None)

    # 4. TESS (Формат: OAF_happy_dog.wav або просто слово емоції в назві файлу)
    if 'happy' in file_name: return 'happy'
    if 'sad' in file_name: return 'sad'
    if 'angry' in file_name: return 'angry'
    if 'fear' in file_name: return 'fearful'
    if 'disgust' in file_name: return 'disgust'
    if 'ps' in file_name or 'surprised' in file_name: return 'surprised'
    if 'neutral' in file_name: return 'neutral'

    return None # Якщо нічого не підійшло

def create_unified_dataset(dataset_dir):
    X, y = [], []
    file_list = []
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                file_list.append(os.path.join(root, file))
    
    print(f"[INFO] Знайдено {len(file_list)} аудіофайлів у різних датасетах.")
    print("[INFO] Починаємо велику обробку... Це може зайняти час.")
    
    skipped_files = 0
    unrecognized_names = set() # Зберігаємо кілька назв пропущених файлів для діагностики
    
    for file_path in tqdm(file_list):
        file_name = os.path.basename(file_path)
        
        # Витягуємо емоцію лише за назвою файлу
        emotion_label = extract_emotion(file_name)
        
        if emotion_label in VALID_EMOTIONS:
            mel_spectrogram = process_audio(file_path)
            if mel_spectrogram is not None:
                X.append(mel_spectrogram)
                y.append(emotion_label)
            else:
                skipped_files += 1
        else:
            skipped_files += 1
            if len(unrecognized_names) < 10:
                unrecognized_names.add(file_name)
            
    return np.array(X), np.array(y), skipped_files, unrecognized_names

if __name__ == "__main__":
    if not os.path.exists(DATASETS_DIR):
        print(f"Помилка: Папку {DATASETS_DIR} не знайдено!")
    else:
        X, y, skipped, unrec_names = create_unified_dataset(DATASETS_DIR)
        
        if len(X) > 0:
            np.save(OUTPUT_X_PATH, X)
            np.save(OUTPUT_Y_PATH, y)
            print("\n" + "="*40)
            print("🎉 ГОТОВО! СУПЕР-ДАТАСЕТ СТВОРЕНО.")
            print(f"✅ Успішно оброблено: {len(X)} файлів")
            print(f"⚠️ Пропущено: {skipped} файлів")
            
            if skipped > 0:
                print("\nПриклади назв файлів, які не вдалося розпізнати:")
                for name in unrec_names:
                    print(f" - {name}")
                    
            print(f"\nРозмір матриці ознак (X): {X.shape}") 
            print(f"Розмір матриці міток (y): {y.shape}")
            print("="*40)