import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def build_emotion_cnn(input_shape, num_classes):
    """
    Будує згорткову нейронну мережу (CNN) для класифікації спектрограм.
    """
    model = models.Sequential([
        # --- ЗГОРТКОВИЙ БЛОК 1 ---
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)), 
        
        # --- ЗГОРТКОВИЙ БЛОК 2 ---
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # --- ЗГОРТКОВИЙ БЛОК 3 ---
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # --- ПЕРЕХІД ДО КЛАСИФІКАТОРА ---
        layers.Flatten(), 
        
        # --- ПОВНІСТЮ З'ЄДНАНІ ШАРИ (КЛАСИФІКАТОР) ---
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), 
        
        # Вихідний шар
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    print("[INFO] Завантаження даних...")
    # 1. Завантажуємо дані, підготовлені першим скриптом
    X = np.load("X_mel_specs.npy")
    y_text = np.load("y_labels.npy")
    
    # 2. Підготовка X (Ознак)
    # CNN очікує картинку у форматі (висота, ширина, канали). 
    # Наш X зараз має розмір (кількість_зразків, 128, ширина). Додаємо 1 канал (чорно-біле).
    X = X[..., np.newaxis] 
    
    # 3. Підготовка y (Міток)
    # Перетворюємо текстові мітки (напр., 'happy', 'sad') у числа (0, 1, 2...)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_text)
    num_classes = len(label_encoder.classes_)
    
    print(f"[INFO] Знайдено {num_classes} класів емоцій: {label_encoder.classes_}")
    
    # 4. Розбиття на тренувальну та тестову вибірки (80% на навчання, 20% на перевірку)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Динамічно визначаємо розмір вхідної картинки (без кількості зразків)
    input_shape = X_train.shape[1:] 
    print(f"[INFO] Розмірність входу нейромережі: {input_shape}")
    
    # 5. Ініціалізація моделі
    model = build_emotion_cnn(input_shape=input_shape, num_classes=num_classes)
    model.summary()
    
    # 6. НАВЧАННЯ МОДЕЛІ
    print("\n[INFO] Починаємо навчання моделі...")
    history = model.fit(
        X_train, y_train, 
        epochs=100,              # Кількість проходів по всьому датасету (можна змінити)
        batch_size=16,          # Кількість спектрограм за один крок
        validation_data=(X_test, y_test) # Дані для перевірки на кожній епосі
    )
    
    # 7. Збереження навченої моделі та енкодера (щоб знати, яке число якій емоції відповідає)
    model.save('emotion_model.h5')
    np.save('classes.npy', label_encoder.classes_)
    print("\n[INFO] Навчання завершено! Модель збережено як 'emotion_model.h5'")
