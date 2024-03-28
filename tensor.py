import cv2
import numpy as np
import os
import glob
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input

# Function to load and preprocess images
def load_data(path):
    images = []
    labels = []

    # Membuat gambar tomat matang
    for file in glob.glob(os.path.join(path, 'Images', 'Riped tomato_*.jpeg')):
        img = cv2.imread(file)
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(1)  # label 1 untuk tomat matang

    # Membuat gambar tomat mentah
    for file in glob.glob(os.path.join(path, 'Images', 'Unriped tomato_*.jpeg')):
        img = cv2.imread(file)
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(0) # label 0 untuk tomat mentah

    # ubah menjadi numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Split data menjadi training and testing
    train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Normaliisasi gambar dalam rentang 0-1
    train_data = train_data.astype('float32') / 255.0
    test_data = test_data.astype('float32') / 255.0

    return train_data, test_data, train_labels, test_labels

# Function untuk membuat model
def create_model():
    model = keras.Sequential([
        Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')  
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Function untuk melatih model
def train_model(model, train_data, train_labels, test_data, test_labels):
    model.fit(train_data, train_labels, epochs=100, validation_data=(test_data, test_labels))
    
    # Simpan file ke dalam format Keras native
    model.save('tomato_model.keras')
    
    return model

# Function untuk evaluasi model
def evaluate_model(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')

# Memuat data
train_data, test_data, train_labels, test_labels = load_data('dataset')

# Membuat model
model = create_model()

# Melatih model
model = train_model(model, train_data, train_labels, test_data, test_labels)

# Mengevaluasi model
evaluate_model(model, test_data, test_labels)
