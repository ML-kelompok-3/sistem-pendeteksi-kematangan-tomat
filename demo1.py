import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Memuat model yang sudah dilatih
model = load_model('tomato_model.keras')

def detect_ripe_tomatoes(frame, model):
    # Ubah bingkai menjadi ruang warna HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the red color (ripe tomatoes)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # beri masking untuk warna merah
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Cari kontur warna pada hasil masking
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ulangi kontur
    for contour in contours:
        # Kalkulasi luas tiap kontur
        area = cv2.contourArea(contour)

        # Tetapkan ambang batas pada area tersebut untuk menyaring kontur kecil
        if area > 100:
            # Cari titik tengah kontur
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Ekstrak tomat dari bingkai dan ubah ukurannya sesuai ukuran yang diharapkan model
            tomato = frame[y:y+h, x:x+w]
            tomato = cv2.resize(tomato, (224, 224))

            # Normalisasi gambar menjadi rentang 0 sampai 1
            tomato = tomato / 255.0

            # Tambahkan dimensi ekstra untuk ukuran batch
            tomato = np.expand_dims(tomato, axis=0)

            # Gunakan model tersebut untuk memprediksi apakah tomat sudah matang atau mentah
            prediction = model.predict(tomato)[0][0]

            # Prediksinya adalah angka antara 0 dan 1 karena fungsi aktivasi sigmoid
            label = "Ripe" if prediction > 0.5 else "Unripe"

            # Gambarlah label pada bingkai
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Tampilkan hasil
    cv2.imshow("Ripe Tomato Detection", frame)

# Buka koneksi ke webcam (indeks kamera 0 secara default)
cap = cv2.VideoCapture(0)

while True:
    # Tangkap bingkai demi bingkai
    ret, frame = cap.read()

    # Periksa apakah frame berhasil ditangkap
    if not ret:
        print("Error: Couldn't read frame")
        break

    # Lakukan deteksi tomat matang pada bingkai
    detect_ripe_tomatoes(frame, model)

    # Putuskan loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()