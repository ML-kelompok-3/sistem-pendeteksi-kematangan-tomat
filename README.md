# Projek sistem pendeteksi kematangan tomat

## Files

1. **`tensor.py`**: File ini berisi kode untuk membuat model pembelajaran mesin menggunakan TensorFlow, Keras, dan Scikit-learn. Model dilatih untuk mendeteksi tomat matang dan disimpan sebagai `tomato_model.keras`.

2. **`demo1.py`**: File demo prediksi real-time. Ini menggunakan teknik visi komputer untuk mendeteksi dan memprediksi tomat matang menggunakan model pembelajaran mesin yang telah dilatih sebelumnya (`tomato_model.h5`).

## Technologies Used

- TensorFlow
- Keras
- Scikit-learn
- Computer Vision
- Python

## Instruksi untuk menjalankan projek

1. Clone projek:

   ```bash
   git clone https://github.com/your-username/tomato-harvesting-robot.git
   cd tomato-ripeness-detector
   ```

2. Jalankan script `tensor.py` untuk melatih model machine learning:

   ```bash
   python tensor.py
   ```

   Nanti akan muncul file `tomato_model.keras`.

3. Jalankan demo prediksi waktu nyata menggunakan `demo1.py`:

   ```bash
   python demo1.py
   ```

   Pastikan library yang diperlukan diinstal dengan menggunakan:

   ```bash
   pip install -r requirements.txt
   ```

   Demo ini akan menggunakan teknik visi komputer untuk memprediksi tomat matang secara real-time.
