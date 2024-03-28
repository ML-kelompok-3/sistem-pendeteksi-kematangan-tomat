# Projek sistem pendeteksi kematangan tomat

## Files

1. **`tensor.py`**: This file contains the code for building a machine learning model using TensorFlow, Keras, and Scikit-learn. The model is trained to detect ripe tomatoes and is saved as `tomato_model.h5`.

2. **`demo1.py`**: The real-time prediction demo file. It utilizes computer vision techniques to detect and predict ripe tomatoes using the pre-trained machine learning model (`tomato_model.h5`).

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
