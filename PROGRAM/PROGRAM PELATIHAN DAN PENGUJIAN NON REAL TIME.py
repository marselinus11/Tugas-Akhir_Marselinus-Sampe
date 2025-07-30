#########################################################################
    #TUGAS AKHIR
    #MARSELINUS SAMPE
    #215114009
    #PRODI TEKNIK ELEKTRO
#########################################################################

import numpy as np
import os
from scipy.io import wavfile
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Path folder data dan penyimpanan model
folder_path = 'D:/TA SEMENTARA/DATA'
model_save_path = 'D:/TA SEMENTARA\data model/model_bpnn_segment256.h5'

# Label nada
label_mapping = {"C_tinggi": 7, "C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
label_names = list(label_mapping.keys())

# Menyiapkan data
X, y = [], []

for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        for label in label_mapping:
            if filename.startswith(label):
                sr, data = wavfile.read(os.path.join(folder_path, filename))
                typ = data.astype(np.float16)  # Pencacahan data
                
                                      # PREPROCESSING    
             # Normalisasi Ke 1
                x1 = typ / np.max(np.abs(typ))   
                
             # tresholding dengan 0,5 
                th = 0.5  
                
             # Pemotongan sinyal   
                x2a = np.where(x1 > th)[0] 
                if len(x2a) == 0:
                    break
                x2b = x2a[0]
                x2 = x1[x2b:]
                
             # Frame blocking
                pjg_frame = 256
                if len(x2) < pjg_frame:
                    break
                x3 = x2[:pjg_frame] 
                
             # Normalisasi ke 2
                x4 = x3 / np.max(np.abs(x3))  
                
             # Windowing Hamming   
                wndw_haming = np.hamming(pjg_frame)
                x5 = x4 * wndw_haming  

                # ======== FFT dan EKSTRAKSI FITUR ========
                dft = np.fft.fft(x5)
                x6 = np.abs(dft[:pjg_frame // 2])
                x6[0] = 0  # Komponen DC di-nol-kan
                x6 = x6 / np.max(x6)

                # Segment averaging: 128 sample 
                num_segments = 32
                samples_per_segment = 4
                if len(x6) != num_segments * samples_per_segment:
                    break
                segments = x6.reshape(num_segments, samples_per_segment)
                features = np.mean(segments, axis=1)

                # Simpan fitur dan label
                X.append(features)
                y.append(label_mapping[label])
                break

# Konversi data ke array numpy
X = np.array(X)
y = np.array(y)
y_cat = to_categorical(y, num_classes=8)

# Split latih-uji per kelas
X_train, X_test, y_train, y_test = [], [], [], []
for i in range(8):
    data_i = shuffle([x for x, label in zip(X, y) if label == i], random_state=42)
    if len(data_i) < 30:
        raise ValueError(f"Data kelas '{label_names[i]}' kurang dari 30.")
    X_train.extend(data_i[:20])
    X_test.extend(data_i[20:30])
    y_train.extend([i] * 20)
    y_test.extend([i] * 10)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train_cat = to_categorical(y_train, 8)
y_test_cat = to_categorical(y_test, 8)

# ======== MODEL BPNN ========
model = Sequential([
    Dense(8, activation='relu', input_shape=(32,)),
    Dense(8, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Latih model
model.fit(X_train, y_train_cat, epochs=30, batch_size=2, validation_data=(X_test, y_test_cat))

# Simpan model
model.save(model_save_path)

# Evaluasi
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels, target_names=label_names))

cm = confusion_matrix(y_test, y_pred_labels)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
