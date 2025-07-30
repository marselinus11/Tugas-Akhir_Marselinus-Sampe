#########################################################################
    #TUGAS AKHIR
    #MARSELINUS SAMPE
    #215114009
    #PRODI TEKNIK ELEKTRO
#########################################################################

import numpy as np
import sounddevice as sd
import FreeSimpleGUI as sg
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ====== KONFIGURASI DASAR ======
FREKUENSI_SAMPLING = 4800
DURASI_REKAMAN = 1
LOKASI_MODEL = 'D:/DATA MODEL/model_bpnn_segment256.h5'

# Muat model BPNN
model = load_model(LOKASI_MODEL)

# Label nada
label_nada = {
    0: "C", 1: "D", 2: "E", 3: "F", 4: "G", 5: "A", 6: "B", 7: "C_tinggi"
}

# ====== FUNGSI DETEKSI ======
def prediksi_nada(audio):
    audio = audio.astype(np.float32)
    
    # Normalisasi 1
    audio = audio / np.max(np.abs(audio))

    # Threshold untuk deteksi suara aktif
    th = 0.5
    idx_awal = np.where(np.abs(audio) > th)[0]
    if len(idx_awal) == 0 or len(audio[idx_awal[0]:]) < 256:
        return "Tidak valid", audio, None, None
    # Potong sinyal
    x2 = audio[idx_awal[0]:][:256]

    # Normalisasi 2 
    x3 = x2 / np.max(np.abs(x2))
    
    # windowing Hamming 
    x4 = x3 * np.hamming(256)
    
    #  FFT
    dft = np.fft.fft(x4)
    x5 = np.abs(dft[:128])
    x5[0] = 0
    x5 = x5 / np.max(x5)

    # Ekstraksi ciri
    segments = x5.reshape(32, 4)
    fitur = np.mean(segments, axis=1).reshape(1, -1)

    # Prediksi Nada (BPNN)
    pred = model.predict(fitur, verbose=0)
    label = np.argmax(pred)

    return label_nada[label], audio, x5, fitur.flatten()

def draw_plot(canvas_elem, data, title, plot_type='line'):
    fig = plt.figure(figsize=(4,3.5))

    if plot_type == 'bar':
        plt.bar(np.arange(len(data)), data)
    else:
        if title == "Rekaman Suara":
           plt.plot(data)  
        else:
            plt.plot(data)

    plt.title(title)

    # Atur label sumbu berdasarkan judul
    if title == "Rekaman Suara":
        plt.xlabel("Data Ke ")
        plt.ylabel("Amplitudo")
    elif title == "FFT":
        plt.xlabel("Frekuensi (Bin)")
        plt.ylabel("Amplitudo")
    elif title == "Ekstraksi Ciri":
        plt.xlabel("Data Ke")
        plt.ylabel("Rata-rata")

    plt.tight_layout()
    canvas_elem.TKCanvas.delete("all")
    fig_canvas = FigureCanvasTkAgg(fig, canvas_elem.TKCanvas)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
    return fig_canvas

# ====== GUI LAYOUT ======
sg.theme('lightGrey1')
layout = [
    [sg.Text("PENGENALAN NADA - REAL TIME", font=('Arial', 14, 'bold'), justification='center', expand_x=True)],
    [
        sg.Push(),
        sg.Button("Reset", size=(10, 1)),
        sg.Button("Selesai", size=(10, 1)),
        sg.Text('Nada Dikenali:', size=(12, 1)),
        sg.Text('', key='-OUTPUT-', size=(15, 1), text_color='Black', font=('Arial', 12, 'bold'), relief='sunken', justification='center'),
        sg.Push()
    ],
    [
        sg.Column([[sg.Canvas(key='-PLOT1-', size=(300, 200))], [sg.Text("Rekaman Suara")]]),
        sg.Column([[sg.Canvas(key='-PLOT3-', size=(300, 200))], [sg.Text("FFT")]]),
        sg.Column([[sg.Canvas(key='-PLOT2-', size=(300, 200))], [sg.Text("Ekstraksi Ciri")]])
    ]
]

window = sg.Window("Deteksi Nada Pianika", layout, finalize=True)
canvas1, canvas2, canvas3 = None, None, None

# ====== LOOP ======
try:
    while True:
        event, _ = window.read(timeout=100)
        if event == sg.WINDOW_CLOSED or event == 'Selesai':
            break

        
        if event == 'Reset':
            window['-OUTPUT-'].update("")
            for canvas in [canvas1, canvas2, canvas3]:
                if canvas:
                    canvas.get_tk_widget().destroy()
            canvas1, canvas2, canvas3 = None, None, None

        # Langsung rekam 1 detik tiap loop
        rekaman = sd.rec(int(FREKUENSI_SAMPLING * DURASI_REKAMAN), samplerate=FREKUENSI_SAMPLING, channels=1)
        sd.wait()
        
        rekaman = rekaman.reshape(-1)
        amplitudo = np.max(np.abs(rekaman))
        if amplitudo > 0.2:
            hasil, sinyal, fft, fitur_avg = prediksi_nada(rekaman)

            if hasil != "Tidak valid":
                window['-OUTPUT-'].update(hasil)

                if canvas1: canvas1.get_tk_widget().destroy()
                if canvas2: canvas2.get_tk_widget().destroy()
                if canvas3: canvas3.get_tk_widget().destroy()

                canvas1 = draw_plot(window['-PLOT1-'], sinyal, "Rekaman Suara", plot_type='line')
                canvas2 = draw_plot(window['-PLOT2-'], fitur_avg, "Ekstraksi Ciri", plot_type='bar')
                canvas3 = draw_plot(window['-PLOT3-'], fft, "FFT", plot_type='line')

except KeyboardInterrupt:
    pass

window.close()
