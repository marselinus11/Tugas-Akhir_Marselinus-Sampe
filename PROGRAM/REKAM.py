import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import time

fs = 4800
seconds = 2

print("Sedang merekam...")

recording = sd.rec(int(seconds * fs), samplerate = fs, channels = 1)
for i in range(seconds, 0, -1):
    print (f"{i}...")
    time.sleep(1)
sd.wait()
print("Rekaman selesai!")

wavfile.write('coba.wav', fs, recording)

fs, data = wavfile.read('coba.wav')

time_axis = np.linspace(0, len(data)/fs, num=len(data))

plt.figure(figsize=(10, 4))
plt.plot(time_axis, data, color='blue')
plt.title('Waveform of Recorded Audio')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()