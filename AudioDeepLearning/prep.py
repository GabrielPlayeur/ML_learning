import librosa
import matplotlib.pyplot as plt
import numpy as np

file = "debussy.wav"

signal, sr = librosa.load(file, sr=22050)
librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

plt.plot(frequency[:int(len(magnitude)/2)], magnitude[:int(len(magnitude)/2)])
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

frame_size = 2048
hop_size = 512

stft = librosa.core.stft(y=signal, n_fft=frame_size, hop_length=hop_size)
spectrogram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrogram)

plt.figure(figsize=(25,10))
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_size, x_axis="time", y_axis="log")
plt.xlabel("Time")
plt.colorbar()
plt.show()