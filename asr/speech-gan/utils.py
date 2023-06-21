import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import os


def plot_time(sig, sr):
    time = np.arange(0, len(sig)) * (1.0 / sr)
    plt.figure(figsize=(20, 5))
    plt.plot(time, sig)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()


def plot_spectrogram(spec, ylabel):
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('number of frames')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def resample(path, new_sample_rate=16000):
    signal, sr = librosa.load(path, sr=None)
    new_signal = librosa.resample(signal, orig_sr=sr, target_sr=new_sample_rate)
    sf.write(path, new_signal, new_sample_rate)


if __name__ == '__main__':
    files = os.listdir('data')
    for file in files:
        resample(os.path.join('data', file))
        print(f'{file} finished resampling')
