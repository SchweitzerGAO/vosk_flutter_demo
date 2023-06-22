from scipy.io import wavfile
from utils import *
import torch


def read_data(file_name):
    sr, sig = wavfile.read(filename=file_name)
    return sr, sig


def pre_emphasize(sig, alpha=0.97):
    sig = np.append(sig[0], sig[1:] - alpha * sig[:-1])
    return sig


def framing(sr, sig, len_frame=0.025, shift_frame=0.01):
    sig_n = len(sig)
    frame_len_n, frame_shift_n = int(round(sr * len_frame)), int(round(sr * shift_frame))
    num_frame = int(np.ceil(float(sig_n - frame_len_n) / frame_shift_n) + 1)
    pad_num = frame_shift_n * (num_frame - 1) + frame_len_n - sig_n
    pad_zero = np.zeros(int(pad_num))
    pad_sig = np.append(sig, pad_zero)

    frame_inner_index = np.arange(0, frame_len_n)

    frame_index = np.arange(0, num_frame) * frame_shift_n

    frame_inner_index_extend = np.tile(frame_inner_index, (num_frame, 1))

    frame_index_extend = np.expand_dims(frame_index, 1)

    each_frame_index = frame_inner_index_extend + frame_index_extend
    each_frame_index = each_frame_index.astype(np.int32, copy=False)

    frame_sig = pad_sig[each_frame_index]
    return frame_sig


def windowing(frame_sig, sr, len_frame=0.025):
    window = np.hamming(int(round(len_frame * sr)))
    return frame_sig * window


def stft(frame_sig, nfft=512):
    frame_spec = np.fft.rfft(frame_sig, nfft)
    frame_mag = np.abs(frame_spec)
    frame_pow = (frame_mag ** 2) * 1.0 / nfft
    return frame_pow


def mel_filter(frame_pow, sr, n_filter, nfft=512):
    mel_min = 0
    mel_max = 2595 * np.log10(1 + sr / 2.0 / 700)
    mel_points = np.linspace(mel_min, mel_max, n_filter + 2)
    hz_points = 700 * (10 ** (mel_points / 2595.0) - 1)
    filter_edge = np.floor(hz_points * (nfft + 1) / sr)

    # 求mel滤波器系数
    fbank = np.zeros((n_filter, int(nfft / 2 + 1)))
    for m in range(1, 1 + n_filter):
        f_left = int(filter_edge[m - 1])  # 左边界点
        f_center = int(filter_edge[m])  # 中心点
        f_right = int(filter_edge[m + 1])  # 右边界点

        for k in range(f_left, f_center):
            fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            fbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    filter_banks = np.dot(frame_pow, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    return filter_banks


def de_mean(filter_banks):
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    return filter_banks


def fbank(file_name):
    sr, sig = read_data(file_name)
    sig = pre_emphasize(sig)
    framed_sig = framing(sr, sig)
    framed_sig = windowing(framed_sig, sr)
    frame_pow = stft(framed_sig)
    filtered_banks = mel_filter(frame_pow, sr, n_filter=64)
    filtered_banks = de_mean(filtered_banks)
    return filtered_banks


if __name__ == '__main__':
    filtered = fbank('../data/1-8.wav')
    pass
