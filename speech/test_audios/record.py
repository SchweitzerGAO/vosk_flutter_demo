import audioop
import time
import wave

import pyaudio

from datasets import Audio, Dataset

import numpy as np
from scipy.io import wavfile


def record(file_name):
    lower = 0
    frames = []
    form = pyaudio.paInt16
    channels = 1
    sr = 16000
    block_size = 2048
    threshold = 300
    wait_time = 5
    recorder = pyaudio.PyAudio()
    stream = recorder.open(format=form,
                           channels=channels,
                           rate=sr,
                           input=True)
    now = time.time()
    while True:
        data = stream.read(block_size)
        rms = audioop.rms(data, 2)
        if rms >= threshold:
            break
        if time.time() - now > wait_time:
            return False

    print('recording')
    while True:
        frames.append(data)
        data = stream.read(block_size)
        rms = audioop.rms(data, 2)
        if rms < threshold:
            lower += 1
        else:
            lower = 0
        if lower >= 15:
            break
    with wave.open(file_name, 'wb') as f:
        print('saving')
        f.setnchannels(channels)
        f.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        f.setframerate(sr)
        f.writeframes(b''.join(frames))

    stream.stop_stream()
    stream.close()
    recorder.terminate()
    return True


if __name__ == '__main__':
    lang = 'cmn-script_simplified'
    # record(f'{lang}.wav')
    audio_dataset = Dataset.from_dict({'audio': [f'./{lang}.wav']}).cast_column('audio', Audio(sampling_rate=16000))
    a = audio_dataset[0]['audio']['array']
    b = np.array(wavfile.read(f'{lang}.wav')[1], dtype=np.float32) / 32768
    pass
