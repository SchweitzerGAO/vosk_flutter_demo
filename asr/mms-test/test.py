from scipy.io import wavfile
import numpy as np
import torch

from transformers import Wav2Vec2ForCTC, AutoProcessor

model_id = "facebook/mms-1b-fl102"

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)


def audio_to_array(file_name):
    bi_channel_array = wavfile.read(file_name)[1]
    mono_channel_array = []
    for segment in bi_channel_array:
        mono_channel_array.append(max(segment[0] / 32768, segment[1] / 32768))
    mono_channel_array = np.array(mono_channel_array, dtype=np.float32)
    return mono_channel_array


def test(lang, mono=True):
    processor.tokenizer.set_target_lang(lang)
    model.load_adapter(lang)
    file_name = f'./{lang}.wav'
    if not mono:
        array_from_scipy = audio_to_array(file_name)
    else:
        array_from_scipy = np.array(wavfile.read(file_name)[1], dtype=np.float32) / 32768
    inputs = processor(array_from_scipy, sampling_rate=16000, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)
    print(transcription)


if __name__ == '__main__':
    lang = 'cmn-script_simplified'
    test(lang)
