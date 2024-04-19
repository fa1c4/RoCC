'''
Ths experiment 1, use the model 'microsoft/speecht5_asr' 
to do speech recognition
two datasets including: 
(1) hf-internal-testing/librispeech_asr_dummy
(2) hf-internal-testing/librispeech_asr_demo
'''

from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
import time
import json


# dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.sort("id")
generator = pipeline(task="automatic-speech-recognition", model="microsoft/speecht5_asr")

audio_recogs = {}

debug_flag = False
test_cnt = 0

for example in dataset:
    # debug if debug_flag == True 
    # with test_cnt < number
    test_cnt += 1
    if debug_flag:
        if test_cnt > 10:
            break

    test_audio_file = example["audio"]["array"]

    start_time = time.time()
    test_wav_gen = generator(test_audio_file)
    end_time = time.time()

    time_comsumed = end_time - start_time
    print('time comsumed: {} s'.format(time_comsumed))
    print(test_wav_gen)

    ori_text = example['text'].lower()
    print('origin text is: {}'.format(ori_text))
    audio_recogs[test_cnt] = [test_wav_gen['text'], ori_text, time_comsumed]


# with open('./librispeech_asr_demo_results.json', 'w', encoding='utf-8') as f:
with open('./jsons/librispeech_asr_dummy_results.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(audio_recogs, indent=4))

print('----- asr completed -----')