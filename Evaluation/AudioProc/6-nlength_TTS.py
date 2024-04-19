'''
the experiment intents to generate audio files from n length of stegotexts
and record the time comsumed for each length of stegotexts
the length n takes 4, 8, 12, 16, 20, 24, 28, 32, 36 tokens of stegotexts 
'''

import time
import json
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, set_seed
from datasets import load_dataset
import torch
import soundfile as sf
import nltk


# load models
print('---- starting ----')
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
print('---- models loaded ----')

# read stegotexts
with open('./jsons/cont_mes_1000bits.json', 'r', encoding='utf-8') as f:
    json_data = json.loads(f.read())

# generate auidio wav files
def single_TTS(sentence, idx, length):
    inputs = processor(text=sentence, return_tensors="pt")
    set_seed(555)  # make deterministic

    # generate speech
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

    with torch.no_grad():
        speech = vocoder(spectrogram)

    time_start = time.time()
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    time_end = time.time()
    time_comsumed = time_end - time_start
    print('sentence {} -> generating audio time: {}s'.format(sentence, time_comsumed))
    sf.write("./wavs/{}_stegotext_{}_length.wav".format(idx, length), speech.numpy(), samplerate=16000)
    return time_comsumed


if __name__ == '__main__':
    debug_flag = False

    TTS_times = {}
    idx = 0
    for context, value in json_data.items():
        if debug_flag:
            if idx > 0: break

        message = value[0][:1000 + 7]
        stegotext = value[1] 
        tokened_stegotext = nltk.word_tokenize(stegotext)
        TTS_times[idx] = [context, message]
        for length in range(4, 44, 4):
            sentence = ' '.join(tokened_stegotext[:length])
            time_com = single_TTS(sentence, idx, length)
            TTS_times[idx].append([sentence, time_com])
        
        idx += 1        


    with open('./jsons/nlength_TTS_times.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(TTS_times, indent=4))
    
    print('----- TTS times saving completed -----')
