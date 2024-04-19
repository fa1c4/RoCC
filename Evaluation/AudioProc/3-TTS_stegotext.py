'''
there are some bugs here
the index between wav and texts dismatches
need to fix it now ---- 2023.6.26
maybe this not the problem, found the wavs in order already
but turns out some wavs or texts are missing, like index == 100 sentence,
the recognition line is 'she said the oak should return to the european single market without permanent membership'
while the origin line is 'Ms Parsons, from Aberdeen, added: "You don't have a passport to set up in the UK'

so, I decide to regenerate the wavs through, this time use index as dict key
then it expects to be easy to index from dict

the bug solved ---- 2023.6.26
'''

import time
import json
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, set_seed
from datasets import load_dataset
import torch
import soundfile as sf


# load models
print('---- starting ----')
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
print('---- models loaded ----')

# read stegotexts
with open('./jsons/all_texts.json', 'r', encoding='utf-8') as f:
    json_data = json.loads(f.read())

# generate auidio wav files
def single_TTS(sentence, idx):
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
    sf.write("./wavs/tts_sentence_{}.wav".format(idx), speech.numpy(), samplerate=16000)
    return time_comsumed


if __name__ == '__main__':
    debug_flag = False

    TTS_times = {}
    idx = 0
    for sentence in json_data:
        if debug_flag:
            # print(len(json_data))
            if idx > 5: break
        
        time_com = single_TTS(sentence, idx)
        TTS_times[idx] = [sentence, time_com]
        idx += 1

    with open('./jsons/TTS_times.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(TTS_times, indent=4))
    
    print('----- TTS times saving completed -----')
