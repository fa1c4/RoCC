'''
this lab intends to preprocess stegotext for each message and each context
the results of process are saved to json files at ./jsons/cont_mes.json
'''

import json
import os
import time

import torch
import soundfile as sf

from config import Settings, text_default_settings, image_default_settings, audio_default_settings
from stega_cy import encode_text, decode_text
# from stega_tts import encode_speech, decode_speech, get_tts_model
from model import get_model, get_feature_extractor, get_tokenizer


def single_stega_text(content, mes, settings: Settings = text_default_settings):
    model = get_model(settings)
    tokenizer = get_tokenizer(settings)

    context = content
    message = mes
    single_example_output = encode_text(model, tokenizer, message, context)
    # print(type(single_example_output.stego_object), single_example_output.stego_object)
    print('length of stegotext: ', len(single_example_output.stego_object))
    print('time cost: ', single_example_output.time_cost)
    print('encoded bits: ', single_example_output.n_bits)
    print()
    message_encoded = message[:single_example_output.n_bits]
    message_decoded = decode_text(model, tokenizer, single_example_output.generated_ids, context)
    # print("message_encoded: ", message_encoded)
    # print("message_decoded: ", message_decoded)
    print(message_encoded == message_decoded)
    # print('message_decoded length is:{}'.format(len(message_decoded)))
    return single_example_output.stego_object


# ----- read contexts -----
with open('./temp/contexts/context.txt', 'r', encoding='utf-8') as f:
    contexts = f.read()

contexts = contexts.split('\n')[:-1]
# print(contexts)

# ----- read messages -----
messages = []
for i in range(10):
    message_file_path = os.path.join('./temp/messages', 'message-{}.txt'.format(i))
    with open(message_file_path, 'r', encoding='utf-8') as f:
        message = f.read()

    messages.append(message)


if __name__ == '__main__':
    # use dict data structure to save each meassge encoded with each context
    # key: context, value: [message, stegotext] 
    data_preprocess = {}

    settings: Settings = text_default_settings
    if torch.cuda.is_available():
        settings.device = torch.device('cuda:0')
    print(settings.device)

    for message in messages:
        for cont in contexts:
            stegotext = single_stega_text(cont, message, settings)
            data_preprocess[cont] = [message, stegotext]
    
    with open('./jsons/cont_mes.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(data_preprocess))

    print('----- saving json file completed -----')