import os
import time

import torch
import soundfile as sf
from PIL import Image

from config import Settings, text_default_settings, image_default_settings, audio_default_settings
from stega_cy import encode_text, decode_text, encode_image, decode_image
# from stega_tts import encode_speech, decode_speech, get_tts_model
from model import get_model, get_feature_extractor, get_tokenizer
import json

message_file_path = os.path.join('./temp/messages', 'message.txt')
with open(message_file_path, 'r', encoding='utf-8') as f:
    message = f.read()
# message *= 10
# print(len(message))


def test_stega_text(settings: Settings = text_default_settings):
    model = get_model(settings)
    tokenizer = get_tokenizer(settings)

    context = 'We were both young when I first saw you, I close my eyes and the flashback starts.'
    # context = '随便一句话'
    single_example_output = encode_text(model, tokenizer, message, context)
    # print(type(single_example_output), single_example_output)
    # print(type(single_example_output.stego_object), single_example_output.stego_object)
    # print(single_example_output.__dict__)
    print('length of stegotext: ', len(single_example_output.stego_object))
    print('time cost: ', single_example_output.time_cost)
    print('encoded bits: ', single_example_output.n_bits)
    print()
    message_encoded = message[:single_example_output.n_bits]
    message_decoded = decode_text(model, tokenizer, single_example_output.generated_ids, context)
    # print("message_encoded: ", message_encoded)
    # print("message_decoded: ", message_decoded)
    # print(message_encoded == message_decoded)
    print('message_decoded length is:{}'.format(len(message_decoded)))
    return [single_example_output.time_cost, len(message_decoded)]


def test_staga_image(settings: Settings = image_default_settings):
    context_ratio = 0.5
    original_img = Image.open('small.png')

    # context_ratio = 0.0
    # original_img = None

    model = get_model(settings)
    feature_extractor = get_feature_extractor(settings)

    single_example_output = encode_image(model,
                                         feature_extractor,
                                         message,
                                         context_ratio=context_ratio,
                                         original_img=original_img)
    print(single_example_output)
    message_encoded = message[:single_example_output.n_bits]

    stego_img = single_example_output.stego_object
    stego_img.save('stego_0801.png')

    # message_decoded = decode_image(model, feature_extractor, stego_img, context_ratio=context_ratio)
    # print(message_encoded == message_decoded)


def test_stega_tts(settings: Settings = audio_default_settings):
    # text = 'We were both young.'
    # text = "Taylor Swift is an American singer-songwriter, record producer, music video director, philanthropist, " \
    #        "and actress. "
    # text = "No."
    # text = 'taylor alison swift born december thirteen, nineteen eighty nine is an american singer songwriter. her discography spans multiple genres, and her narrative songwriting—often inspired by her personal life—has received critical praise and widespread media coverage. born in west reading, pennsylvania, swift moved to nashville, tennessee, at the age of fourteen to pursue a career in country music. she signed a songwriting contract with sony/atv music publishing in two thousand four and a recording deal with big machine records in two thousand five, and released her eponymous debut studio album in two thousand six.'
    text = 'Taylor Alison Swift (born December 13, 1989) is an American singer-songwriter. Her discography spans multiple genres, and her narrative songwriting—often inspired by her personal life—has received critical praise and widespread media coverage.'
    vocoder, tacotron, cmudict = get_tts_model(settings)
    single_example_output, sr = encode_speech(vocoder, tacotron, cmudict, message, text, settings)

    print(single_example_output)

    wav = single_example_output.stego_object
    sf.write('wav_0802.flac', wav, sr)
    # sf.write('wav_0802.wav', wav, sr)
    message_encoded = message[:single_example_output.n_bits]

    message_decoded = decode_speech(vocoder, tacotron, cmudict, wav, text, settings)
    # print(message_decoded)
    print(message_encoded == message_decoded)


if __name__ == '__main__':
    # test_stega_text()
    # test_staga_image()
    print(torch.__version__)
    print(torch.cuda.is_available())
    # pause()
    settings: Settings = text_default_settings
    if torch.cuda.is_available():
        settings.device = torch.device('cuda:0')
    print(settings.device)
    # settings.device = torch.device('cpu')
    # test_stega_tts(settings)
    times = {}
    for leng in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        settings.length = leng
        # time_start = time.time()
        time_cost, mes_length = test_stega_text(settings)
        # time_end = time.time()
        times[leng] = [time_cost, mes_length] 
        # encode time, we regard decode time as the same with encode
        print('time for text stego is: {}s'.format(time_cost))

    with open('./jsons/length_stego_times.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(times, indent=4))