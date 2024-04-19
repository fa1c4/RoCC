import os
import time
import random
import torch
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from nltk import sent_tokenize
from datasets import load_dataset

from config import Settings, text_default_settings, image_default_settings, audio_default_settings
from model import get_model, get_feature_extractor, get_tokenizer
from stega_cy import SingleExampleOutput, encode_text, encode_image
# from random_sample_cy import SingleExampleOutput, encode_text, encode_image
from stega_tts import get_tts_model, encode_speech
from utils import check_dir

top_p_lst = [0.80, 0.92, 0.95, 0.98, 1.0]
text_context_dataset = 'imdb'
image_context_dataset = 'huggan/CelebA-faces'
tts_dataset = 'imdb'

message_file_path = os.path.join('temp', 'message.txt')
with open(message_file_path, 'r', encoding='utf-8') as f:
    message = f.read()

summary_columns = [
    'temperature', 'top-p', 'total_n_bits', 'total_n_tokens', 'total_entropy', 'total_time_cost', 'ave_time_cost', 'ave_kld',
    'max_kld', 'ave_embedding_rate', 'ave_entropy', 'utilization_rate', 'ave_perplexity', 'ave_minimum_entropy'
]


class Summary:
    def __init__(self, settings: Settings) -> None:
        self.n_examples = 0
        self.total_perplexity = 0
        self.total_ave_kld = 0
        self.total_minimum_entropy = 0
        self.output = {
            'algorithm': settings.algo,
            'temperature': settings.temp,
            'top-p': settings.top_p,
            'total_n_bits': 0,
            'total_n_tokens': 0,
            'total_entropy': 0,
            'total_time_cost': 0,
            'max_kld': 0
        }

    def __str__(self) -> str:
        self.process()
        selected_attr = list(self.output.keys())
        return '\n'.join('{} = {}'.format(x, self.output[x]) for x in selected_attr)

    def add_example(self, example: SingleExampleOutput) -> None:
        self.output['total_n_bits'] += example.n_bits
        self.output['total_n_tokens'] += example.n_tokens
        self.output['total_entropy'] += example.total_entropy
        self.output['total_time_cost'] += example.time_cost
        self.total_perplexity += example.perplexity
        self.total_ave_kld += example.ave_kld
        if example.max_kld > self.output['max_kld']:
            self.output['max_kld'] = example.max_kld
        self.n_examples += 1
        self.total_minimum_entropy += example.total_minimum_entropy

    def process(self) -> None:
        self.output['ave_embedding_rate'] = self.output['total_n_bits'] / self.output['total_n_tokens']
        self.output['utilization_rate'] = self.output['total_n_bits'] / self.output['total_entropy'] if self.output[
            'total_entropy'] != 0 else 0
        self.output['ave_entropy'] = self.output['total_entropy'] / self.output['total_n_tokens']
        self.output['ave_perplexity'] = self.total_perplexity / self.n_examples
        self.output['ave_kld'] = self.total_ave_kld / self.n_examples
        self.output['ave_time_cost'] = self.output['total_time_cost'] / self.output['total_n_bits'] if self.output[
            'total_n_bits'] != 0 else 0
        self.output['ave_minimum_entropy'] = self.total_minimum_entropy / self.output['total_n_tokens']

    def gather(self) -> pd.DataFrame:
        self.process()
        ret_lst = []
        for column in summary_columns:
            ret_lst.append(self.output[column])
        df = pd.DataFrame(ret_lst, index=summary_columns).T
        return df


def get_text_statistics(settings: Settings = text_default_settings, n_examples=100, save_data: bool = False) -> None:
    model = get_model(settings)
    tokenizer = get_tokenizer(settings)

    dataset = load_dataset(text_context_dataset, split='train')[:n_examples]['text']

    if save_data:
        save_data_main_dir = os.path.join(
            'data', settings.task,
            '{}_{}_{}'.format(settings.model_name.split('/')[-1], settings.device, time.strftime("%m%d_%H%M", time.localtime())))
        check_dir(save_data_main_dir)

    df = pd.DataFrame(columns=summary_columns)
    for top_p in top_p_lst:
        settings.top_p = top_p
        summary = Summary(settings)

        if save_data:
            save_context_path = os.path.join(save_data_main_dir, '{:.2f}_context.txt'.format(top_p))
            save_stego_path = os.path.join(save_data_main_dir, '{:.2f}_stego.txt'.format(top_p))
            f_context = open(save_context_path, 'w')
            f_stego = open(save_stego_path, 'w')
            context_lst = []
            stego_lst = []

        for i in tqdm(range(n_examples), ncols=70, desc='p={:.2f}'.format(top_p)):
            random.seed(os.urandom(1))
            message_start_index = random.randint(0, 10000)

            context = dataset[i]
            context = context.replace('<br /><br />', ' ').replace('<br />', ' ')  # remove all '<br />'
            context = ' '.join(sent_tokenize(context)[:3])  # Selecting leading 3 sentences as `context`
            settings.seed = os.urandom(1)
            example = encode_text(model, tokenizer, message[message_start_index:], context, settings)
            summary.add_example(example)
            if save_data:
                context_lst.append(context)
                stego_lst.append(example.stego_object)
        if save_data:
            f_context.write('\n'.join(context_lst))
            f_stego.write('\n'.join(stego_lst))

        print(summary)
        print()
        df = pd.concat([df, summary.gather()], ignore_index=True)
    save_table_dir = os.path.join('results', settings.task)
    check_dir(save_table_dir)
    save_table_filename = '{}_{}_{}.xlsx'.format(
        settings.model_name.split('/')[-1], settings.device, time.strftime("%m%d_%H%M", time.localtime()))
    save_table_path = os.path.join(save_table_dir, save_table_filename)
    df.to_excel(save_table_path)
    if save_data:
        f_context.close()
        f_stego.close()


def get_image_statistics(settings: Settings = image_default_settings, n_examples=100, save_data: bool = False) -> None:
    context_ratio = 0.5

    model = get_model(settings)
    feature_extractor = get_feature_extractor(settings)

    dataset = load_dataset(image_context_dataset, split='train')[:n_examples]['image']
    width, height = dataset[0].size
    width_after = feature_extractor.size
    height_after = round(width_after / width * height)

    if save_data:
        save_data_main_dir = os.path.join(
            'data', settings.task,
            '{}_{}_{}'.format(settings.model_name.split('/')[-1], settings.device, time.strftime("%m%d_%H%M", time.localtime())))

    df = pd.DataFrame(columns=summary_columns)
    for top_p in top_p_lst:
        settings.top_p = top_p
        summary = Summary(settings)

        if save_data:
            save_data_sub_dir = os.path.join(save_data_main_dir, '{:.2f}'.format(top_p))
            check_dir(save_data_sub_dir)

        for i in tqdm(range(n_examples), ncols=70, desc='p={:.2f}'.format(top_p)):
            random.seed(os.urandom(1))
            message_start_index = random.randint(0, 10000)

            original_img = dataset[i]
            original_img = original_img.resize([width_after, height_after])
            original_img = original_img.crop((0, 4, 32, 36))

            settings.seed = os.urandom(1)
            example = encode_image(model,
                                   feature_extractor,
                                   message[message_start_index:],
                                   settings,
                                   context_ratio=context_ratio,
                                   original_img=original_img)
            summary.add_example(example)
            if save_data:
                save_data_path = os.path.join(save_data_sub_dir, '{}.png'.format(i))
                example.stego_object.save(save_data_path)
        print(summary)
        print()
        df = pd.concat([df, summary.gather()], ignore_index=True)
    save_table_dir = os.path.join('results', settings.task)
    check_dir(save_table_dir)
    save_table_filename = '{}_{}_{}.xlsx'.format(
        settings.model_name.split('/')[-1], settings.device, time.strftime("%m%d_%H%M", time.localtime()))
    save_table_path = os.path.join(save_table_dir, save_table_filename)
    df.to_excel(save_table_path)


def get_audio_statistics(settings: Settings = audio_default_settings, n_examples=30, save_data: bool = False) -> None:
    vocoder, tacotron, cmudict = get_tts_model(settings)

    dataset = load_dataset(tts_dataset, split='train')[:n_examples]['text']

    if save_data:
        save_data_main_dir = os.path.join(
            'data', settings.task,
            '{}_{}_{}'.format(settings.model_name.split('/')[-1], settings.device, time.strftime("%m%d_%H%M", time.localtime())))
        check_dir(save_data_main_dir)

    df = pd.DataFrame(columns=summary_columns)
    for top_p in top_p_lst:
        settings.top_p = top_p

        summary = Summary(settings)

        if save_data:
            save_data_sub_dir = os.path.join(save_data_main_dir, '{:.2f}'.format(top_p))
            check_dir(save_data_sub_dir)
            save_text_path = os.path.join(save_data_main_dir, '{:.2f}.txt'.format(top_p))
            text_lst = []
            f_text = open(save_text_path, 'w')

        for i in tqdm(range(n_examples), ncols=70, desc='p={:.2f}'.format(top_p)):
            random.seed(os.urandom(1))
            message_start_index = random.randint(0, 10000)

            text = dataset[i]
            text = text.replace('<br /><br />', ' ').replace('<br />', ' ')  # remove all '<br />'
            text = ' '.join(sent_tokenize(text)[:1])  # Selecting leading 1 sentences as `context`
            settings.seed = os.urandom(1)
            example, sr = encode_speech(vocoder, tacotron, cmudict, message[message_start_index:], text, settings)
            summary.add_example(example)
            if save_data:
                save_data_path = os.path.join(save_data_sub_dir, '{}.flac'.format(i))
                sf.write(save_data_path, example.stego_object, sr)
                text_lst.append(text)
        if save_data:
            f_text.write('\n'.join(text_lst))

        print(summary)
        print()
        df = pd.concat([df, summary.gather()], ignore_index=True)
    save_table_dir = os.path.join('results', settings.task)
    check_dir(save_table_dir)
    save_table_filename = '{}_{}_{}.xlsx'.format(settings.model_name, settings.device,
                                                 time.strftime("%m%d_%H%M", time.localtime()))
    save_table_path = os.path.join(save_table_dir, save_table_filename)
    df.to_excel(save_table_path)
    if save_data:
        f_text.close()


if __name__ == '__main__':
    text_settings = text_default_settings
    text_settings.model_name = 'distilgpt2'
    text_settings.device = torch.device('cuda:1')
    get_text_statistics(text_settings, n_examples=3)
    # get_text_statistics(text_settings)

    # image_settings = image_default_settings
    # # get_image_statistics(image_settings)
    # image_settings.device = torch.device('cuda:1')
    # get_image_statistics(image_settings, save_data=True)

    audio_settings = audio_default_settings
    audio_settings.device = torch.device('cuda:1')
    get_audio_statistics(audio_settings, save_data=True)
    # audio_settings.device = torch.device('cuda:1')
    # get_audio_statistics(audio_settings)