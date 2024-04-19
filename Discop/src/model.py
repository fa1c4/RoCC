from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import BertTokenizer, AutoModelForCausalLM
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
from transformers import ImageGPTFeatureExtractor, ImageGPTForCausalImageModeling

from transformers import PreTrainedTokenizer, PreTrainedModel
from config import Settings


def get_model(settings: Settings) -> PreTrainedModel:
    if settings.task == 'text':
        # if settings.model_name in ['gpt2', 'distilgpt2']:
        if 'gpt' in settings.model_name:
            model = AutoModelForCausalLM.from_pretrained(settings.model_name).to(settings.device)
        elif settings.model_name == 'transfo-xl-wt103':
            model = TransfoXLLMHeadModel.from_pretrained(settings.model_name).to(settings.device)
        else:
            raise NotImplementedError
    elif settings.task == 'image':
        if settings.model_name == 'openai/imagegpt-small':
            model = ImageGPTForCausalImageModeling.from_pretrained(settings.model_name).to(settings.device)
        else:
            raise NotImplementedError
    model.eval()
    return model


def get_tokenizer(settings: Settings) -> PreTrainedTokenizer:
    assert settings.task == 'text'
    # if settings.model_name in ['gpt2', 'distilgpt2']:
    if 'gpt' in settings.model_name:
        tokenizer = AutoTokenizer.from_pretrained(settings.model_name)  # local_files_only=True
        # tokenizer = BertTokenizer.from_pretrained(settings.model_name)  # local_files_only=True
    elif settings.model_name == 'transfo-xl-wt103':
        tokenizer = TransfoXLTokenizer.from_pretrained(settings.model_name)  # local_files_only=True
    else:
        raise NotImplementedError
    return tokenizer


def get_feature_extractor(settings: Settings) -> ImageGPTFeatureExtractor:
    assert settings.task == 'image'
    if settings.model_name == 'openai/imagegpt-small':
        feature_extractor = ImageGPTFeatureExtractor.from_pretrained(settings.model_name)  # local_files_only=True
    else:
        raise NotImplementedError
    return feature_extractor
