import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig as bnb_config
import os
from peft import prepare_model_for_kbit_training, \
    LoraConfig, get_peft_config, get_peft_model_state_dict, prepare_model_for_int8_training, set_peft_model_state_dict
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer


def setup_cache_dir(path):
    """
    :param path: the dir if not exist
    """
    if not os.path.exists(path):
        os.mkdir(path)


def setup_pretrained_model(model_name, cache_dir):
    """
    :param model_name:
    :param cache_dir: Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir) # tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_token({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir, load_in_4bit=True,
                                                 quantization_config=lr_config, trust_remote_code=True)
    return model, tokenizer

