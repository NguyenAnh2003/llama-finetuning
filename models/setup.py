import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, \
    BitsAndBytesConfig as bnb_config
import os
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
    :param path: the dir for caching
    """
    if not os.path.exists(path):
        os.mkdir(path)

# set up QLoRA config
def setup_4_bit_quant_config(params):
    params['bnb_4bit_compute_dtype'] = torch.bfloat16
    bit4_config = bnb_config(
        #
        load_in_4bit=params['load_in_4bit'],
        bnb_4bit_quant_type=params['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=params['bnb_4bit_compute_dtype'],
        bnb_4bit_use_double_quant=params['bnb_4bit_use_double_quant']
    )
    return bit4_config

def setup_peft_config(params):
    peft_config = LoraConfig(
        lora_alpha=params['alpha'],
        lora_dropout=params['lora_dropout'],
        r=params['peft_r'],
        bias=params['peft_bias'],
        task_type=params['task_type'],
    )
    return peft_config
    
def setup_pretrained_model(model_name, cache_dir, bit4_config):
    """
    :param model_name:
    :param cache_dir: Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir)  # tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_token({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir, load_in_4bit=True,
                                                 quantization_config=bit4_config, trust_remote_code=True)
    return model, tokenizer

def setup_training_params(params):
    train_params = TrainingArguments(
        output_dir=params["output_dir"],
        num_train_epochs=params["epochs"],
        per_device_train_batch_size=params["per_device_train_batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        optim=params["optim"],
        save_steps=params["save_steps"],
        logging_steps=params["logging_steps"],
        learning_rate=params["learning_rate"],
        # fp16=True,
        max_grad_norm=params["max_grad_norm"],
        max_steps=params["max_steps"],
        warmup_ratio=params["warmup_ratio"],
        group_by_length=params["group_by_length"],
        lr_scheduler_type=params["lr_scheduler_type"],
        report_to="wandb" if params["use_wandb"] else None,
        run_name=params["wandb_run_name"] if params["use_wandb"] else None,
    )
    return train_params

def setup_trainer(model, tokenizer, dataset, peft_config, max_len, train_args):
    """
    :param model: LLMs
    :param tokenizer: LLMs tokenizer
    :param dataset:
    :param peft_config:
    :param max_len:
    :param train_args:
    :return: SFT trainer
    """
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_len,
        train_args=train_args
    )
    return trainer

def training_dataset(split, dataset_url: str = None):
    datasets = load_dataset("json",data_files=dataset_url, split=split)
    return datasets