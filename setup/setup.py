import torch
from transformers import AutoTokenizer, \
    AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer
import os
from peft import prepare_model_for_kbit_training, \
    LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
from functools import *

def setup_cache_dir(path):
    """
    :param path: the dir for caching
    """
    if not os.path.exists(path):
        os.mkdir(path)

# set up QLoRA config
def setup_4_bit_quant_config(params):
    """
    QLoRA configuration with BitsAndBytes lib
    :param params: params
    :return: QLoRA config
    """
    params['bnb_4bit_compute_dtype'] = torch.float16
    config = BitsAndBytesConfig(
        load_in_4bit=params['load_in_4bit'],
        bnb_4bit_quant_type=params['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=params['bnb_4bit_compute_dtype'],
        bnb_4bit_use_double_quant=params['bnb_4bit_use_double_quant']
    )
    return config # QLoRA

def setup_peft_config(params):
    """ PEFT config setup using trainable params with PEFT
    :param params: params setup
    :return: PEFT config
    """
    peft_config = LoraConfig(
        lora_alpha=params['alpha'],
        lora_dropout=params['lora_dropout'],
        r=params['peft_r'],
        bias=params['peft_bias'],
        task_type=params['task_type'],
        # set up inference mode
        inference_mode=False
    )
    return peft_config

# PEFT model
def setup_peft_model(model, peft_config):
    """
    :param model: taking pre-trained model
    :param peft_config: defined PEFT config get trainable params
    :return: PEFT model
    """
    model = get_peft_model(model, peft_config); # getting peft model
    model.print_trainable_parameters() # trainable params
    return model

def setup_pretrained_model(model_name, bnb_config):
    """
    :param model_name:
    :param cache_dir: Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True,
                                              torch_dtype=torch.float16,)  # tokenizer
    # if tokenizer.pad_token is None:
        # tokenizer.add_special_token({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token # replace pad with eos token
    tokenizer.add_eos_token = True
    # config use_cache: False -> don't use old params
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.float16,
                                                 load_in_4bit=True,
                                                 load_in_8bit=False,
                                                 quantization_config=bnb_config,
                                                 trust_remote_code=True)
    """ getting model for kbit quantization
    Casts all the non kbit modules to full precision(fp32) for stability
    Adds a forward hook to the input embedding layer to calculate the
    gradients of the input hidden states
    Enables gradient checkpointing for more memory-efficient training
    """
    model.config.use_cache = False # avoid caching for not remember old weights
    model.gradient_checkpointing_enable() # save model grad "check point"
    model = prepare_model_for_kbit_training(model) #
    return model, tokenizer


def setup_training_params(params):
    """
    :param params: defined params
    :return: Training argurments transformers
    """
    train_params = TrainingArguments(
        output_dir=params["output_dir"],
        num_train_epochs=params["epochs"],
        per_device_train_batch_size=params["per_device_train_batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        optim=params["optim"],
        save_steps=params["save_steps"],
        logging_steps=params["logging_steps"],
        learning_rate=float(params["learning_rate"]),
        fp16=params['fp16'],
        bf16=params['bf16'],
        max_grad_norm=params["max_grad_norm"],
        max_steps=params["max_steps"],
        warmup_ratio=params["warmup_ratio"],
        group_by_length=params["group_by_length"],
        lr_scheduler_type=params["lr_scheduler_type"],
        # report_to="wandb" if params["use_wandb"] else None,
        # run_name=params["wandb_run_name"] if params["use_wandb"] else None,
    )
    return train_params

def setup_trainer(model, tokenizer, train_dataset, eval_dataset, peft_config, max_len, train_args):
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_seq_length=max_len,
        args=train_args,
        dataset_batch_size=32
    )
    return trainer


# Transformers Trainer
def setup_transformers_trainer(model, train_data, args, collator):
    """
    :param model: PEFT model
    :param train_data: train set
    :param eval_data: dev set
    :param args: training args
    :param collator: data colllator
    :return: transformer Trainer class
    """
    trainer = Trainer(
    model=model,
    train_dataset=train_data,
    args=args,
    data_collator=collator
    )
    return trainer

def training_dataset(dataset_url: str = None):
    """
    :param dataset_url: json file
    :return: set of data
    """
    datasets = load_dataset("json",data_files=dataset_url)
    return datasets

# generate and tokenize prompt
def gen_tokenize(point, tokenizer: AutoTokenizer):
    """
    Using Llama tokenizer to tokenizer the token to numerical vector for fine-tuning
    :param point: data point pass to dataset libs
    :param tokenizer: tokenizer
    :return: set train
    """

    # setup prompt and design
    def generate_prompt(point):
        """ Data format having {Instruction, Input, Output}
        :param point(data point) passing through data collator
        dataset attr (instruction, input, output)
        """
        return f"""
        Bạn là trợ lý AI hữu ích. Hãy trả lời câu hỏi của người dùng một cách có logic nhất
        dưới đây ### Instruction: {point['instruction']} là sự hướng dẫn hoặc cũng có thể là input của người dùng
        hãy dựa vào đây để trả lời câu hỏi
        ### Input: {point['input']} cũng có thể là input của người dùng (ở đây có thể có hoặc không)
        ### Output: {point['output']} sẽ là kết quả của câu hỏi
        """

    prompt = generate_prompt(point) # generate prompt based on data point
    # tokenize using defined tokenizer
    tokenized_prompt = tokenizer(prompt, padding=True, truncation=True)
    # returning result
    return tokenized_prompt