from setup import *
from helpers.utils import load_params
from functools import lru_cache, cache
import os
from dotenv import load_dotenv

# env config
load_dotenv()
@cache
def index():
    # timdettmers/openassistant-guanaco
    # get params
    params = load_params('../config/config.yml')
    print(f'Params: {params}')
    # quantization config 4 bit
    quant_config = setup_4_bit_quant_config(params)
    # pretrained model setup
    # model, tokenizer = setup_pretrained_model(params['base_model'],
    #                                   params['cache_dir'],
    #                                   bit4_config=quant_config)
    # PEFT config
    peft_config = setup_peft_config(params)
    # trainer arg
    train_args = setup_training_params(params)
    # dataset
    links = os.getenv('INSTRUCTION_DATASET')
    dataset = training_dataset(dataset_url=links, split='train')
    # Get trainer
    # trainer = setup_trainer(model=model,
    #                         tokenizer=tokenizer,
    #                         peft_config=peft_config,
    #                         max_len=2048,
    #                         train_args=train_args)
    print(f"Quant config: {quant_config.to_dict()} PEFT config: {peft_config.to_dict()}"
          f"Train Arg: {train_args.to_dict()} Dataset: {dataset}")
    # trainer.train()
# execute train
index()