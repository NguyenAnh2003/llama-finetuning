from setup import *
from helpers.utils import load_params

def index():
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
    # Get trainer
    # trainer = setup_trainer(model=model,
    #                         tokenizer=tokenizer,
    #                         peft_config=peft_config,
    #                         max_len=2048,
    #                         train_args=train_args)
    print(f"Quant config: {quant_config} PEFT config: {peft_config}"
          f"Train Arg: {train_args}")

# execute train
index()