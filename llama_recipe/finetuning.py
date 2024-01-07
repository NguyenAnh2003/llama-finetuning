from setup.setup import *
from helpers.utils import load_params
import os
from dotenv import load_dotenv

# env config
load_dotenv()
def index():
    """
    fine tuning model with QLoRA config
    and PEFT config
    """
    # get params
    params = load_params('../config/config.yml')
    # quantization config 4 bit
    quant_config = setup_4_bit_quant_config(params)

    # setup cache dir
    cache_dir = params['cache_dir']
    setup_cache_dir(cache_dir)
    # pretrained model setup
    model, tokenizer = setup_pretrained_model(params['base_model'],
                                      cache_dir=cache_dir,
                                      bit4_config=quant_config)

    
    # PEFT config
    peft_config = setup_peft_config(params)

    model_peft = setup_peft_model(model=model, peft_config=peft_config)
    
    # trainer arg
    train_args = setup_training_params(params)

    # dataset
    links = os.getenv('INSTRUCTION_DATASET')
    dataset = training_dataset(dataset_url=links)

    # Get trainer
    trainer = setup_trainer(model=model_peft,
                            tokenizer=tokenizer,
                            peft_config=peft_config,
                            max_len=2048,
                            train_args=train_args)

    print(f"Quant config: {quant_config.to_dict()} PEFT config: {peft_config.to_dict()}"
          f"Train Arg: {train_args.to_dict()} Dataset: {dataset}")

    # training
    trainer.train() # train with SFTTrainer

    # save model
    trainer.save_model(params['output_dir'])

index()
