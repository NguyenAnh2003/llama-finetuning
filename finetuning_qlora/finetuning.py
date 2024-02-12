from setup.setup import *
from helpers.utils import load_params
import os
from transformers import DataCollatorForLanguageModeling
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# .env config
load_dotenv()


def index():
    """
    fine-tuning model with QLoRA config
    and PEFT config
    """
    # get params
    params = load_params('../config/config.yml')
    # quantization config 4 bit
    quant_config = setup_4_bit_quant_config(params)

    # pretrained model setup
    model, tokenizer = setup_pretrained_model(params['base_model'],
                                              bnb_config=quant_config)

    # PEFT config
    peft_config = setup_peft_config(params)

    model_peft = setup_peft_model(model=model, peft_config=peft_config)

    # trainer arg
    train_args = setup_training_params(params)

    # dataset
    links = os.getenv('INSTRUCTION_DATASET')
    dataset = training_dataset(dataset_url=links)

    # logging dataset
    logger.info("-- Preparing Dataset --")

    # custom data with prompt (including tokenize)
    set = dataset['train'].train_test_split(test_size=0.3, seed=42)

    data_train = set['train'].map(
        lambda sample: gen_tokenize(point=sample, tokenizer=tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # logging trainer
    logger.info("-- Preparing Trainer --")

    # Get trainer
    trainer = setup_transformers_trainer(model=model_peft,
                                         train_data=data_train,
                                         args=train_args,
                                         collator=data_collator)

    # logging
    logger.info("-- Training Model --")

    # training
    trainer.train()  # train with SFTTrainer
    # log to WB

    # logging complete
    logger.info("-- Train Complete")

    # save model
    # trainer.save_model(params['output_dir'])

if __name__ == "__main__":
    index()
