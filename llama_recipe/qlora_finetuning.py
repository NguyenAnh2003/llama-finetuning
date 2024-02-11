from setup.setup import *
from helpers.utils import load_params
import os
from transformers import DataCollatorForLanguageModeling
from dotenv import load_dotenv

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


    # custom data with prompt

    # Get trainer
    trainer = setup_transformers_trainer(model=model_peft,
                                         train_data=dataset,
                                         args=train_args,
                                         collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))

    print(f"Quant config: {quant_config.to_dict()} PEFT config: {peft_config.to_dict()}"
          f"Train Arg: {train_args.to_dict()} Dataset: {dataset}")

    # training
    trainer.train()  # train with SFTTrainer
    # log to WB

    # save model
    # trainer.save_model(params['output_dir'])

if __name__ == "__main__":
    index()
