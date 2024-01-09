from setup.setup import *
from llama_recipe.model_utils import *
from helpers.utils import *

params = load_params("../config/config.yml")
print(f"{params}")
quantization_config = setup_4_bit_quant_config(params)  # quantization config
# test getting PEFT model
model, tokenizer = setup_pretrained_model(params['base_model'],
                                          bnb_config=quantization_config)  # define model

print(f"Model state dict: {model.state_dict()}")
# PEFT config
peft_config = setup_peft_config(params)  #
peft_model = setup_peft_model(model=model, peft_config=peft_config)
print(f"PEFT model: {get_model_state(model)} PEFT config: {peft_config.to_dict()}")