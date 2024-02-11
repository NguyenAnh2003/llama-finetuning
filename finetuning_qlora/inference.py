import peft
from helpers.utils import load_params
from peft import PeftModel, PeftConfig
import logging
"""
load model: including quantization config, load_in4bit
get peft model
model.eval()
"""

def main():
    params = load_params("../config/config.yml") # params file
    """
    load pre-trained peft saved model for inference
    """
    model = PeftModel.from_pretrained(params['saved_model']) # peft saved model
    print(f"Model info: {model.state_dict()}") # logging model