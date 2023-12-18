import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig as bnb_config
import os


# set up LoRA config
def setup_4_bit_quant_config():
    compute_dtype = getattr(torch, "float16")
    bit4_config = bnb_config(
        #
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False
    )
    return bit4_config

def setup_peft_config():


