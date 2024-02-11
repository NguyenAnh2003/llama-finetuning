import wandb
from transformers import logging
from dotenv import load_dotenv
import os

# load env vars
load_dotenv()
# auth wandb
wandb.login(key=os.getenv("WANDB_API"))