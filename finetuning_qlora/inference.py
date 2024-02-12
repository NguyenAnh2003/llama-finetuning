import peft
from helpers.utils import load_params
from peft import PeftModel, PeftConfig

def main():
    """
    Inference with fine tuned model
    PEFT model turns out can be considered as Adapter because it takes
    trainable params and trains these params, for inference we take these
    params by PeftModel class that take fine-tuned params
    """
    params = load_params("../config/config.yml") # params file
    """
    load pre-trained peft saved model for inference
    """
    model = PeftModel.from_pretrained(params['saved_model']) # peft saved model
    print(f"Model info: {model.state_dict()}") # logging model

if __name__ == "__main__":
    main()