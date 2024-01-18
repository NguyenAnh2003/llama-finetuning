from peft import *

def get_model_state(model):
    """ get PEFT model state dict """
    state_dict = get_peft_model_state_dict(model)
    return state_dict

def get_trainable_params(model, peft_config):
    """ get trainable params
     Still keep model params
     Scaling from 7B to 3B
     """
    model = get_peft_model(model=model, peft_config=peft_config)
    state = model.print_trainable_parameters()  # get trainable params
    return state

# model params
def get_model_params(model):
    """ get model params -> 3B """
    params = [p.nelement() for p in model.parameters()]
    return params