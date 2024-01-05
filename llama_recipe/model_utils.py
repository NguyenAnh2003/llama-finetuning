from peft import *
def get_model_state(model):
    """ get PEFT model state dict """
    state_dict = get_peft_model_state_dict(model)
    return state_dict

def get_trainable_params(model, peft_config):
    model = get_peft_model(model=model, peft_config=peft_config)
    model.print_trainable_parameters() # get trainable params