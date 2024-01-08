from transformers import logging
def setup_logger():
    """ Setup logger """
    logging.set_verbosity_error() #

    return logging