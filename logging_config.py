import logging


def setup_logger(name="text_tuning"):
    # Create a logger with the given name
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)

    # Create a console handler and set the level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create a file handler and set the level to debug
    fh = logging.FileHandler('j_1_mgn_1.log')
    fh.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add the handler to the logger
    log.addHandler(ch)
    log.addHandler(fh)

    return log

logger = setup_logger()
