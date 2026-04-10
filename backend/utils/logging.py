import logging 
import os
import yaml
from logging.handlers import RotatingFileHandler

CONFIG_PATH = os.path.join(os.path.dirname(__file__),'..','config','config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

log_config = config.get('logging',{})
LOG_DIR = log_config.get('log_dir','logs')

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def set_logger(name):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(log_config.get('level','INFO'))

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        #File-Handler
        file_path = os.path.join(LOG_DIR, log_config.get('log_file','app.log')) #app.log is default is not present any
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=log_config.get('max_bytes','1048576'),
            backupCount=log_config.get('backup_count','5')
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        #console-logging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger