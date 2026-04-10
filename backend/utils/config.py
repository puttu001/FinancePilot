import yaml
import os
from .logging import set_logger
logger = set_logger(__name__)

def load_config():
    #sourcing path to config.yaml
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path,'config','config.yaml')

    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}", exc_info=True)
        return {}