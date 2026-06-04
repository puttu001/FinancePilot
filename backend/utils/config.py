import yaml
import os
import re
from functools import lru_cache
from dotenv import load_dotenv
from .logging import set_logger
logger = set_logger(__name__)

ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

def _resolve_env_placeholders(value):
    """Recursively replace ${VAR_NAME} placeholders using os.environ."""
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]
    if isinstance(value, str):
        def _replace(match):
            var_name = match.group(1)
            env_val = os.getenv(var_name)
            if env_val is None:
                logger.warning(f"Environment variable '{var_name}' is not set")
                return match.group(0)
            return env_val

        return ENV_VAR_PATTERN.sub(_replace, value)
    return value

@lru_cache(maxsize=1)
def load_config():
    #sourcing path to config.yaml
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path,'config','config.yaml')

    try:
        # Load .env values into process environment before resolving placeholders.
        load_dotenv()
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f) or {}
            return _resolve_env_placeholders(raw_config)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}", exc_info=True)
        return {}
def get_config(section_name, default=None):
    """
    Helper to fetch a specific section.
    """
    config = load_config()
    return config.get(section_name, default or {})