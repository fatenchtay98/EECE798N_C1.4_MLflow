import yaml
import logging

def load_config(filepath="config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {filepath}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        return None

def setup_logging():
    """Sets up basic logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)
