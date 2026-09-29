from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')


def load_config(path=None):
    config_path = Path(path) if path else ROOT / 'configs' / 'config.yaml'
    with config_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['_root'] = str(ROOT)
    cfg['_roboflow_api_key'] = os.getenv('ROBOFLOW_API_KEY')
    return cfg
