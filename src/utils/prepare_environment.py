import os
from pathlib import Path
from omegaconf import OmegaConf

def read_config(base_dir='config/base', local_dir='config/local'):
    merged_configs = {}

    base_path = Path(os.getcwd()).joinpath(base_dir)
    local_path = Path(os.getcwd()).joinpath(local_dir)
    conf_files = list(set(os.listdir(base_path)+os.listdir(local_path)))
    for filename in conf_files:
        if filename.endswith(".yml") or filename.endswith(".yaml"):
            base_path_file = os.path.join(base_path, filename)
            local_path_file = os.path.join(local_path, filename)

            if os.path.exists(local_path_file):
                local_conf = OmegaConf.load(local_path_file)
                if os.path.exists(base_path_file):
                    base_conf = OmegaConf.load(base_path_file)
                    merged_conf = OmegaConf.merge(base_conf, local_conf)
                else:
                    merged_conf = local_conf
            else:
                base_conf = OmegaConf.load(base_path_file)
                merged_conf = base_conf

            merged_configs[filename.split('.')[0]] = merged_conf

    return merged_configs

def set_api_keys():

    config = read_config()
    for k, v in config['credentials']['API_KEYS'].items():
        print(f"Setting up environment variable: {k}")
        os.environ[k] = v
