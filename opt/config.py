import os
import yaml
from opt.parser import parse_args

def init_config():
    config = {}

    current_path = os.path.dirname(os.path.realpath(__file__))
    basic_init_file = os.path.join(current_path, '../assets/overall.yaml')
    basic_conf = yaml.load(open(basic_init_file), Loader=yaml.loader.SafeLoader)
    config.update(basic_conf)

    args = parse_args()
    args_conf = vars(args)
    config['bs'] = args_conf.get('bs', 4)
    # config['seed'] = args_conf.get('seed', 998)
    config['max_new_tokens'] = args_conf.get('max_new_tokens', 8192)
    config['model'] = args_conf.get('model', config.get('model', ''))
# Optionally keep model_dir for legacy/local, but recommend just model

    # If you need a local model_dir, pass it via --model_dir at runtime
    # 1)  record backend  (default 'local' was added to parser.py)
    config['backend'] = args_conf.get('backend', 'local')

    # 2)  choose which model‑specific yaml to read    
    model_file = 'local' if config['backend'] == 'local' else 'openai'
    model_init_file = os.path.join(current_path, f'../assets/{model_file}.yaml')
    model_conf = yaml.load(open(model_init_file), Loader=yaml.loader.SafeLoader)

    # 3)  if the yaml is a list, flatten it into one dict
    if isinstance(model_conf, list):
        flat_conf = {}
        for d in model_conf:                 # each d is a  {key: value}
            flat_conf.update(d)
        model_conf = flat_conf

    config.update(model_conf)

    # 4)  override with cli‑flags
    for k, v in config.items():
        if k in args_conf and args_conf[k] is not None:
            config[k] = args_conf[k]
        else:
            config[k] = v

    return config


