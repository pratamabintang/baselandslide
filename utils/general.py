import os
import re
import glob
import logging
from pathlib import Path


def set_logging(name=None, verbose=True):
    """Configure python logging format."""
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (verbose and rank in [-1, 0]) else logging.WARNING
    )
    return logging.getLogger(name)


def check_file(file):
    """Check if file exists and return resolved Path object."""
    file = Path(file)
    if file.is_file():
        return file
    files = glob.glob(str(file))
    assert len(files), f"File Not Found: {file}"
    assert len(files) == 1, f"Multiple files match: {file}"
    return Path(files[0])


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    Increments path, i.e. runs/train/exp --> runs/train/exp, runs/train/exp2, runs/train/exp3 etc.
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f"{path}{sep}{n}{suffix}")
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def colorstr(*inputs):
    """Format string with ANSI escape colors."""
    *colors, string = inputs if len(inputs) > 1 else ('blue', 'bold', inputs[0])
    colors_dict = {
        'black': '\033[30m', 'red': '\033[31m', 'green': '\033[32m', 'yellow': '\033[33m',
        'blue': '\033[34m', 'magenta': '\033[35m', 'cyan': '\033[36m', 'white': '\033[37m',
        'bright_black': '\033[90m', 'bright_red': '\033[91m', 'bright_green': '\033[92m',
        'bright_yellow': '\033[93m', 'bright_blue': '\033[94m', 'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m', 'bright_white': '\033[97m', 'end': '\033[0m',
        'bold': '\033[1m', 'underline': '\033[4m'
    }
    return ''.join(colors_dict[x] for x in colors) + f'{string}' + colors_dict['end']


def parse_options_with_config(parser):
    """
    Parse CLI options while supporting pre-loading default values from a YAML configuration file.
    CLI arguments explicitly specified by the user will override values defined in the YAML config.
    """
    import argparse
    import yaml

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default='', help='path to yaml config file')
    config_args, _ = config_parser.parse_known_args()

    if config_args.config:
        config_path = check_file(config_args.config)
        with open(config_path, 'r') as f:
            yaml_defaults = yaml.safe_load(f) or {}

        if yaml_defaults and isinstance(yaml_defaults, dict):
            normalized_defaults = {k.replace('-', '_'): v for k, v in yaml_defaults.items()}
            parser.set_defaults(**normalized_defaults)

    return parser.parse_args()

