import argparse
import yaml


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise argparse.ArgumentTypeError('Use true or false')


def load_config(path, section):
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    values = data.get(section, data)
    if not isinstance(values, dict):
        raise ValueError(f'{section} must be a mapping')
    return values


def save_intermediates(path):
    return load_config(path, 'output').get('save_intermediates', False) if path else False


def parse_config(parser, section=None):
    parser.add_argument('--config', help='YAML defaults; command-line values take precedence')
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        values = load_config(preliminary.config, section)
        known = {action.dest for action in parser._actions}
        unknown = values.keys() - known
        if unknown:
            parser.error(f'Unknown config keys: {sorted(unknown)}')
        parser.set_defaults(**values)
    return parser.parse_args()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--get', choices=['output.save_intermediates'], required=True)
    args = parser.parse_args()
    print(str(bool(save_intermediates(args.config))).lower())
