"""Create a local NumPy-compatible Chumpy copy for legacy FLAME pickle loading."""
import argparse
import importlib.util
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / '.deps/chumpy')
    args = parser.parse_args()
    spec = importlib.util.find_spec('chumpy')
    if spec is None or spec.origin is None:
        parser.error('Install chumpy from requirements.txt before running this command')
    source = Path(spec.origin).parent
    if source.resolve() != args.output.resolve():
        shutil.copytree(source, args.output, dirs_exist_ok=True)
    path = args.output / '__init__.py'
    text = path.read_text().replace(
        'from numpy import bool, int, float, complex, object, unicode, str, nan, inf',
        'from builtins import bool, int, float, complex, object, str\n'
        'from numpy import nan, inf\nunicode = str')
    path.write_text(text)
    print(f'NumPy-compatible Chumpy: {args.output.resolve()}')


if __name__ == '__main__':
    main()
