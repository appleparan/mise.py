import tarfile
from pathlib import Path
import re

def untar():
    p = Path('.')
    files = list(p.glob('**/DUex-*.tar.gz'))
    regexexp = re.compile(r'DUex-'
        r'(?P<year>[2][0-9][0-9][0-9])-(?P<month>[0-1][0-9])\.tar\.gz')
    for f in files:
        matched = re.match(regexexp, str(f))
        if matched:
            y = matched.group('year')
            m = matched.group('month')
            # extract path
            extract_path = Path(y) / Path(m)
            extract_path.mkdir(parents=True, exist_ok=True)

            tar = tarfile.open(f, 'r:gz')
            for item in tar:
                tar.extract(item, extract_path)


if __name__ == '__main__':
    untar()
