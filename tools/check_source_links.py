"""Validate pinned case-study links offline; optionally verify local checkouts."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout-root', type=Path)
    args = parser.parse_args()
    repos = json.loads((ROOT/'case_studies/sources.json').read_text())['repositories']
    index = {r['repository'].lower():r for r in repos}
    verified = 0
    if args.checkout_root:
        for repo in repos:
            folder = args.checkout_root/repo['repository'].split('/')[-1]
            for entry in repo['files']:
                path = folder/entry['path']
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                    raise ValueError(f'source differs: {path}')
                verified += 1
    links = 0
    pattern = r'https://github.com/([^/]+/[^/]+)/blob/([^/]+)/([^\s)#]+)(?:#L(\d+))?'
    for page in (ROOT/'case_studies').rglob('*.md'):
        for name,commit,path,line in re.findall(pattern,page.read_text()):
            repo = index[name.lower()]
            if commit != repo['commit'] or path not in {f['path'] for f in repo['files']}:
                raise ValueError(f'unrecorded version/path in {page}: {name}/{commit}/{path}')
            if line and args.checkout_root:
                local = args.checkout_root/repo['repository'].split('/')[-1]/path
                if not 1 <= int(line) <= len(local.read_text().splitlines()):
                    raise ValueError(f'line outside file: {page}: {path}#{line}')
            links += 1
    print(f'Sources: {links} pinned links match manifest; {verified} local file hashes checked.')


if __name__ == '__main__':
    main()
