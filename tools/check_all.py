"""Run the complete GPU-free course checks with the current Python interpreter."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECKS = (
    'check_launch_and_schedule.py', 'check_theory.py', 'check_patterns.py',
    'check_attention_backward.py', 'check_resource_budget.py',
    'check_classic_operators.py', 'check_advanced_attention.py',
    'check_pipeline_protocol.py', 'check_source_links.py', 'check_docs.py',
)


def main():
    if not __debug__:
        raise SystemExit('Run without -O: teaching checks use assertions.')
    for name in CHECKS:
        print(f'\n{name}', flush=True)
        subprocess.run([sys.executable, str(ROOT/'tools'/name)], cwd=str(ROOT), check=True)
    print(f'\nAll {len(CHECKS)} checks passed. No GPU execution or performance validated.')


if __name__ == '__main__':
    main()
