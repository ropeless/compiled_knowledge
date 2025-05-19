"""
Run all the demo scripts (except nominated exclusions).
This may be used as a 'smoke test'.
"""
import sys
from pathlib import Path
from subprocess import call

from ck_demos.utils.stop_watch import StopWatch

DEMO_TOP_DIR = '.'  # relative to the directory this script is in

# These are scripts and directories we explicitly will not include
EXCEPTIONS = [
    'utils',  # package of utilities
    'ace',  # 3rd party, optional dependency
]

# These are what are printed as a separator lines
LINE = '-' * 80
LINE_2 = '=' * 80


def main() -> int:
    python_exe = sys.executable
    print('Python executable:', python_exe)
    print()

    errors = []
    script_count = 0
    python_env = None
    total_time = StopWatch()

    # Start search of demo directories in the script directory.
    script_path = Path(__file__)
    script_name = script_path.stem
    demos_top_dir_path = (script_path.parent / DEMO_TOP_DIR)
    dirs = find_demo_dirs(demos_top_dir_path)

    # Always exclude self if we happen to find our own script.
    exceptions = set(EXCEPTIONS + [Path(__file__).name])

    for demo_dir in dirs:
        for script in demo_dir.iterdir():
            if (
                    script.is_file() and
                    script.name.endswith('.py') and
                    (not script.name.startswith('_')) and
                    script.name not in exceptions
            ):
                print(LINE)
                print(script.name)
                print(LINE)
                script_count += 1

                time = StopWatch()
                return_code = call([python_exe, script.as_posix()], env=python_env)
                time.stop()

                if return_code != 0:
                    errors.append(f'Error code {return_code}: {script.name}')
                print(LINE)
                print(f'exited with code {return_code}, time = {time}, file = {script.name}')

    print(LINE_2)
    print(f'Done running {script_name}, {script_count} scripts in {total_time}s')
    print(f'Number of errors: {len(errors)}')
    for error in errors:
        print(error)

    # Provide a useful exit code.
    if len(errors) > 0:
        return -1
    else:
        return 0


def find_demo_dirs(demos_top_dir_path: Path):
    return [demos_top_dir_path] + [
        d
        for d in demos_top_dir_path.iterdir()
        if d.is_dir() and d.name not in EXCEPTIONS
    ]


if __name__ == '__main__':
    ret_code = main()
    exit(ret_code)
