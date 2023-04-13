import argparse
import pathlib
import subprocess
import os
from typing import Dict, List, Set


def load_environment(environment_path: pathlib.Path) -> Dict[str, str]:
    env = os.environ.copy()
    with environment_path.open(mode='r') as f:
        for line in f:
            var_name, var_value = line.strip().split('=')
            env[var_name] = var_value
    return env

def run_full_gridsearch_session(
    report_path: pathlib.Path,
    device_idx: int,
    environment: Dict[str, str]
):
    if report_path.exists():
        print('Continuing session...')
        action = 'CONTINUE'
        return_code = subprocess.call([
            'deepsysid',
            'session',
            '--enable-cuda',
            f'--device-idx={device_idx}',
            f'--reportin={report_path}',
            report_path,
            action
        ], env=environment)
    else:
        print('Starting session from fresh.')
        action = 'NEW'
        return_code = subprocess.call([
            'deepsysid',
            'session',
            '--enable-cuda',
            f'--device-idx={device_idx}',
            report_path,
            action
        ], env=environment)

    if return_code != 0:
        raise ValueError('Failed running gridsearch session.')

    action = 'TEST_BEST'
    subprocess.call([
        'deepsysid',
        'session',
        '--enable-cuda',
        f'--device-idx={device_idx}',
        f'--reportin={report_path}',
        report_path,
        action
    ], env=environment)

def main():
    parser = argparse.ArgumentParser('Run experiments for the 4-DOF ship in-distribution dataset.')
    parser.add_argument('device')
    args = parser.parse_args()

    device_idx = int(args.device)

    main_path = pathlib.Path(__file__).parent.parent.absolute()
    report_path = main_path.joinpath('configuration').joinpath('progress-ship.json')
    environment_path = main_path.joinpath('environment').joinpath('ship-ind.env')

    environment = load_environment(environment_path)

    run_full_gridsearch_session(
        report_path=report_path,
        device_idx=device_idx,
        environment=environment
    )


if __name__ == '__main__':
    main()