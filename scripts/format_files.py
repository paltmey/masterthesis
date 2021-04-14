import subprocess
from argparse import ArgumentParser


def run(src_dir, fast=False):
    print(f'Formatting all files under {src_dir} using black.')
    cmd = ['black']
    if fast:
        cmd.append('--fast')
    cmd.append(src_dir)
    subprocess.run(cmd)


if __name__ == '__main__':
    parser = ArgumentParser(description='Format all files using the black Python formatter.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--src_dir', type=str, help='Directory where to run formatting.',
                               required=True)
    parser.add_argument("--fast", help='If --fast given, skip temporary sanity checks.', action='store_true', default=False)

    args = parser.parse_args()

    run(args.src_dir, args.fast)
