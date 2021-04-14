import os
import subprocess
from argparse import ArgumentParser


def run(start_dir, output_path, file_whitelist):
    print(f'Getting file paths under {start_dir}')
    print(f'File whitelist: "{file_whitelist}"')
    start_dir = os.path.abspath(start_dir)
    with open(output_path, 'w') as file:
        process = subprocess.Popen(['find', start_dir, '-type', 'f', '-name', file_whitelist], stdout=file)
        process.wait()


if __name__ == '__main__':
    parser = ArgumentParser(description='Create file containing all file paths.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--start_dir', type=str, help='Location of the directory containing the files to find.',
                               required=True)
    requiredNamed.add_argument('--output_path', type=str, help='Location of the output file containing the file paths.')

    parser.add_argument('--file_whitelist', type=str, default='*.py', help='Shell Pattern to find matching files')

    args = parser.parse_args()

    run(args.start_dir, args.output_path, args.file_whitelist)