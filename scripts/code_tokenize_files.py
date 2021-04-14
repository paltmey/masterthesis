import jsonlines
import os
import tokenize
import sys

from argparse import ArgumentParser
from simhash import Simhash
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.code_tokenizer import CodeTokenizer, BEGIN_COMMENT_MARKER
from src.comment_filter import CommentFilter
from util.multiprocessing_utils import parallel_map
from util.file_utils import merge_files, remove_files


def process_file(file_path, code_tokenizer):
    with open(file_path, errors='ignore') as file:
        tokens = tokenize.generate_tokens(file.readline)
        tokenized_lines = code_tokenizer.process_tokens(tokens)

        joined_lines = [' '.join(line) for line in tokenized_lines]
        tokenized_file = '\n'.join(joined_lines)

        hash = Simhash(tokenized_file).value
        comment_positions = [index for index, line in enumerate(tokenized_lines) if BEGIN_COMMENT_MARKER in line]

        return tokenized_file, hash, comment_positions, len(tokenized_lines)


def process_files(process_id, file_metas, output_dir, metadata_output_path):
    comment_filter = CommentFilter()
    code_tokenizer = CodeTokenizer(comment_filter)

    path = f'{metadata_output_path}.part{process_id}'
    with jsonlines.open(path, mode='w') as writer:
        for file_meta in tqdm(file_metas, desc=f'proc {process_id}', position=process_id):
            tokenized_file, hash, comment_positions, num_lines = process_file(file_meta['file_path'], code_tokenizer)

            repo_org, repo_name = file_meta['repo'].split('/')
            basename = os.path.splitext(os.path.basename(file_meta['filename']))[0]
            new_filename = os.path.join(repo_org, repo_name, f'{basename}.txt')
            new_file_path = os.path.normpath(os.path.join(output_dir, new_filename))

            new_file_meta = {
                'filename': new_filename,
                'file_path': new_file_path,
                'hash': hash,
                'num_lines': num_lines,
                'positions': comment_positions,
                'repo': file_meta['repo'],
            }
            writer.write(new_file_meta)

            os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
            with open(new_file_path, 'w') as file:
                file.write(tokenized_file)


def run(input_path, output_dir, metadata_output_path, num_processes=1):
    print(f'Reading file metadata from {input_path}')
    with jsonlines.open(input_path) as reader:
        file_metas = list(reader)

    print(f'Tokenizing {len(file_metas)} files using {num_processes} processes.')
    parallel_map(process_files, file_metas, num_processes, output_dir, metadata_output_path)

    print(f'Merging {num_processes} files to {metadata_output_path}.')
    src_paths = [f'{metadata_output_path}.part{process_id}' for process_id in range(num_processes)]
    merge_files(src_paths, metadata_output_path)

    print(f'Removing {len(src_paths)} merged files.')
    remove_files(src_paths)


if __name__ == '__main__':
    parser = ArgumentParser(description='Code tokenize raw source code files.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--input_path', type=str, help='Location of the file containing the metadata of files with suitable comments.',
                               required=True)
    requiredNamed.add_argument('--output_dir', type=str,
                               help='Location of the directory, under which all tokenized files should be saved.',
                               required=True)
    requiredNamed.add_argument('--metadata_output_path', type=str,
                               help='Location of the output file containing the metadata, including the calculated hash, of all tokenized files. Outputs in JSON lines format.',
                               required=True)

    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to spin up.')

    args = parser.parse_args()

    run(args.input_path, args.output_dir, args.metadata_output_path, args.num_processes)
