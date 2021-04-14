import json
import os
import sys
import sentencepiece as spm

from argparse import ArgumentParser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.code_tokenizer import END_LINE_MARKER, BEGIN_FILE_MARKER, END_FILE_MARKER
from util.multiprocessing_utils import parallel_map
from util.file_utils import merge_files, remove_files


def process_file(file_path, tokenizer):
    with open(file_path) as file:
        lines = file.read().splitlines()
        if lines:
            file_tokens = [BEGIN_FILE_MARKER]
            for line in lines:
                file_tokens.extend(tokenizer.encode(line, out_type=str))
                file_tokens.append(END_LINE_MARKER)

            file_tokens.append(END_FILE_MARKER)
            return ' '.join(file_tokens)


def process_files(process_id, file_paths, output_path, tokenizer_model):
    path = f'{output_path}.part{process_id}'
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model)

    with open(path, 'w') as writer:
        for file_path in tqdm(file_paths, desc=f'proc {process_id}', position=process_id):
            tokenized_file = process_file(file_path, tokenizer)
            if tokenized_file:
                writer.write(tokenized_file + os.linesep)


def run(splits_path, output_dir, split, tokenizer_model, num_processes):
    output_path = os.path.join(output_dir, f'{split}_tokens.txt')

    print(f'Reading split file paths from {splits_path}')
    with open(splits_path) as file:
        splits = json.load(file)

    split_file_paths = splits[split]

    print(f'Tokenizing {len(split_file_paths)} files using {num_processes} processes.')
    parallel_map(process_files, split_file_paths, num_processes, output_path, tokenizer_model)

    print(f'Merging {num_processes} files to {output_path}.')
    src_paths = [f'{output_path}.part{process_id}' for process_id in range(num_processes)]
    merge_files(src_paths, output_path)

    print(f'Removing {len(src_paths)} merged files.')
    remove_files(src_paths)


if __name__ == '__main__':
    parser = ArgumentParser(description='Create a subword tokenized datset using the given sentencepiece model.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--splits_path', type=str, help='Location of the file containing the dataset splits.',
                               required=True)
    requiredNamed.add_argument('--output_dir', type=str,
                               help='Directory where the tokenized dataset should be saved. Each line in the output file corresponds to one source code file.',
                               required=True)
    requiredNamed.add_argument('--split', type=str, help='Name of the split that should be processed.',
                               required=True)
    requiredNamed.add_argument('--tokenizer_model', type=str, help='Location of the trained sentencepiece model.',
                               required=True)

    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to spin up.')

    args = parser.parse_args()

    run(args.splits_path, args.output_dir, args.split, args.tokenizer_model, args.num_processes)
