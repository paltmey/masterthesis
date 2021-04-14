import jsonlines
import os
import sys

from argparse import ArgumentParser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.comment_block import CommentBlock
from src.comment_classifier import CommentClassifier
from util.multiprocessing_utils import parallel_map
from util.file_utils import merge_files, remove_files


def get_comment_block(position, lines):
    start = position[0], position[1]
    end = position[2], position[3]
    comment_lines = lines[start[0]:end[0]+1]

    comment_block = CommentBlock(start, end, comment_lines)
    return comment_block


def process_file_comments(file_comments, comment_classifier):
    positions = file_comments['positions']

    if len(positions) < 1:
        return None

    file_path = file_comments['file_path']

    if not os.access(file_path, os.R_OK):
        return None

    with open(file_path, encoding='utf-8', errors='ignore') as file:
        lines = file.read().splitlines()

    pred = []
    for position in positions:
        label = 'NA'
        prob = 0

        comment_block = get_comment_block(position, lines)
        comment = str(comment_block)

        if comment:
            label, prob = comment_classifier.predict(comment)

        pred.append((label, f'{prob:.4f}'))

    classified_comments = {**file_comments, 'predictions': pred}

    return classified_comments


def process_files(process_id, all_file_comments, output_path, comment_classifier_model):
    path = f'{output_path}.part{process_id}'
    comment_classifier = CommentClassifier(comment_classifier_model)

    with jsonlines.open(path, 'w') as writer:
        for file_comments in tqdm(all_file_comments, desc=f'proc {process_id}', position=process_id):
            classified_comments = process_file_comments(file_comments, comment_classifier)
            if classified_comments:
                writer.write(classified_comments)


def run(input_path, output_path, comment_classifier_model, num_processes=1):
    print(f'Reading file comments from {input_path}')
    with jsonlines.open(input_path) as reader:
        all_file_comments = list(reader)

    print(f'Classifying comments in {len(all_file_comments)} files using {num_processes} processes.')
    parallel_map(process_files, all_file_comments, num_processes, output_path, comment_classifier_model)

    print(f'Merging {num_processes} files to {output_path}.')
    src_paths = [f'{output_path}.part{process_id}' for process_id in range(num_processes)]
    merge_files(src_paths, output_path)

    print(f'Removing {len(src_paths)} merged files.')
    remove_files(src_paths)


if __name__ == '__main__':
    parser = ArgumentParser(description='Classifies comments using the given comment classier.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--input_path', type=str,
                               help='Location of the file containing the comments of each source code file.',
                               required=True)
    requiredNamed.add_argument('--output_path', type=str,
                               help='Location of the output file containing the comment positions and classification predicftions per input file. Outputs in JSON lines format.',
                               required=True)
    requiredNamed.add_argument('--comment_classifier_model', type=str, help='Location of the trained fastText model.',
                               required=True)

    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to spin up.')

    args = parser.parse_args()

    run(args.input_path, args.output_path, args.comment_classifier_model, args.num_processes)
