import jsonlines
import os
import sys
import tokenize

from argparse import ArgumentParser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.comment_block import CommentBlock
from src.comment_filter import CommentFilter
from util.multiprocessing_utils import parallel_map
from util.file_utils import merge_files, remove_files


def get_comment_blocks(tokens):
    new_block = True
    first_nl = False
    comment_block = None

    for token in tokens:
        if token.type == tokenize.COMMENT:
            if new_block:
                new_block = False
                start = (token.start[0]-1, token.start[1])  # -1, because line starts with 1
                end = (token.end[0]-1, token.end[1]) # -1, because line starts with 1
                lines = [token.line]
                comment_block = CommentBlock(start=start, end=end, lines=lines)
            else:
                comment_block.end = (token.end[0]-1, token.end[1]) # -1, because line starts with 1
                comment_block.lines.append(token.line)

        if token.type == tokenize.NL and not first_nl:
            first_nl = True
        else:
            first_nl = False

        if token.type != tokenize.COMMENT and not new_block and not first_nl:
            new_block = True
            yield comment_block


def get_comment_blocks_for_file(filename, comment_filter):
    if os.access(filename, os.R_OK):
        with open(filename, encoding='utf-8', errors='ignore') as file:
            try:
                tokens = tokenize.generate_tokens(file.readline)
                comment_blocks = get_comment_blocks(tokens)

                return comment_filter.filter(comment_blocks)

            except tokenize.TokenError:
                tqdm.write(f'TokenError in file {filename}', file=sys.stderr)
                return []
            except IndentationError:
                tqdm.write(f'IndentationError in file {filename}', file=sys.stderr)
                return []
            except ValueError:
                tqdm.write(f'ValueError in file {filename}', file=sys.stderr)
                return []
    else:
        tqdm.write(f'Can not access file {filename}', file=sys.stderr)
        return []


def process_file(file_path, base_path, comment_filter):
    relative_filepath = os.path.normpath(file_path.replace(base_path, '').lstrip(os.sep))
    path_parts = relative_filepath.split(os.sep)
    org = path_parts[0]
    repo = path_parts[1]

    comment_blocks = get_comment_blocks_for_file(file_path, comment_filter)
    comment_positions = [comment_block.start + comment_block.end for comment_block in comment_blocks]
    file_comments = {
        'file_path': file_path,
        'filename': relative_filepath,
        'num_comments': len(comment_blocks),
        'positions': comment_positions,
        'repo': f'{org}/{repo}'
    }

    lines = []
    for comment_block in comment_blocks:
        lines.append(str(comment_block))

    return file_comments, lines


def process_files(process_id, file_paths, base_path, output_path, line_output_path):
    path = f'{output_path}.part{process_id}'
    comment_filter = CommentFilter()

    line_writer = None
    if line_output_path:
        line_path = f'{line_output_path}.part{process_id}'
        line_writer = open(line_path, 'w')

    with jsonlines.open(path, 'w') as writer:
        for file_path in tqdm(file_paths, desc=f'proc {process_id}', position=process_id):
            file_comments, lines = process_file(file_path, base_path, comment_filter)
            if file_comments and file_comments['num_comments'] > 0:
                writer.write(file_comments)

                if line_writer:
                    line_writer.write('\n'.join(lines))

    if line_writer:
        line_writer.close()


def run(file_paths_file, base_path, output_path, line_output_path='', num_processes=1):
    print(f'Reading file paths from {file_paths_file}')
    with open(file_paths_file) as file:
        file_paths = file.read().splitlines()

    print(f'Finding comments in {len(file_paths)} files using {num_processes} processes.')
    parallel_map(process_files, file_paths, num_processes, base_path, output_path, line_output_path)

    print(f'Merging {num_processes} files to {output_path}.')
    src_paths = [f'{output_path}.part{process_id}' for process_id in range(num_processes)]
    merge_files(src_paths, output_path)

    print(f'Removing {len(src_paths)} merged files.')
    remove_files(src_paths)

    if line_output_path:
        print(f'Merging {num_processes} files to {line_output_path}.')
        src_paths = [f'{line_output_path}.part{process_id}' for process_id in range(num_processes)]
        merge_files(src_paths, line_output_path)

        print(f'Removing {len(src_paths)} merged files.')
        remove_files(src_paths)


if __name__ == '__main__':
    parser = ArgumentParser(description='Find comments within source files. The comments are prefiltered.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--file_paths_file', type=str, help='Location of the file containing the file paths.',
                               required=True)
    requiredNamed.add_argument('--base_path', type=str, help='Location of the directory containing the repositories.',
                               required=True)
    requiredNamed.add_argument('--output_path', type=str,
                               help='Location of the output file containing the comment positions per input file. Outputs in JSON lines format.',
                               required=True)

    parser.add_argument('--line_output_path', type=str, default='',
                        help='If specified, outputs all comments as plain text to the given location. Outputs in txt format.')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to spin up.')

    args = parser.parse_args()

    run(args.file_paths_file, args.base_path, args.output_path, args.line_output_path, args.num_processes)
