import jsonlines
import os
import shutil
import sys

from argparse import ArgumentParser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.multiprocessing_utils import parallel_map


def copy_file(file_meta, output_dir):
    src_path = file_meta['file_path']
    dst_path = os.path.join(output_dir, file_meta['filename'])

    if not os.path.exists(dst_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copyfile(src_path, dst_path, follow_symlinks=False)


def process_files(process_id, suitable_file_metas, output_dir):
    for file_meta in tqdm(suitable_file_metas, desc=f'proc {process_id}', position=process_id):
        copy_file(file_meta, output_dir)


def contains_suitable_comment(file_meta, threshold):
    predictions = file_meta['predictions']

    for prediction in predictions:
        label = prediction[0]
        prob = float(prediction[1])

        if label == 'positive' and prob >= threshold:
            return True

    return False


def replace_base_path(file_meta, output_dir):
    file_meta['file_path'] = os.path.normpath(os.path.join(output_dir, file_meta['filename']))
    return file_meta


def run(input_path, output_dir, metadata_output_path, threshold, num_processes=1):
    print(f'Reading file metadata from {input_path}')
    with jsonlines.open(input_path) as reader:
        suitable_file_metas = [file_meta for file_meta in reader if contains_suitable_comment(file_meta, threshold)]

    print(f'Copying suitable {len(suitable_file_metas)} files using {num_processes} processes.')
    parallel_map(process_files, suitable_file_metas, num_processes, output_dir)

    print(f'Saving file metadata with updated file_paths to {metadata_output_path}')
    with jsonlines.open(metadata_output_path, mode='w') as writer:
        new_file_metas = [replace_base_path(file_meta, output_dir) for file_meta in suitable_file_metas]
        writer.write_all(new_file_metas)


if __name__ == '__main__':
    parser = ArgumentParser(description='Copies files with at least one comment above the threshold to an output directory.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--input_path', type=str, help='Location of the file containing the classified comments metadata.',
                               required=True)
    requiredNamed.add_argument('--output_dir', type=str,
                               help='Location of the directory, under which all files with suitable comments should be copied.',
                               required=True)
    requiredNamed.add_argument('--metadata_output_path', type=str,
                               help='Location of the output file containing the metadata of all files containing suitable comments. Outputs in JSON lines format.',
                               required=True)
    requiredNamed.add_argument('--threshold', type=float, help='Minimum confidence threshold to consider a positive comment as suitable.',
                               required=True)

    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to spin up.')

    args = parser.parse_args()

    run(args.input_path, args.output_dir, args.metadata_output_path, args.threshold, args.num_processes)
