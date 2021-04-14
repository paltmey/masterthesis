import nltk
import os
import re
import sys
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.file_utils import get_line_count


def process_text(text):
    text_lower = text.lower()
    sentences = nltk.sent_tokenize(text_lower)
    tokens = []
    for sentence in sentences:
        cleaned_sentence = re.sub(r'[.\-()\[\]`"\'#]', '', sentence)
        tokens.extend(nltk.word_tokenize(cleaned_sentence))

    return ' '.join(tokens)


def create_cleaned_file(input_file, output_file):
    with open(input_file) as read_file:
        with open(output_file, 'w') as write_file:
            line = read_file.readline()

            while line:
                processed_line = process_text(line)
                write_file.write(processed_line + '\n')
                line = read_file.readline()


def run(input_path, output_path):
    print(f'Getting line count of {input_path}')
    num_lines = get_line_count(input_path)

    print(f'Preprocessing {num_lines} comments.')
    with open(input_path) as read_file, open(output_path, 'w') as write_file, tqdm(total=num_lines) as pbar:
        line = read_file.readline()

        while line:
            processed_line = process_text(line)
            write_file.write(processed_line + '\n')
            pbar.update()
            line = read_file.readline()


if __name__ == '__main__':
    parser = ArgumentParser(description='Preprocess comment strings for fasttext.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--input_path', type=str, help='Location of the file containing the comment positions.', required=True)
    requiredNamed.add_argument('--output_path', type=str, help='Location of the processed comments.', required=True)
    args = parser.parse_args()

    run(args.input_path, args.output_path)
