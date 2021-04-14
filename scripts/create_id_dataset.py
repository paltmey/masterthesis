import os
import sys
import sentencepiece as spm
import torch

from argparse import ArgumentParser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.file_utils import get_line_count


def run(input_dir, split, output_dir, tokenizer_model):
    print('Loading tokenizer.')
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model)

    input_path = os.path.join(input_dir, f'{split}_tokens.txt')
    output_path = os.path.join(output_dir, f'{split}_ids.pt')

    print(f'Getting line count of {input_path}')
    num_lines = get_line_count(input_path)

    print(f'Loading tokens from {input_path}')
    token_ids = []
    with open(input_path) as file:
        for line in tqdm(file, total=num_lines):
            tokens = line.split()
            ids = tokenizer.piece_to_id(tokens)
            token_ids.append(torch.tensor(ids, dtype=torch.int32))

    print(f'Saving token ids to {output_path}')
    torch.save(token_ids, output_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='Create an id dataset.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--input_dir', type=str, help='Location of the file containing the tokenized code files.',
                               required=True)
    requiredNamed.add_argument('--split', type=str, help='Name of the split that should be processed.',
                               required=True)
    requiredNamed.add_argument('--output_dir', type=str,
                               help='Location of the output dir containing the id dataset. Each input line is transformed to ids and saved as a pytorch tensor.',
                               required=True)
    requiredNamed.add_argument('--tokenizer_model', type=str, help='Location of the trained sentencepiece model.',
                               required=True)

    args = parser.parse_args()

    run(args.input_dir, args.split, args.output_dir, args.tokenizer_model)
