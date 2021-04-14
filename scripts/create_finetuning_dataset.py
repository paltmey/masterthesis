import os
import sentencepiece as spm
import sys
import torch

from argparse import ArgumentParser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.file_utils import get_line_count


def run(context_input_path, target_input_path, output_path, tokenizer_model):
    print('Loading tokenizer.')
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model)

    print('Reading line count.')
    num_lines = get_line_count(context_input_path)

    print(f'Loading tokens from {context_input_path} and {target_input_path}.')
    with open(context_input_path) as context_reader, open(target_input_path) as target_reader:
        token_ids = []
        for context, target in tqdm(zip(context_reader, target_reader), total=num_lines):
            context_tokens = context.split()
            target_tokens = target.split()
            sample_tokens = context_tokens + target_tokens
            ids = tokenizer.piece_to_id(sample_tokens)
            token_ids.append((torch.tensor(ids, dtype=torch.int32), len(context_tokens)))

        print(f'Saving token ids to {output_path}')
        torch.save(token_ids, output_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='Create a finetuning set.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--context_input_path', type=str, help='Location of the file containing the context sequences.',
                               required=True)
    requiredNamed.add_argument('--target_input_path', type=str, help='Location of the file containing the target sequences.',
                               required=True)
    requiredNamed.add_argument('--output_path', type=str,
                               help='Location of the output file containing the id dataset. Each input line is transformed to ids and saved as a pytorch tensor.',
                               required=True)
    requiredNamed.add_argument('--tokenizer_model', type=str, help='Location of the trained sentencepiece model.',
                               required=True)

    args = parser.parse_args()

    run(args.context_input_path, args.target_input_path, args.output_path, args.tokenizer_model)
