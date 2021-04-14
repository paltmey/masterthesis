import os
import sentencepiece as spm
import sys
import torch

from argparse import ArgumentParser
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.code_generator import CodeGenerator
from util.file_utils import get_line_count


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_prediction(line, tokenizer, code_generator, generate_args):
    tokens = line.split()
    input_ids = tokenizer.piece_to_id(tokens)
    generated_code = code_generator.generate_raw([input_ids], generate_args)[0]
    prediction_ids = generated_code[len(input_ids):]
    prediction = tokenizer.id_to_piece(prediction_ids.tolist())
    prediction = ['<EOL>' if token == '<unk>' else token for token in prediction] #  TODO remove
    return prediction


def run(input_path, output_path, model, tokenizer_model, limit_input, generate_args):
    print(generate_args)

    print('Reading line count.')
    num_lines = get_line_count(input_path)

    print('Loading model.')
    model = AutoModelForCausalLM.from_pretrained(model).to(DEFAULT_DEVICE)
    print('Loading tokenizer.')
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model)
    code_generator = CodeGenerator(model, tokenizer, DEFAULT_DEVICE)

    limit = limit_input if limit_input > 0 else num_lines

    print(f'Generating predictions with device {DEFAULT_DEVICE}.')
    with open(input_path) as file, open(output_path, 'w') as writer:
        for index, line in tqdm(enumerate(file), total=limit):
            try:
                prediction = get_prediction(line, tokenizer, code_generator, generate_args)
                prediction_str = ' '.join(prediction)
                writer.write(prediction_str + os.linesep)
            except Exception as e:
                tqdm.write(f'Error for index {index}: {e}', file=sys.stderr)

            if index >= limit:
                break


if __name__ == '__main__':
    parser = ArgumentParser(description='Create predictions for the given contexts.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--input_path', type=str,
                               help='Location of the file containing the context file.',
                               required=True)
    requiredNamed.add_argument('--output_path', type=str,
                               help='Location of the output file containing the comment predictions. Outputs in txt format.',
                               required=True)
    requiredNamed.add_argument('--model', type=str, help='Location of the trained transformer model.',
                               required=True)
    requiredNamed.add_argument('--tokenizer_model', type=str, help='Location of the trained sentencepiece model.',
                               required=True)

    parser.add_argument('--limit_input', type=int, default=0, help='Limits the of input data to the given number of examples.')
    parser.add_argument('--min_length', type=int, default=2, help='The minimum length of the sequence to be generated.')
    parser.add_argument('--max_length', type=int, default=512, help='The maximum length of the sequence to be generated.')
    parser.add_argument('--sample', action='store_true', default=False, help='Whether or not to use sampling, use greedy decoding otherwise.')
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams for beam search. 1 means no beam search.')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='The number of independently computed returned sequences for each element in the batch.')
    parser.add_argument('--temperature', type=float, default=1.0, help='The value used to module the next token probabilities.')
    parser.add_argument('--top_k', type=int, default=0, help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
    parser.add_argument('--top_p', type=float, default=1.0, help='If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.')

    args = parser.parse_args()

    generate_args = {
        'min_length': args.min_length,
        'max_length': args.max_length,
        'do_sample': args.sample,
        'num_beams': args.num_beams,
        'num_return_sequences': args.num_return_sequences,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'bad_words_ids': [[9]]  # prevent generating BOF tokens
    }

    run(args.input_path, args.output_path, args.model, args.tokenizer_model, args.limit_input, generate_args)
