import jsonlines
import os
import sys
import sentencepiece as spm

from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.code_tokenizer import INDENT_MARKER, DEDENT_MARKER, BEGIN_COMMENT_MARKER, END_COMMENT_MARKER, NEWLINE_MARKER, BEGIN_FILE_MARKER, END_FILE_MARKER, MISSING_MARKER


def run(input_path, output_path, vocab_size, character_coverage, input_sentence_size, num_reserved_markers=0, num_threads=1):
    with jsonlines.open(input_path) as reader:
        train_files = [file_meta['file_path'] for file_meta in reader]

    user_defined_symbols = [INDENT_MARKER, DEDENT_MARKER, BEGIN_COMMENT_MARKER, END_COMMENT_MARKER, NEWLINE_MARKER,
                            BEGIN_FILE_MARKER, END_FILE_MARKER, MISSING_MARKER]

    for i in range(num_reserved_markers):
        user_defined_symbols.append(f'<RES{i}>')

    spm.SentencePieceTrainer.Train(input=train_files,
                                   model_prefix=output_path,
                                   vocab_size=vocab_size,
                                   character_coverage=character_coverage,
                                   pad_id=3,
                                   input_sentence_size=input_sentence_size,
                                   user_defined_symbols=user_defined_symbols,
                                   num_threads=num_threads)


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a sentencepiece classifier.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--input_path', type=str, help='Location of the file containing the file metadata.',
                               required=True)
    requiredNamed.add_argument('--output_path', type=str, help='Location of the trained model and vocab. Creates two files "output_path".model and "output_path".vocab.',
                               required=True)

    parser.add_argument('--vocab_size', type=int, default=50000, help='Vocabulary size.')
    parser.add_argument('--character_coverage', type=float, default=1., help='Amount of characters covered by the model.')
    parser.add_argument('--input_sentence_size', type=int, default=40000000, help='Maximum samples the trainer loads.')
    parser.add_argument('--num_reserved_markers', type=int, default=0, help='If specified, adds the given number of reserved markers <RES#> to the user_defined_symbols.')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads to use for training.')

    args = parser.parse_args()

    run(args.input_path, args.output_path, args.vocab_size, args.character_coverage, args.input_sentence_size, args.num_reserved_markers, args.num_threads)
