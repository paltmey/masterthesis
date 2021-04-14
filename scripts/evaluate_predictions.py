import os
import sentencepiece as spm
import sys

from argparse import ArgumentParser
from Levenshtein import distance
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.code_tokenizer import CodeTokenizer
from src.comment_filter import CommentFilter


def calculate_levenshtein_similarity(refs, hyps):
    scores = []
    for ref, hyp in zip(refs, hyps):
        edit_distance = distance(ref, hyp)
        score = 1-(edit_distance/max(len(ref), len(hyp)))
        scores.append(score)

    avg_score = sum(scores)/len(scores)
    return avg_score


def untokenize(tokens, tokenizer, code_tokenizer):
    decoded_lines = tokenizer.decode([tokens])
    decoded_line_tokens = [line.split() for line in decoded_lines]
    code = code_tokenizer.untokenize(decoded_line_tokens)
    return code


def preprocess(lines, case_sensitive, tokenizer, code_tokenizer):
    preprocessed_lines = []

    for line in lines:
        tokens = line.split()

        if code_tokenizer:
            code_str = untokenize(tokens, tokenizer, code_tokenizer)
        else:
            code_str = tokenizer.decode(tokens)

        if not code_str:
            code_str = ' '

        if not case_sensitive:
            code_str = code_str.lower()
        preprocessed_lines.append(code_str)
    return preprocessed_lines


def run(predictions_path, targets_path, tokenizer_model, metrics, case_sensitive=False, untokenize=False):
    metric_list = [metric.strip() for metric in metrics.split(',')]

    if not metric_list:
        print('No metrics given.')
        return

    print('Reading targets.')
    with open(targets_path) as file:
        targets = file.read().splitlines()

    print('Reading predictions')
    with open(predictions_path) as file:
        predictions = file.read().splitlines()

    if len(predictions) != len(targets):
        print('Targets and predictions are not of the same length. Truncating targets.')
        targets = targets[:len(predictions)]

    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model)

    code_tokenizer = None
    if untokenize:
        comment_filter = CommentFilter()
        code_tokenizer = CodeTokenizer(comment_filter, strip_docstring=True)

    print('Preprocessing targets and predictions.')
    refs = preprocess(targets, case_sensitive, tokenizer, code_tokenizer)
    hyps = preprocess(predictions, case_sensitive, tokenizer, code_tokenizer)

    print('Calculating scores.')
    if 'rouge' in metrics:
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=True)

        for rouge_mode, score in scores.items():
            print(f'{rouge_mode}: precision:{score["p"]:.2%}\trecall:{score["r"]:.2%}\tf-1:{score["f"]:.2%}')

    if 'bleu' in metrics:
        smooth = SmoothingFunction()
        bleu = corpus_bleu(refs, hyps, smoothing_function=smooth.method2)
        print(f"Corpus BLEU-4: {bleu}")

    if 'levenshtein' in metrics:
        levenshtein_similarity = calculate_levenshtein_similarity(refs, hyps)
        print(f"Levenshtein similarity: {levenshtein_similarity}")


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate predictions using the given metrics.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--predictions_path', type=str,
                               help='Location of the file containing the predictions. Expects one prediction per line.',
                               required=True)
    requiredNamed.add_argument('--targets_path', type=str,
                               help='Location of the file containing the target. Expects one target per line.',
                               required=True)
    requiredNamed.add_argument('--tokenizer_model', type=str, help='Location of the trained sentencepiece model.',
                               required=True)
    requiredNamed.add_argument('--metrics', type=str, help='List of metrics to calculate. Options are "rouge", "bleu" and "levenshtein".',
                               required=True)

    parser.add_argument('--case_sensitive', action='store_true', default=False, help='Wether to consider case for computing the metric.')
    parser.add_argument('--untokenize', action='store_true', default=False, help='Wether to untokenize the code using the code tokenizer after decoding.')

    args = parser.parse_args()
    run(args.predictions_path, args.targets_path, args.tokenizer_model, args.metrics, args.case_sensitive, args.untokenize)
