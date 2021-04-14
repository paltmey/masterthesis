import os
import sentencepiece as spm
import sys

from argparse import ArgumentParser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.code_tokenizer import INDENT_MARKER, DEDENT_MARKER, BEGIN_COMMENT_MARKER, END_COMMENT_MARKER, END_LINE_MARKER, MISSING_MARKER
from src.comment_classifier import CommentClassifier
from util.file_utils import get_line_count


class CodeBlock:
    def __init__(self):
        self.head = [0, 0]
        self.body = [0, 0]
        self.code_blocks = []


def get_code_blocks(tokens, start_token):
    root = CodeBlock()
    current_block = root

    block_stack = []
    indent_stack = []

    current_ind_level = 0
    in_block_def = False

    for index, token in enumerate(tokens):
        if token == start_token:
            block_stack.append(current_block)
            indent_stack.append(current_ind_level)
            current_ind_level = 0
            current_block = CodeBlock()
            current_block.head[0] = index
            in_block_def = True
        elif token == INDENT_MARKER:
            current_ind_level += 1
        elif token == DEDENT_MARKER:
            current_ind_level -= 1
            if current_ind_level == 0 and len(block_stack) > 0:
                current_block.body[1] = index
                parent_block = block_stack.pop()
                parent_block.code_blocks.append(current_block)
                current_block = parent_block
                current_ind_level = indent_stack.pop()
        elif in_block_def and token == END_LINE_MARKER:
            current_block.head[1] = index
            current_block.body[0] = index+1
            in_block_def = False

    current_block.body[1] = len(tokens)-1

    return root.code_blocks


def is_suitable_comment(comment_tokens, tokenizer, comment_classifier, threshold):
    comment = tokenizer.decode(comment_tokens)
    label, prob = comment_classifier.predict(comment)

    return label == 'positive' and prob >= threshold


def get_comment_indices(tokens, tokenizer, comment_classifier, threshold):
    comment_indices = []

    comment_start = 0
    for index, token in enumerate(tokens):
        if token == BEGIN_COMMENT_MARKER:
            comment_start = index
        if token == END_COMMENT_MARKER:
            comment_end = index+1  # add one, because next token is always EOL
            if comment_classifier:
                comment_tokens = tokens[comment_start+1:index]
                if is_suitable_comment(comment_tokens, tokenizer, comment_classifier, threshold):
                    comment_indices.append(comment_end)
            else:
                comment_indices.append(comment_end)

    return comment_indices


def get_next_line(tokens, start):
    try:
        next_line_end = tokens.index(END_LINE_MARKER, start)
        line = tokens[start:next_line_end+1]  # add one to include EOL
        return line, next_line_end
    except ValueError:
        return None, None


def get_next_n_lines(tokens, start, n):
    lines = []
    for i in range(n):
        line, line_end = get_next_line(tokens, start)
        if line:
            lines.extend(line)
            start = line_end+1

    return lines


def get_context_sequence(tokens, start, end, max_context_length, fold_functions, code_blocks):
    if fold_functions and start > 0:
        current_len = end
        mask_token = -1
        masked_sequence = tokens[:end]
        for code_block in code_blocks:
            if current_len <= max_context_length or code_block.body[1] >= end:  # break when folding is sufficient or block of comment is reached
                break

            body_len = code_block.body[1] - code_block.body[0]
            current_len -= body_len-1  # include offset for fill token
            masked_sequence[code_block.body[0]:code_block.body[1]] = [mask_token]*(body_len-1) + [MISSING_MARKER]  # mask code_block body

        removed_sequence = [token for token in masked_sequence if token != mask_token]
        start = max(0, len(removed_sequence)-max_context_length)
        return removed_sequence[start:]
    else:
        return tokens[start:end]


def get_context_sequences(tokens, comment_indices, code_blocks, max_context_length, num_target_lines, max_target_length, fold_functions):
    context_sequences = []
    targets = []
    for comment_index in comment_indices:
        end = comment_index+1
        next_lines = get_next_n_lines(tokens, start=end, n=num_target_lines)
        valid_target = len(next_lines) > 0

        if max_target_length != -1:
            valid_target = len(next_lines) < max_target_length

        if valid_target:
            targets.append(next_lines)
            start = max(0, end - max_context_length)

            context_sequence = get_context_sequence(tokens, start, end, max_context_length, fold_functions, code_blocks)
            context_sequences.append(context_sequence)

    return context_sequences, targets


def process_file(line, tokenizer, max_context_length, max_target_length, num_target_lines, fold_functions, comment_classifier, threshold):
    tokens = line.split()
    comment_indices = get_comment_indices(tokens, tokenizer, comment_classifier, threshold)
    code_blocks = get_code_blocks(tokens, tokenizer.encode('def', out_type=str)[0])

    return get_context_sequences(tokens, comment_indices, code_blocks, max_context_length, num_target_lines, max_target_length, fold_functions)


def run(input_path, context_output_path, target_output_path, tokenizer_model, max_context_length, max_target_length, num_target_lines, fold_functions=False, comment_classifier_model='', threshold=0.5, files_meta_output_path=''):
    print('Loading tokenizer.')
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model)

    comment_classifier = None
    if comment_classifier_model:
        print('Loading comment classifer.')
        comment_classifier = CommentClassifier(comment_classifier_model)

    print('Reading line count.')
    num_lines = get_line_count(input_path)

    if not os.path.exists(os.path.dirname(context_output_path)):
        os.mkdir(os.path.dirname(context_output_path))
    if not os.path.exists(os.path.dirname(target_output_path)):
        os.mkdir(os.path.dirname(target_output_path))

    files_meta_writer = None
    if files_meta_output_path:
        files_meta_writer = open(files_meta_output_path, 'w')

    print('Creating context sequences.')
    with open(input_path) as file, open(context_output_path, 'w') as context_writer, open(target_output_path, 'w') as target_writer:
        for index,line in enumerate(tqdm(file, total=num_lines)):
            context_sequences, targets = process_file(line, tokenizer, max_context_length, max_target_length, num_target_lines, fold_functions, comment_classifier, threshold)

            if context_sequences and targets:
                for context_sequence, target in zip(context_sequences, targets):
                    context_sequences_str = ' '.join(context_sequence)
                    target_str = ' '.join(target)

                    context_writer.write(context_sequences_str + os.linesep)
                    target_writer.write(target_str + os.linesep)
                    if files_meta_writer:
                        files_meta_writer.write(str(index) + os.linesep)

    if files_meta_writer:
        files_meta_writer.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='Creates context sequences and the corresponding targets.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--input_path', type=str, help='Location of the input token file.',
                               required=True)
    requiredNamed.add_argument('--context_output_path', type=str, help='Location of the context sequences.',
                               required=True)
    requiredNamed.add_argument('--target_output_path', type=str, help='Location of the target sequences.',
                               required=True)
    requiredNamed.add_argument('--tokenizer_model', type=str, help='Location of the trained sentencepiece model.',
                               required=True)

    parser.add_argument('--max_context_length', type=int, default=432, help='Maximum length of the context preceding a target.')
    parser.add_argument('--max_target_length', type=int, default=80, help='Maximum target length.')
    parser.add_argument('--num_target_lines', type=int, default=2, help='Number of lines of each target.')
    parser.add_argument('--fold_functions', action='store_true', default=False, help='If specified, fold functions to fit the context into the maximum length.')
    parser.add_argument('--comment_classifier_model', type=str, default='', help='If specified, uses the given fastText model to classify comments in the input tokens.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Minimum confidence threshold used for the comment classifier.')
    parser.add_argument('--files_meta_output_path', type=str, default='', help='If specified, outputs the file path of all files, which contain context sequences to the given path.')

    args = parser.parse_args()
    run(args.input_path, args.context_output_path, args.target_output_path, args.tokenizer_model, args.max_context_length, args.max_target_length, args.num_target_lines, args.fold_functions, args.comment_classifier_model, args.threshold, args.files_meta_output_path)