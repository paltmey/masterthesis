import csv
import jsonlines
import json
import random

from argparse import ArgumentParser

SEED = 'RANDOMSEED'


def get_repo_infos(input_path):
    repos = {}
    file_hashes = set()

    total_files = 0
    total_lines = 0
    total_comments = 0
    duplicates = 0

    with jsonlines.open(input_path) as reader:
        for item in reader:
            if item['hash'] in file_hashes:
                duplicates += 1
                continue

            file_hashes.add(int(item['hash']))

            repo_name = item['repo']
            repo = repos.get(item['repo'], {'files': [], 'num_lines': 0, 'num_comments': 0})
            repo['files'].append(item['file_path'])
            repo['num_lines'] += item['num_lines']
            repo['num_comments'] += len(item['positions'])
            repos[repo_name] = repo

            total_files += 1
            total_lines += item['num_lines']
            total_comments += len(item['positions'])

    return repos, total_files, total_lines, duplicates


def get_repos_with_max_lines(repos, max_lines):
    res = []

    num_lines = 0
    for repo in repos:
        if num_lines > max_lines:
            break
        res.append(repo)
        num_lines += repo['num_lines']

    return res


def get_stats(splits, excludes, test, train, validation, duplicates):
    exlude_files = len(splits['excludes'])
    exlude_lines = sum(repo['num_lines'] for repo in excludes)
    test_files = len(splits['test'])
    test_lines = sum(repo['num_lines'] for repo in test)
    train_files = len(splits['train'])
    train_lines = sum(repo['num_lines'] for repo in train)
    validation_files = len(splits['validation'])
    validation_lines = sum(repo['num_lines'] for repo in validation)

    stats = [
        {'split': 'Excludes', 'files': exlude_files, 'lines': exlude_lines, 'duplicates': ''},
        {'split': 'Test', 'files': test_files, 'lines': test_lines, 'duplicates': ''},
        {'split': 'Train', 'files': train_files, 'lines': train_lines, 'duplicates': ''},
        {'split': 'Validation', 'files': validation_files, 'lines': validation_lines, 'duplicates': ''},
        {'split': 'Total',
         'files': exlude_files + test_files + train_files + validation_files,
         'lines': exlude_lines + test_lines + train_lines + validation_lines,
         'duplicates': duplicates
         }
    ]
    return stats


def save_csv(stats, output_path):
    header = ['split', 'files', 'lines', 'duplicates']

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for split_stats in stats:
            row = [split_stats['split'], split_stats['files'], split_stats['lines'], split_stats['duplicates']]
            writer.writerow(row)


def run(input_path, output_path, train_validation_split, test_percentage, exclude_percentage, stats_output_path=''):
    repos, total_files, total_lines, duplicates = get_repo_infos(input_path)

    num_exclude = int(exclude_percentage * total_lines)
    num_test = int(test_percentage * (total_lines - num_exclude))
    num_train = int(train_validation_split * (total_lines - num_exclude - num_test))

    rand = random.Random(SEED)

    repos_list = list(repos.values())
    rand.shuffle(repos_list)

    excludes = get_repos_with_max_lines(repos_list, num_exclude)
    repos_list = repos_list[len(excludes):]
    test = get_repos_with_max_lines(repos_list, num_test)
    repos_list = repos_list[len(test):]
    train = get_repos_with_max_lines(repos_list, num_train)
    validation = repos_list[len(train):]

    splits = {
        'excludes': [file for repo in excludes for file in repo['files']],
        'test': [file for repo in test for file in repo['files']],
        'train': [file for repo in train for file in repo['files']],
        'validation': [file for repo in validation for file in repo['files']]
    }

    with open(output_path, 'w') as file:
        json.dump(splits, file)

    if stats_output_path:
        stats = get_stats(splits, excludes, test, train, validation, duplicates)
        save_csv(stats, stats_output_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='Create dataset splits with the given split percentages.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--input_path', type=str, help='Location of the file containing the file metadata.',
                               required=True)
    requiredNamed.add_argument('--output_path', type=str, help='Location of the file containing the file paths for each split. Outputs in JSON format.', required=True)

    parser.add_argument('--train_validation_split', type=float, default=0.8, help='Percentage of the train validation split.')
    parser.add_argument('--test_percentage', type=float, default=0.2, help='Percentage of test data to put aside.')
    parser.add_argument('--exclude_percentage', type=float, default=0.2, help='Percentage of excluded data to put aside.')
    parser.add_argument('--stats_output_path', type=str, default='', help='If specified, saves the stats for each split to the given path. Outputs in csv format.')

    args = parser.parse_args()

    run(args.input_path, args.output_path, args.train_validation_split, args.test_percentage, args.exclude_percentage, args.stats_output_path)
