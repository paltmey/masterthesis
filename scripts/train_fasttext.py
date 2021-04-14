import csv
import fasttext

from argparse import ArgumentParser
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm


def train(lr, dim, ws, epoch, wordNgrams, input_path, num_folds, num_threads, pretrained_vectors):
    accuracy_sum = 0

    for i in range(num_folds):
        train_filename = f'{input_path}_{i}.train'
        validation_filename = f'{input_path}_{i}.valid'

        if pretrained_vectors:
            model = fasttext.train_supervised(input=train_filename, lr=lr, dim=dim, ws=ws, epoch=epoch, wordNgrams=wordNgrams, pretrainedVectors=pretrained_vectors, verbose=0, thread=num_threads)
        else:
            model = fasttext.train_supervised(input=train_filename, lr=lr, dim=dim, ws=ws, epoch=epoch, wordNgrams=wordNgrams, verbose=0, thread=num_threads)

        accuracy = model.test(validation_filename)[1]
        accuracy_sum += accuracy

    avg_accuracy = accuracy_sum/num_folds

    return model, avg_accuracy


def grid_search(param_grid, input_path, num_folds, num_threads, pretrained_vectors):
    best_avg_accuracy = 0
    best_params = None
    best_model = None

    res = []

    for params in tqdm(ParameterGrid(param_grid)):
        model, avg_accuracy = train(params['lr'], params['dim'], params['ws'], params['epoch'], params['wordNgrams'], input_path, num_folds, num_threads, pretrained_vectors)
        res.append((params, avg_accuracy))

        if avg_accuracy > best_avg_accuracy:
            best_avg_accuracy = avg_accuracy
            best_params = params
            best_model = model

            tqdm.write(f'Best accuracy: {best_avg_accuracy}')

    return res, best_avg_accuracy, best_params, best_model


def save_csv(res, output_path):
    header = ['lr', 'dim', 'ws', 'epoch', 'wordNgrams', 'avg_accuracy']

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for params, avg_accuracy in res:
            row = [params['lr'], params['dim'], params['ws'],params['epoch'], params['wordNgrams'], avg_accuracy]
            writer.writerow(row)


def run(input_path, num_folds, stats_output_path, param_grid, num_threads=1, pretrained_vectors='', best_model_path=''):
    print('Starting grid search.')
    print('Parameters to search:')
    print(param_grid)

    res, best_avg_accuracy, best_params, best_model = grid_search(param_grid, input_path, num_folds, num_threads, pretrained_vectors)

    print(f'Best accuracy: {best_avg_accuracy}')
    print(f'Best params: {best_params}')

    print(f'Saving results to {stats_output_path}')
    save_csv(res, stats_output_path)

    if best_model_path:
        print(f'Saving best model to {best_model_path}')
        best_model.save_model(best_model_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a supervised fasttext classifier using grid search.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--input_path', type=str, help='Base name of the training and validation split files. Format should be "input_path_(fold).(train|validation).',
                               required=True)
    requiredNamed.add_argument('--num_folds', type=int, help='Number of folds to average over.', required=True)
    requiredNamed.add_argument('--stats_output_path', type=str, help='Location of the output csv file containing parameters and corresponding results.', required=True)

    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads to use for training.')
    parser.add_argument('--pretrained_vectors', type=str, default='', help='Pretrained vectors file that can be used for training.')
    parser.add_argument('--best_model_path', type=str, default='', help='If specified, saves the best model to the given path.')

    #Hypterparameters
    parser.add_argument('--lr', type=str, default='0.1, 0.25, 0.5, 0.75, 1.', help='Learning rate values to sweep.')
    parser.add_argument('--dim', type=str, default='100, 300', help='Dimension values to sweep.')
    parser.add_argument('--ws', type=str, default='1, 2, 3, 4, 5, 8, 15', help='Window sizes to sweep.')
    parser.add_argument('--epoch', type=str, default='5, 10, 25, 50', help='Number of train epochs to sweep.')
    parser.add_argument('--word_ngrams', type=str, default='1, 2, 3, 4, 5', help='Word Ngrams to sweep.')

    args = parser.parse_args()

    param_grid = {
        'lr': [float(item) for item in args.lr.split(',')],
        'dim': [int(item) for item in args.dim.split(',')],
        'ws': [int(item) for item in args.ws.split(',')],
        'epoch': [int(item) for item in args.epoch.split(',')],
        'wordNgrams': [int(item) for item in args.word_ngrams.split(',')]
    }

    run(args.input_path, args.num_folds, args.stats_output_path, param_grid, args.num_threads, args.pretrained_vectors, args.best_model_path)
