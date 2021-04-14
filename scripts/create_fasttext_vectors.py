import fasttext
from argparse import ArgumentParser


def save_vectors(model, filename):
    # get all words from model
    words = model.get_words()

    with open(filename, 'w') as file_out:

        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + ' ' + str(model.get_dimension()) + '\n')

        # line by line, you append vectors to VEC file
        for w in words:
            v = model.get_word_vector(w)
            vstr = ''
            for vi in v:
                vstr += ' ' + str(vi)
            try:
                file_out.write(w + vstr + '\n')
            except:
                pass


def run(input_path, output_path, dim):
    print('Creating fastText vectors.')
    model = fasttext.train_unsupervised(input_path, dim=dim)
    save_vectors(model, output_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='Trains an unsupervised fasText model and saves the trained embeddings as vectors.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--input_path', type=str, help='Location of the file containing the preprocessed comments.', required=True)
    requiredNamed.add_argument('--output_path', type=str, help='Location of the trained fasttext vectores.', required=True)

    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads to use.')

    #Hypterparameters
    parser.add_argument('--dim', type=int, default=100, help='Size of word vectors.')

    args = parser.parse_args()
    run(args.input_path, args.output_path, args.dim)
