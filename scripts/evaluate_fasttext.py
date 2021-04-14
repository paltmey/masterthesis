import csv
import fasttext
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from sklearn.metrics import auc, roc_curve

sns.set()

def read_fasttext_file(filename):
    with open(filename) as file:
        lines = file.readlines()

    data = [line[:-1].split(' ', maxsplit=1) for line in lines]
    return data


def print_metrics(predictions):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for _, label, predicted_label, prob in predictions:
        if label == '__label__positive':
            if predicted_label == label:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if predicted_label == label:
                true_negative += 1
            else:
                false_positive += 1

    total = len(predictions)
    accuracy = (true_positive+true_negative)/total
    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive+false_negative)
    f_1 = 2 * (precision * recall) / (precision + recall)

    print(f'Total predictions:\t{total}')
    print(f'True positives:\t{true_positive}')
    print(f'True negative:\t{true_negative}')
    print(f'False positives:\t{false_positive}')
    print(f'False negatives:\t{false_negative}')
    print(f'Accuracy:\t{accuracy}')
    print(f'Precision:\t{precision}')
    print(f'Recall:\t{recall}')
    print(f'F1 score:\t{f_1}')


def output_roc(predictions, output_path):
    y_true = []
    y_score = []

    for _, label, predicted_label, prob in predictions:
        if label == '__label__positive':
            y_true.append(1)
        else:
            y_true.append(0)

        if predicted_label == '__label__positive':
            y_score.append(prob)
        else:
            y_score.append(1 - prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    f = plt.figure(figsize=[7.4, 4.8])

    lw = 2
    plt.plot(fpr, tpr, color='C1',
             lw=lw, label='FastText Classsifier (AUC = %0.2f)' % roc_auc)
    plt.fill_between(fpr, tpr, color='C1', alpha=0.1)
    plt.plot([0, 1], [0, 1], color='C0', lw=lw, linestyle='--', label='Baseline')
    plt.stem([fpr[40]], [tpr[40]], linefmt='k--', markerfmt='ko')
    plt.annotate('t = 0.75', (fpr[40], tpr[40]), textcoords='offset points', xytext=(10, -0.5), va='center')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    f.savefig(output_path, bbox_inches='tight')


def save_predictions(predictions, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['entry', 'label', 'predicted_label', 'correct', 'probability'])

        for entry, label, pred_label, prob in predictions:
            correct = 'true' if label == pred_label else 'false'
            writer.writerow([entry, label[9:], pred_label[9:], correct, f'{prob:.4f}'])


def run(comment_classifier_model, validation_data_path, prediction_output_path, output_metrics, roc_output_path):
    model = fasttext.load_model(comment_classifier_model)
    validation_data = read_fasttext_file(validation_data_path)

    predictions = []
    for label, entry in validation_data:
        pred_labels, probs = model.predict(entry)
        predictions.append((entry, label, pred_labels[0], probs[0]))

    if prediction_output_path:
        save_predictions(predictions, prediction_output_path)

    if output_metrics:
        print_metrics(predictions)

    if roc_output_path:
        output_roc(predictions, roc_output_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate the fasttext comment classifier.')
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('--comment_classifier_model', type=str, help='Location of the trained fastText model.',
                               required=True)
    requiredNamed.add_argument('--validation_data_path', type=str, help='Location of the file containing the labeled validation data.',
                               required=True)

    parser.add_argument('--prediction_output_path', type=str, default='', help='If specified, output the predictions to the given path. Outputs in csv format.')
    parser.add_argument('--print_metrics', action='store_true', default=False, help='Wether to print metrics (Precision, Recall, F-1).')
    parser.add_argument('--roc_output_path', type=str, default='', help='If specified, outputs ROC curve as PDF to the given path.')

    args = parser.parse_args()
    run(args.comment_classifier_model, args.validation_data_path, args.prediction_output_path, args.print_metrics, args.roc_output_path)
