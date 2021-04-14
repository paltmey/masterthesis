import fasttext
import nltk
import re


class CommentClassifier:
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)

    @staticmethod
    def _preprocess(text):
        lower_text = text.lower()
        sentences = nltk.sent_tokenize(lower_text)
        tokens = []
        for sentence in sentences:
            cleaned_sentence = re.sub(r'[.()\[\]`"\']', '', sentence)
            tokens.extend(nltk.word_tokenize(cleaned_sentence))

        return ' '.join(tokens)

    def predict(self, comment, preprocess=True):
        if preprocess:
            comment = self._preprocess(comment)
        pred_labels, probs = self.model.predict(comment)
        return pred_labels[0][9:], probs[0]  # cutoff '__label__'
