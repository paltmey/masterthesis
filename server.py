from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin

import sentencepiece as spm
from transformers import AutoModelForCausalLM
from src.code_generator import CodeGenerator

MODEL_PATH = 'model/finetuned/simple_truncation'
TOKENIZER_MODEL_PATH = 'model/sentencepiece/sentencepiece.model'
FOLD_FUNCTIONS = False

GENERATE_ARGS = {
    'min_length': 0,
    'max_length': 512,
    'do_sample': True,
    'num_beams': 1,
    'num_return_sequences': 3,
    'temperature': 1.,
    'top_k': 0,
    'top_p': 0.7,
    'no_repeat_ngram_size': 0.,
    'repetition_penalty': 1.,
    'length_penalty': 1.,
    'bad_words_ids': [[9]]
}


def create_app():
    # create and configure the app
    app = Flask(__name__, template_folder='frontend/dist')

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)
    code_generator = CodeGenerator(model, tokenizer)

    def get_predictions(content):
        generated_sequences = code_generator.generate(content, GENERATE_ARGS, add_line_markers=True, format=True, fold_functions=FOLD_FUNCTIONS)
        return {
            'suggestions': [
                {
                    'label': generated_code,
                    'kind': 'Text',
                    'insertText': generated_code
                }
                for generated_code in generated_sequences
            ]
        }

    @app.route('/completion', methods=['POST'])
    @cross_origin()
    def code_completion():
        data = request.get_json()
        content = data['content']

        if content:
            result = get_predictions(content)
            return jsonify(result)
        else:
            return jsonify({'error':'no content received'}), 400

    @app.route("/")
    def index():
        return render_template('index.html')

    return app


app = create_app()
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

if __name__ == '__main__':
   app.run()
