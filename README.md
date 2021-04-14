## Project Structure
The source code of this project is structured into a ``/scripts`` directory, a ``/src`` directory, a ``/util`` directory and a ``/frontend`` directory. The  ``/scripts`` directory contains all scripts to preprocess the data and train the models. The ``/src`` directory contains methods and classes, which are used by multiple scripts and the ``/util`` directory contains helper functions. The ``/frontend`` directory contains the JavaScript source code for the frontend application. In addition, the ``/model`` directory contains all trained models and the ``/data`` directory contains all raw and preprocessed data. A Python server that serves the Demo Application is implemented in ``server.py``.

## Pretrained Models
Pretrained models can be downloaded [here](https://drive.google.com/file/d/1FbiFlcuI8h5CxJ4z5f2sjc60BgiaBEcY/view?usp=sharing). Unzip the folder and copy the entire structure to the ``/model`` directory.

## Setup
The code implementation is Python 3 only. The necessary dependecies for this project can be installed with the following command:
```
pip install -r requirements.txt
```

## Preprocessing and Training Steps
To generate the training data and train the models, several steps are required:
1. Dataset crawling
    1. Scrape a list of most starred repositories from GitHub
    2. Download the repositories
2. File Preprocessing
    1. Get list of file paths for all source code files
    2. Find comments in files
    3. Classify comments using a comment classifier implemented as a fastText model
    4. Copy all suitable files with at least one relevant comment to a new directory
    5. Format copied files using the black Python code formatter
    6. Create tokenized dataset using the code tokenizer
    7. Train sentencepiece model
    7. Create dataset splits
    8. Create subword tokenized dataset for each split using a sentencepiece model
    9. Create pre-training id dataset of tensors using the subword tokenized files
3. Training and Evaluation
    1. Pre-Train the GPT-2 model
    2. Create evaluation sequences
    3. Create predictions using the trained model
    4. Evaluate the predictions
4. Finetuning
    1. Create context and target sequences from the subword tokenized files
    2. Create Finetuning id dataset that transforms the context sequences to tensors
    3. Finetune the pre-trained model
    4. Create predictions using the finetuned model
    5. Evaluate the predictions

Additional steps are required to train and evaluate the fastText classifier:
1. Preprocess comments for fastText
2. Train unsupervised fastText word embeddings
3. Train supervised fastText classifier using grid search
4. Evaluate fastText classifier

Each preprocessing and training step has an associated script.

## Scripts
### 1. Dataset crawling

The ``fetch_github_top_repos.py`` script is used to fetch the most starred repos from GitHub. The only required argument is the path of the output file. Optional arguments are the min and max number of stars to consider, a list of access tokens and an append flag. Using the append flag, the script continues operation when the outfile already exists. 
```
positional arguments:
  outfile

optional arguments:
  --min_stars MIN_STARS
  --max_stars MAX_STARS
  --access_tokens [ACCESS_TOKENS ...]
  --append
```

The ``download_repos.py`` script takes the list of repos from the previous script and downloads the files to the given output directory.
```
positional arguments:
  infile

optional arguments:
  --output_dir OUTPUT_DIR
  --file_whitelist [FILE_WHITELIST ...]
  --access_tokens [ACCESS_TOKENS ...]
  --num_processes NUM_PROCESSES
```

### 2. File Preprocessing
The ``get_file_paths.py`` script creates a list of all files below the start\_dir. The file\_whitelist can be used to define a shell pattern of files to include.
```
optional arguments:
  --file_whitelist FILE_WHITELIST
                        Shell Pattern to find matching files

required named arguments:
  --start_dir START_DIR
                        Location of the directory containing the files to
                        find.
  --output_path OUTPUT_PATH
                        Location of the output file containing the file paths
```

The ``find_comments.py`` script takes the list of path location and returns a JSON line file containing all comment positions.
```
optional arguments:
  --line_output_path LINE_OUTPUT_PATH
                        If specified, outputs all comments as plain text to
                        the given location. Outputs in txt format.
  --num_processes NUM_PROCESSES
                        Number of processes to spin up.

required named arguments:
  --file_paths_file FILE_PATHS_FILE
                        Location of the file containing the file paths.
  --base_path BASE_PATH
                        Location of the directory containing the repositories.
  --output_path OUTPUT_PATH
                        Location of the output file containing the comment
                        positions per input file. Outputs in JSON lines
                        format.
```

The ``classify_comments.py`` script takes in the comment positions in the JSON lines format from the previous script and classifies all comments using the given comment classifier model.
```
optional arguments:
  --num_processes NUM_PROCESSES
                        Number of processes to spin up.

required named arguments:
  --input_path INPUT_PATH
                        Location of the file containing the comments of each
                        source code file.
  --output_path OUTPUT_PATH
                        Location of the output file containing the comment
                        positions and classification predicftions per input
                        file. Outputs in JSON lines format.
  --comment_classifier_model COMMENT_CLASSIFIER_MODEL
                        Location of the trained fastText model.
```

The ``copy_suitable_comments.py`` script takes the JSON lines file containing the classified comments and copies all files, which contain at least one comment with a predicted score higher than the threshold to the output directory. If the metadata\_output\_path is given, a file containing the new file locations is created.
```
optional arguments:
  --num_processes NUM_PROCESSES
                        Number of processes to spin up.

required named arguments:
  --input_path INPUT_PATH
                        Location of the file containing the classified
                        comments metadata.
  --output_dir OUTPUT_DIR
                        Location of the directory, under which all files with
                        suitable comments should be copied.
  --metadata_output_path METADATA_OUTPUT_PATH
                        Location of the output file containing the metadata of
                        all files containing suitable comments. Outputs in
                        JSON lines format.
  --threshold THRESHOLD
                        Minimum confidence threshold to consider a positive
                        comment as suitable.
```

The ``format_files.py`` script uses the black code formatter to format all files under the given directory.
```
optional arguments:
  --fast             If --fast given, skip temporary sanity checks.

required named arguments:
  --src_dir SRC_DIR  Directory where to run formatting.
```

The ``code_tokenize_files.py`` script tokenizes the files using the Python tokenize module. The input path is the location of the file created by the metadata\_output\_path option of the ``copy_suitable_comments.py`` script.
```
  --num_processes NUM_PROCESSES
                        Number of processes to spin up.

required named arguments:
  --input_path INPUT_PATH
                        Location of the file containing the metadata of files
                        with suitable comments.
  --output_dir OUTPUT_DIR
                        Location of the directory, under which all tokenized
                        files should be saved.
  --metadata_output_path METADATA_OUTPUT_PATH
                        Location of the output file containing the metadata,
                        including the calculated hash, of all tokenized files.
                        Outputs in JSON lines format.
```

The ``train_sentencepiece.py`` script trains a sentencepiece model using all file paths included in the metadata file. The hyperparameter arguments are set to the default values used to train the included model.
```
optional arguments:
  --vocab_size VOCAB_SIZE
                        Vocabulary size.
  --character_coverage CHARACTER_COVERAGE
                        Amount of characters covered by the model.
  --input_sentence_size INPUT_SENTENCE_SIZE
                        Maximum samples the trainer loads.
  --num_reserved_markers NUM_RESERVED_MARKERS
                        If specified, adds the given number of reserved
                        markers <RES#> to the user_defined_symbols.
  --num_threads NUM_THREADS
                        Number of threads to use for training.

required named arguments:
  --input_path INPUT_PATH
                        Location of the file containing the file metadata.
  --output_path OUTPUT_PATH
                        Location of the trained model and vocab. Creates two
                        files "output_path".model and "output_path".vocab.
```

The ``create_dataset_splits.py`` script is used to create a JSON file containing the file paths of the respective splits. The default split percentages are set to the values used in the thesis.
```
optional arguments:
  --train_validation_split TRAIN_VALIDATION_SPLIT
                        Percentage of the train validation split.
  --test_percentage TEST_PERCENTAGE
                        Percentage of test data to put aside.
  --exclude_percentage EXCLUDE_PERCENTAGE
                        Percentage of excluded data to put aside.
  --stats_output_path STATS_OUTPUT_PATH
                        If specified, saves the stats for each split to the
                        given path. Outputs in csv format.

required named arguments:
  --input_path INPUT_PATH
                        Location of the file containing the file metadata.
  --output_path OUTPUT_PATH
                        Location of the file containing the file paths for
                        each split. Outputs in JSON format.
```

The ``create_tokenized_dataset.py`` script uses the trained sentencepiece model to subword tokenize the files of the given split. The resulting dataset is a single file containing the tokens of all files, where each file corresponds to one line. The output file is named ``{split}_tokens.txt``, where the {split} is replaced with the name of the given split.
```
optional arguments:
  --num_processes NUM_PROCESSES
                        Number of processes to spin up.

required named arguments:
  --splits_path SPLITS_PATH
                        Location of the file containing the dataset splits.
  --output_dir OUTPUT_DIR
                        Directory where the tokenized dataset should be saved.
                        Each line in the output file corresponds to one source
                        code file.
  --split SPLIT         Name of the split that should be processed.
  --tokenizer_model TOKENIZER_MODEL
                        Location of the trained sentencepiece model.
```

The ``create_id_dataset.py`` script transforms the subword tokenized dataset from the previous step into pytorch tensors.
```
required named arguments:
  --input_dir INPUT_DIR
                        Location of the file containing the tokenized code
                        files.
  --split SPLIT         Name of the split that should be processed.
  --output_dir OUTPUT_DIR
                        Location of the output dir containing the id dataset.
                        Each input line is transformed to ids and saved as a
                        pytorch tensor.
  --tokenizer_model TOKENIZER_MODEL
                        Location of the trained sentencepiece model.
```

### 3. Training and Evaluation
The ``train_scratch.py`` uses the [Hugging Face Transformer](https://huggingface.co/transformers/) library to train a GPT-2 model. A locally setup [Weights & Biases](https://wandb.ai) account is required.
```
optional arguments:
  --vocab_size VOCAB_SIZE
                        The vocab size of the model embedding layer.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --weight_decay WEIGHT_DECAY
                        Weight decay if we apply some.
  --lr_scheduler LR_SCHEDULER
                        Learning rate scheduler used for training.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --steps_per_epoch STEPS_PER_EPOCH
                        Number of steps per training epoch.
  --lr_decay LR_DECAY   Learning rate decay per epoch.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_steps MAX_STEPS
                        The total number of training steps to perform.
                        Overrides --num_train_epochs
  --train_batch_size TRAIN_BATCH_SIZE
                        Number of batches used in training dataloader.
  --validation_batch_size VALIDATION_BATCH_SIZE
                        Number of batches used in training dataloader.
  --block_size BLOCK_SIZE
                        Number of tokens joined to a single feature.
  --dataloader_drop_last DATALOADER_DROP_LAST
                        Whether to drop the last incomplete batch.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate the gradients
                        for, before performing a backward/update pass.
  --fp16 FP16           Whether to use 16-bit (mixed) precision training.
  --logging_steps LOGGING_STEPS
                        Number of update steps between two logs.
  --eval_steps EVAL_STEPS
                        Number of update steps between two evaluations.
  --save_steps SAVE_STEPS
                        Number of updates steps before two checkpoint saves.
  --limit_train_batches LIMIT_TRAIN_BATCHES
                        Limits the percentage of train batches used.
  --limit_validation_batches LIMIT_VALIDATION_BATCHES
                        Limits the percentage of validation batches checked.
  --checkpoint_dir CHECKPOINT_DIR
                        Continue Training from Checkpoint.
  --init_epochs INIT_EPOCHS
  --overwrite_output_dir OVERWRITE_OUTPUT_DIR
                        Whether to overwrite the content of the output
                        directory.

required named arguments:
  --train_dataset_path TRAIN_DATASET_PATH
                        Location of the training dataset.
  --validation_dataset_path VALIDATION_DATASET_PATH
                        Location of the training dataset.
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
```

The ``create_context_sequences.py`` script creates context sequences from the subword tokenized dataset file. 
```
optional arguments:
  --max_context_length MAX_CONTEXT_LENGTH
                        Maximum length of the context preceding a target.
  --max_target_length MAX_TARGET_LENGTH
                        Maximum target length.
  --num_target_lines NUM_TARGET_LINES
                        Number of lines of each target.
  --fold_functions      If specified, fold functions to fit the context into
                        the maximum length.
  --comment_classifier_model COMMENT_CLASSIFIER_MODEL
                        If specified, uses the given fastText model to
                        classify comments in the input tokens.
  --threshold THRESHOLD
                        Minimum confidence threshold used for the comment
                        classifier.
  --files_meta_output_path FILES_META_OUTPUT_PATH
                        If specified, outputs the file path of all files,
                        which contain context sequences to the given path.

required named arguments:
  --input_path INPUT_PATH
                        Location of the input token file.
  --context_output_path CONTEXT_OUTPUT_PATH
                        Location of the context sequences.
  --target_output_path TARGET_OUTPUT_PATH
                        Location of the target sequences.
  --tokenizer_model TOKENIZER_MODEL
                        Location of the trained sentencepiece model.
```

The ``create_predicions.py`` script is used to create predictions from the given context file.
```
optional arguments:
  --limit_input LIMIT_INPUT
                        Limits the of input data to the given number of
                        examples.
  --min_length MIN_LENGTH
                        The minimum length of the sequence to be generated.
  --max_length MAX_LENGTH
                        The maximum length of the sequence to be generated.
  --sample              Whether or not to use sampling, use greedy decoding
                        otherwise.
  --num_beams NUM_BEAMS
                        Number of beams for beam search. 1 means no beam
                        search.
  --num_return_sequences NUM_RETURN_SEQUENCES
                        The number of independently computed returned
                        sequences for each element in the batch.
  --temperature TEMPERATURE
                        The value used to module the next token probabilities.
  --top_k TOP_K         The number of highest probability vocabulary tokens to
                        keep for top-k-filtering.
  --top_p TOP_P         If set to float < 1, only the most probable tokens
                        with probabilities that add up to top_p or higher are
                        kept for generation.

required named arguments:
  --input_path INPUT_PATH
                        Location of the file containing the context file.
  --output_path OUTPUT_PATH
                        Location of the output file containing the comment
                        predictions. Outputs in txt format.
  --model MODEL         Location of the trained transformer model.
  --tokenizer_model TOKENIZER_MODEL
                        Location of the trained sentencepiece model.
```

The ``evaluate_predicions.py`` script is used to evaluate the created predictions.  BLEU, ROUGE-L and Levenshtein similarity scores can be measured.
```
optional arguments:
  --case_sensitive      Wether to consider case for computing the metric.
  --untokenize          Wether to untokenize the code using the code tokenizer
                        after decoding.

required named arguments:
  --predictions_path PREDICTIONS_PATH
                        Location of the file containing the predictions.
                        Expects one prediction per line.
  --targets_path TARGETS_PATH
                        Location of the file containing the target. Expects
                        one target per line.
  --tokenizer_model TOKENIZER_MODEL
                        Location of the trained sentencepiece model.
  --metrics METRICS     List of metrics to calculate. Options are "rouge",
                        "bleu" and "levenshtein".
```

### 4. Finetuning
The ``create_context_sequences.py`` script is again used to create the fine tuning sequences.

The ``create_finetuning_dataset.py`` script is used to create an id dataset of pytorch tensors from the given context and target files.
```
required named arguments:
  --context_input_path CONTEXT_INPUT_PATH
                        Location of the file containing the context sequences.
  --target_input_path TARGET_INPUT_PATH
                        Location of the file containing the target sequences.
  --output_path OUTPUT_PATH
                        Location of the output file containing the id dataset.
                        Each input line is transformed to ids and saved as a
                        pytorch tensor.
  --tokenizer_model TOKENIZER_MODEL
                        Location of the trained sentencepiece model.
```

The ``finetuning_scratch.py`` script is used to finetune the pre-trained model. The optional parameters are similar those of the ``training_scratch.py``. Note that a wandb account is required.
```
  --vocab_size VOCAB_SIZE
                        The vocab size of the model embedding layer.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --weight_decay WEIGHT_DECAY
                        Weight decay if we apply some.
  --lr_scheduler LR_SCHEDULER
                        Learning rate scheduler used for training.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --steps_per_epoch STEPS_PER_EPOCH
                        Number of steps per training epoch.
  --lr_decay LR_DECAY   Learning rate decay per epoch.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_steps MAX_STEPS
                        The total number of training steps to perform.
                        Overrides --num_train_epochs
  --train_batch_size TRAIN_BATCH_SIZE
                        Number of batches used in training dataloader.
  --validation_batch_size VALIDATION_BATCH_SIZE
                        Number of batches used in training dataloader.
  --block_size BLOCK_SIZE
                        Number of tokens joined to a single feature.
  --dataloader_drop_last DATALOADER_DROP_LAST
                        Whether to drop the last incomplete batch.
  --dataloader_num_workers DATALOADER_NUM_WORKERS
                        Number of subprocesses to use for data loading.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate the gradients
                        for, before performing a backward/update pass.
  --fp16 FP16           Whether to use 16-bit (mixed) precision training.
  --logging_steps LOGGING_STEPS
                        Number of update steps between two logs.
  --eval_steps EVAL_STEPS
                        Number of update steps between two evaluations.
  --save_steps SAVE_STEPS
                        Number of updates steps before two checkpoint saves.
  --limit_train_batches LIMIT_TRAIN_BATCHES
                        Limits the percentage of train batches used.
  --limit_validation_batches LIMIT_VALIDATION_BATCHES
                        Limits the percentage of validation batches checked.
  --pad_token_id PAD_TOKEN_ID
                        Id of the padding token to use.
  --overwrite_output_dir OVERWRITE_OUTPUT_DIR
                        Whether to overwrite the content of the output
                        directory.

required named arguments:
  --train_dataset_path TRAIN_DATASET_PATH
                        Location of the training dataset.
  --validation_dataset_path VALIDATION_DATASET_PATH
                        Location of the training dataset.
  --pretrained_model_dir PRETRAINED_MODEL_DIR
                        Location of the pretrained model.
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
```

The ``create_predictions.py`` script is used again to create the predictions of the finetuned model given the input contexts.

The ``evaluate_predicions.py`` script is used again to evaluate the predictions of the finetuned model.

### FastText classifier
The ``preprocess_comments_for_fasttext.py`` script preprocesses all comments in the given file that contains the comment positions, which was previously created using the ``find_comments.py`` script.
```
required named arguments:
  --input_path INPUT_PATH
                        Location of the file containing the comment positions.
  --output_path OUTPUT_PATH
                        Location of the processed comments.
```

The ``create_fasttext_vectors.py`` script trains an unsupervised fastText embedding on the preprocessed comments file created in the previous step.
```
optional arguments:
  --num_threads NUM_THREADS
                        Number of threads to use.
  --dim DIM             Size of word vectors.

required named arguments:
  --input_path INPUT_PATH
                        Location of the file containing the preprocessed
                        comments.
  --output_path OUTPUT_PATH
                        Location of the trained fasttext vectores.
```

The ``train_fasttext.py`` script train a supervised fastText model on labeled comments. The pre-trained vectors can be given as an optional argument. To train the model, a number of folds of labeled comments are necessary. The labeled comment dataset used in the thesis is included in the ``/data/annotated_comments`` directory. A grid search approach is used to train and find the best model.
```
optional arguments:
  --num_threads NUM_THREADS
                        Number of threads to use for training.
  --pretrained_vectors PRETRAINED_VECTORS
                        Pretrained vectors file that can be used for training.
  --best_model_path BEST_MODEL_PATH
                        If specified, saves the best model to the given path.
  --lr LR               Learning rate values to sweep.
  --dim DIM             Dimension values to sweep.
  --ws WS               Window sizes to sweep.
  --epoch EPOCH         Number of train epochs to sweep.
  --word_ngrams WORD_NGRAMS
                        Word Ngrams to sweep.

required named arguments:
  --input_path INPUT_PATH
                        Base name of the training and validation split files.
                        Format should be
                        "input_path_(fold).(train|validation).
  --num_folds NUM_FOLDS
                        Number of folds to average over.
  --stats_output_path STATS_OUTPUT_PATH
                        Location of the output csv file containing parameters
                        and corresponding results.
```

The ``evaluate_fasttext.py`` script is used to evaluate the fastText model. A validation set of labeled comments is required. The dataset used in the thesis is also included in in the ``/data/annotated_comments`` directory.
```
optional arguments:
  --prediction_output_path PREDICTION_OUTPUT_PATH
                        If specified, output the predictions to the given
                        path. Outputs in csv format.
  --print_metrics       Wether to print metrics (Precision, Recall, F-1).
  --roc_output_path ROC_OUTPUT_PATH
                        If specified, outputs ROC curve as PDF to the given
                        path.

required named arguments:
  --comment_classifier_model COMMENT_CLASSIFIER_MODEL
                        Location of the trained fastText model.
  --validation_data_path VALIDATION_DATA_PATH
                        Location of the file containing the labeled validation
                        data.
```

## Demo Application
The demo application consists of a web app that embeds a [Microsoft Monaco Editor](https://microsoft.github.io/monaco-editor), similar to VS Code and a Python server that uses a trained transformer model to generate code completions. The JavaScript frontend is located under the ``/frontend`` folder. The server code is implemented in the ``server.py`` file. To start the server, the ``server.py`` can be run as a normal python script. The default address of the server is http://localhost:5000. The networking code is implemented using [Flask](https://palletsprojects.com/p/flask/), which includes a development server. For a production deployment, a real WSGI capable server is required.

To customize the models or parameters used by the server, the constants within the source code have to be changed. Note that the ``FOLD_FUNCTIONS`` option has to be set to ``True`` in order for the finetuned function folding model to operate correctly. The pre-trained model and both finetuned models are included under ``model/pretrained``, ``model/fintuned/simple_truncation`` and ``model/fintuned/function_folding``.

The already compiled frontend code is located under ``/frontend/dist``. It consists of an ``index.html`` entrypoint and JavaScript bundles which load the Editor. To build the frontend from source, a Node.js environment with npm or yarn installed is required. The dependecies can be installed within the ``/frontend`` directory with the following commands:
```
cd frontend
npm install .
```
To build the frontend, the build script has to be executed using:
```
npm run build
```
The compiled fronted is saved under ``/frontend/dist``.
