import logging
import os
import torch
import transformers
import wandb

from argparse import ArgumentParser
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, GPT2LMHeadModel, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import is_main_process

logger = logging.getLogger(__name__)

LABEL_PAD_ID = -100


class DataCollatorWithPadding:
    def __init__(self, pad_token_id, label_pad_id, pad_left=True):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id
        self.pad_left = pad_left

    def _pad(self, batch, pad_token_id):
        max_length = max([value.size(0) for value in batch])
        padded_batch = torch.full(size=(len(batch), max_length), fill_value=pad_token_id)

        for i, value in enumerate(batch):
            if self.pad_left:
                padded_batch[i][max_length-len(value):] = value
            else:
                padded_batch[i][:len(value)] = value

        return padded_batch

    def __call__(self, features):
        batch = {}
        input_ids = [f['input_ids'] for f in features]
        labels = [f['labels'] for f in features]
        attention_mask = [torch.ones(item.size()) for item in input_ids]

        batch['input_ids'] = self._pad(input_ids, self.pad_token_id)
        batch['labels'] = self._pad(labels, self.label_pad_id)
        batch['attention_mask'] = self._pad(attention_mask, 0)

        return batch


class FinetuneDataset(Dataset):
    def __init__(self, file_path, block_size, limit=1.):
        self.file_path = file_path
        self.block_size = block_size
        self.limit = limit
        self.data = self._load_data(file_path, limit)

    def _load_data(self, file_path, limit):
        data = torch.load(file_path)
        limit_index = int(len(data)*limit)
        limited_data = data[:limit_index]
        return limited_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, context_len = self.data[idx]
        input_ids = input_ids.long()
        labels = input_ids.clone().detach()
        labels[:context_len] = LABEL_PAD_ID  # -100 is default loss ignore_index

        return {'input_ids': input_ids, 'labels': labels}


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, lr_decay, steps_per_epoch, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0., lr_decay**(max(0, (current_step-num_warmup_steps)/steps_per_epoch))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_optimizer_and_scheduler(model, training_args, lr_decay, steps_per_epoch):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps, lr_decay=lr_decay, steps_per_epoch=steps_per_epoch
    )

    return optimizer, lr_scheduler


def setup_logging(training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)


def add_argparse_args(parser):
    requiredNamed = parser.add_argument_group('required named arguments')

    # Model args
    parser.add_argument('--vocab_size', type=int, default=50000, help='The vocab size of the model embedding layer.')
    # Training args
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='The initial learning rate for Adam.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay if we apply some.')
    parser.add_argument('--lr_scheduler', type=str, default='linear', help='Learning rate scheduler used for training.')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Linear warmup over warmup_steps.')
    parser.add_argument('--steps_per_epoch', type=int, default=0, help='Number of steps per training epoch.')
    parser.add_argument('--lr_decay', type=float, default=1., help='Learning rate decay per epoch.')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Total number of training epochs to perform.')
    parser.add_argument('--max_steps', type=int, default=-1, help='The total number of training steps to perform. Overrides --num_train_epochs')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Number of batches used in training dataloader.')
    parser.add_argument('--validation_batch_size', type=int, default=1, help='Number of batches used in training dataloader.')
    parser.add_argument('--block_size', type=int, default=512, help='Number of tokens joined to a single feature.')
    parser.add_argument('--dataloader_drop_last', type=bool, default=False, help='Whether to drop the last incomplete batch.')
    parser.add_argument('--dataloader_num_workers', type=int, default=0, help='Number of subprocesses to use for data loading.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates steps to accumulate the gradients for, before performing a backward/update pass.')
    parser.add_argument('--fp16', type=bool, default=False, help='Whether to use 16-bit (mixed) precision training.')
    parser.add_argument('--logging_steps', type=int, default=500, help='Number of update steps between two logs.')
    parser.add_argument('--eval_steps', type=int, default=500, help='Number of update steps between two evaluations.')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Number of updates steps before two checkpoint saves.')
    parser.add_argument('--limit_train_batches', type=float, default=1.,
                        help='Limits the percentage of train batches used.')
    parser.add_argument('--limit_validation_batches', type=float, default=1.,
                        help='Limits the percentage of validation batches checked.')
    parser.add_argument('--pad_token_id', type=int, default=3,
                        help='Id of the padding token to use.')
    # Script args
    requiredNamed.add_argument('--train_dataset_path', type=str, help='Location of the training dataset.', required=True)
    requiredNamed.add_argument('--validation_dataset_path', type=str, help='Location of the training dataset.', required=True)
    requiredNamed.add_argument('--pretrained_model_dir', type=str, help='Location of the pretrained model.', required=True)
    requiredNamed.add_argument('--output_dir', type=str,
                        help='The output directory  where the model predictions and checkpoints will be written.', required=True)
    parser.add_argument('--overwrite_output_dir', type=str, default=False,
                        help='Whether to overwrite the content of the output directory.')

    return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_argparse_args(parser)

    args = parser.parse_args()

    wandb.init(project="master")
    wandb.config.update(args)

    output_dir = os.path.join(wandb.config.output_dir, wandb.run.id)

    training_args = TrainingArguments(
        learning_rate=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay,
        warmup_steps=wandb.config.warmup_steps,
        num_train_epochs=wandb.config.num_train_epochs,
        max_steps=wandb.config.max_steps,
        per_device_train_batch_size=wandb.config.train_batch_size,
        per_device_eval_batch_size=wandb.config.validation_batch_size,
        dataloader_drop_last=wandb.config.dataloader_drop_last,
        dataloader_num_workers=wandb.config.dataloader_num_workers,
        gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
        fp16=wandb.config.fp16,
        logging_steps=wandb.config.logging_steps,
        evaluation_strategy='steps',
        eval_steps=wandb.config.eval_steps,
        save_steps=wandb.config.save_steps,
        output_dir=output_dir,
        overwrite_output_dir=wandb.config.overwrite_output_dir
    )

    setup_logging(training_args)
    # Set seed before initializing model.
    set_seed(training_args.seed)

    logger.info('Loading Dataset')
    train_data = FinetuneDataset(wandb.config.train_dataset_path, wandb.config.block_size, wandb.config.limit_train_batches)
    validation_data = FinetuneDataset(wandb.config.validation_dataset_path, wandb.config.block_size, wandb.config.limit_validation_batches)

    model = GPT2LMHeadModel.from_pretrained(wandb.config.pretrained_model_dir)
    model.train()

    data_collator = DataCollatorWithPadding(pad_token_id=wandb.config.pad_token_id, label_pad_id=LABEL_PAD_ID)

    optimizers = (None, None)
    if wandb.config.steps_per_epoch > 0:
        optimizers = create_optimizer_and_scheduler(model, training_args, lr_decay=wandb.config.lr_decay,
                                                    steps_per_epoch=wandb.config.steps_per_epoch)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=validation_data,
        optimizers=optimizers
    )

    trainer.train()
    trainer.save_model()
