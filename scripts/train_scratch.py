import logging
import os
import random
import torch
import transformers
import wandb

from argparse import ArgumentParser
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, TrainerCallback, default_data_collator, set_seed
from transformers.trainer_utils import is_main_process

logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)


class IdDataset(Dataset):
    def __init__(self, file_path, block_size, limit=1., init_epochs=0):
        self.file_path = file_path
        self.block_size = block_size
        self.limit = limit
        self.data = self._load_data()

        for _ in range(init_epochs):
            self.data = self._load_data(shuffle=True)

    def _load_data(self, shuffle=False):
        tensor_list = torch.load(self.file_path)

        if shuffle:
            random.shuffle(tensor_list)
        data = torch.cat(tensor_list)
        limit_index = int(len(data)*self.limit)
        limited_data = data[:limit_index]
        num_chunks = len(limited_data) // self.block_size
        reshaped_data = torch.reshape(limited_data[:num_chunks * self.block_size], (-1, self.block_size)).long()
        return reshaped_data

    def reload(self):
        logger.info(f'Reloading dataset {self.file_path}')
        self.data = self._load_data(shuffle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input_ids': self.data[idx], 'labels': self.data[idx]}


class ReloadDatasetCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, train_dataloader=None, **kwargs):
        train_dataloader.dataset.reload()


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
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='Continue Training from Checkpoint.')
    parser.add_argument('--init_epochs', type=int, default=0)
    # Script args
    requiredNamed.add_argument('--train_dataset_path', type=str, help='Location of the training dataset.', required=True)
    requiredNamed.add_argument('--validation_dataset_path', type=str, help='Location of the training dataset.', required=True)
    requiredNamed.add_argument('--output_dir', type=str,
                        help='The output directory where the model predictions and checkpoints will be written.', required=True)
    parser.add_argument('--overwrite_output_dir', type=str, default=False,
                        help='Whether to overwrite the content of the output directory.')

    return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_argparse_args(parser)

    args = parser.parse_args()

    wandb.init(project='master')
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
        dataloader_num_workers=0,
        gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
        fp16=wandb.config.fp16,
        logging_steps=wandb.config.logging_steps,
        evaluation_strategy='steps',
        eval_steps=wandb.config.eval_steps,
        save_steps=wandb.config.save_steps,
        output_dir=output_dir,
        overwrite_output_dir=wandb.config.overwrite_output_dir,
        logging_first_step=True
    )

    setup_logging(training_args)
    # Set seed before initializing model.
    set_seed(training_args.seed)

    logger.info('Loading Dataset')
    train_data = IdDataset(wandb.config.train_dataset_path, wandb.config.block_size, wandb.config.limit_train_batches, init_epochs=wandb.config.init_epochs)
    validation_data = IdDataset(wandb.config.validation_dataset_path, wandb.config.block_size, wandb.config.limit_validation_batches)

    if wandb.config.checkpoint_dir:
        logger.info(f'Loading model from checkpoint {wandb.config.checkpoint_dir}')
        model = GPT2LMHeadModel.from_pretrained(wandb.config.checkpoint_dir)
    else:
        config = GPT2Config(
            vocab_size=wandb.config.vocab_size,
        )
        model = GPT2LMHeadModel(config)

    optimizers = (None, None)
    if wandb.config.steps_per_epoch > 0:
        optimizers = create_optimizer_and_scheduler(model, training_args, lr_decay=wandb.config.lr_decay, steps_per_epoch=wandb.config.steps_per_epoch)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=train_data,
        eval_dataset=validation_data,
        optimizers=optimizers,
        callbacks=[ReloadDatasetCallback]
    )

    if wandb.config.checkpoint_dir:
        logger.info(f'Rusming training from checkpoint {wandb.config.checkpoint_dir}')
        trainer.train(wandb.config.checkpoint_dir)
    else:
        trainer.train()
    trainer.save_model()
