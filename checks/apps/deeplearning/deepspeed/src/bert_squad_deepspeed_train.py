# This script is based on
# https://keras.io/examples/nlp/text_extraction_with_bert/

import argparse
import datasets
import deepspeed
import os
import torch
import utility.data_processing as du
from transformers import BertTokenizer, BertForQuestionAnswering
from tokenizers import BertWordPieceTokenizer
from datasets.utils import disable_progress_bar
from datasets import disable_caching


disable_progress_bar()
disable_caching()

# Benchmark settings
parser = argparse.ArgumentParser(description='BERT finetuning on SQuAD')
parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='Model')
parser.add_argument('--download-only', action='store_true',
                    help='Download model, tokenizer, dataset and exit')
parser.add_argument('--num-epochs', type=int, default=1,
                    help='number of benchmark iterations')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

bert_cache = os.path.join(os.getcwd(), 'cache')
slow_tokenizer = BertTokenizer.from_pretrained(
    args.model,
    cache_dir=os.path.join(bert_cache, f'{args.model}-tokenizer_slow')
)
vocab_path = os.path.join(bert_cache, f'{args.model}-tokenizer_fast')
if not os.path.exists(vocab_path):
    os.makedirs(vocab_path)
    slow_tokenizer.save_pretrained(vocab_path)


tokenizer = BertWordPieceTokenizer(
    os.path.join(vocab_path, 'vocab.txt'),
    lowercase=True
)

hf_dataset = datasets.load_dataset(
    'squad',
    cache_dir=os.path.join(bert_cache, 'dataset')
)

model = BertForQuestionAnswering.from_pretrained(
    args.model,
    cache_dir=os.path.join(bert_cache, f'{args.model}_qa')
)

if args.download_only:
    exit()

max_len = 384
ds_filtered = hf_dataset.filter(
    lambda example: not du.skip_squad_example(example, max_len, tokenizer),
    num_proc=2
)
ds_processed = ds_filtered.map(
    lambda example: du.process_squad_item(example, max_len, tokenizer),
    remove_columns=hf_dataset["train"].column_names,
    num_proc=2
)
train_set = ds_processed["train"]
train_set.set_format(type='torch')


model.train()
# deepspeed
parameters = filter(lambda p: p.requires_grad, model.parameters())

model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=parameters,
    training_data=train_set
)

# training
num_epochs = args.num_epochs
for epoch in range(num_epochs):
    for i, batch in enumerate(trainloader, 0):
        outputs = model(
            input_ids=batch['input_ids'].to(model_engine.device),
            token_type_ids=batch['token_type_ids'].to(model_engine.device),
            attention_mask=batch['attention_mask'].to(model_engine.device),
            start_positions=batch['start_token_idx'].to(model_engine.device),
            end_positions=batch['end_token_idx'].to(model_engine.device)
        )
        loss = outputs[0]
        model_engine.backward(loss)
        model_engine.step()

rank = torch.distributed.get_rank()
if rank == 0:
    model_path_name = f'{args.model}-trained-deepspeed'
    torch.save(model.state_dict(), model_path_name)  # save model's state_dict
    print('Training complete')
