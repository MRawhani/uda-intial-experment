import torch
from config.config import Config
from sklearn.metrics import accuracy_score, f1_score,precision_recall_fscore_support
import numpy as np
from transformers import TrainingArguments, EvalPrediction,default_data_collator
from adapters import AdapterTrainer

import collections


import os
def evaluate_model(model, dataloader):
    model.to(Config.DEVICE)
    model.eval()
    predictions, true_labels = [], []
    for valid_step, batch in enumerate(dataloader):

        with torch.no_grad():
            batch = {k: v.to(Config.DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    return accuracy, f1


def print_trainable_parameters(model):

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def train_model(model,prepended_path,train_data, eval_data=None):
    training_args = TrainingArguments(
        
        output_dir=f"{Config.RESULTS_SAVE_PATH}/{prepended_path}/results",                 # Where to store the output (checkpoints and predictions)
        num_train_epochs=10,                     # Total number of training epochs
        per_device_train_batch_size=32,         # Batch size for training
        per_device_eval_batch_size=64,          # Batch size for evaluation
        warmup_steps=500,                       # Number of warmup steps for learning rate scheduler
        learning_rate=1e-4,
        weight_decay=0.01,                      # Strength of weight decay
        logging_dir=f"{Config.RESULTS_SAVE_PATH}/{prepended_path}/logs",                   # Directory for storing logs
        logging_steps=500,                       # Log every X updates steps
        evaluation_strategy="steps" if eval_data is not None else "no",            # Evaluate model every X steps
        eval_steps=500,                         # Number of steps to perform evaluation
        save_steps=500,                         # Save checkpoint every X steps
        save_total_limit=2,                     # Limit the total amount of checkpoints
        load_best_model_at_end=True if eval_data is not None else False,            # Load the best model when finished training
        metric_for_best_model="accuracy",       # Use accuracy to find the best model
        greater_is_better=True,                 # Higher accuracy is better
        report_to="none"                        # Do not report to any online service
    )
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    trainer = AdapterTrainer(
        model=model,                           # The instantiated 🤗 Transformers model to be trained
        args=training_args,                    # Training arguments, defined above
        train_dataset=train_data,           # Training dataset
        eval_dataset=eval_data if eval_data is not None else None,
        compute_metrics=compute_metrics if eval_data is not None else None    )
    trainer.train()
    return trainer

def group_texts(examples,chunk_size):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result



def whole_word_masking_data_collator(features,tokenizer):   
    wwm_probability = 0.15

    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)