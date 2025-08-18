"""
Fine-tune a MarianMT model (Helsinki-NLP opus) for en↔hi or en↔mr using Hugging Face
Transformers Trainer, save locally, and make the app prefer local models.

Usage (example):
    python training/fine_tune_mt.py \
      --src_lang en --tgt_lang hi \
      --train_file data/train.csv --eval_file data/dev.csv \
      --output_dir models/local/en-hi \
      --num_train_epochs 1 --per_device_train_batch_size 8

CSV format:
    src,tgt
    Hello world,नमस्ते दुनिया

Notes:
- Tokenizer: Helsinki Marian tokenizer inferred from model name.
- Model name default is based on src/tgt pair; override via --model_name if needed.
- Evaluation prints BLEU using sacrebleu (detokenized strings) on the fly.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
import sacrebleu

PAIR_TO_MODEL = {
    ("en","hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi","en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en","mr"): "Helsinki-NLP/opus-mt-en-mr",
    ("mr","en"): "Helsinki-NLP/opus-mt-mr-en",
}


def load_csv(path: str) -> Dataset:
    df = pd.read_csv(path)
    if not {"src","tgt"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: src,tgt")
    return Dataset.from_pandas(df[["src","tgt"]])


def preprocess(examples, tokenizer, src_lang: str, tgt_lang: str, max_len: int = 128):
    inputs = examples["src"]
    targets = examples["tgt"]
    model_inputs = tokenizer(inputs, max_length=max_len, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_len, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_bleu(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = [[l for l in label if l != -100] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
    return {"bleu": bleu}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_lang", required=True, choices=["en","hi","mr"]) 
    ap.add_argument("--tgt_lang", required=True, choices=["en","hi","mr"]) 
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--eval_file", required=False)
    ap.add_argument("--model_name", required=False)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    args = ap.parse_args()

    pair = (args.src_lang, args.tgt_lang)
    model_name = args.model_name or PAIR_TO_MODEL.get(pair)
    if not model_name:
        raise ValueError(f"No base model for pair {pair}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_ds = load_csv(args.train_file)
    eval_ds = load_csv(args.eval_file) if args.eval_file else None

    tokenized_train = train_ds.map(lambda x: preprocess(x, tokenizer, args.src_lang, args.tgt_lang, args.max_len),
                                   batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = None
    if eval_ds is not None:
        tokenized_eval = eval_ds.map(lambda x: preprocess(x, tokenizer, args.src_lang, args.tgt_lang, args.max_len),
                                     batched=True, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="steps" if tokenized_eval is not None else "no",
        logging_strategy="steps",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        predict_with_generate=True,
        fp16=False,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(lambda p: compute_bleu(p, tokenizer)) if tokenized_eval is not None else None,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved fine-tuned model to {output_dir}")
    # Hint the app to use local models first
    os.environ["LOCAL_MODEL_ROOT"] = str(Path("models/local").absolute())
    print("Set LOCAL_MODEL_ROOT to models/local (app will prefer local fine-tuned models if present)")


if __name__ == "__main__":
    main()
