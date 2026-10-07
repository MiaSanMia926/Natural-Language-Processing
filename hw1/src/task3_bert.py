# -*- coding: utf-8 -*-
"""Task 3：BERT-base-uncased，最大长度 64，完整训练 3 epochs。"""
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed,
)
from src.experiment import Experiment
from src.utils import RANDOM_SEED, evaluate, load_nyt_splits, write_json

MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 64
NUM_EPOCHS = 3
BATCH_SIZE = 16
SEED = RANDOM_SEED


class NYTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id):
        if len(texts) != len(labels):
            raise ValueError("文本与标签数量不一致。")
        unknown = set(labels) - set(label2id)
        if unknown:
            raise ValueError(f"数据中有训练集未出现的类别：{sorted(unknown)}")
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=MAX_LENGTH,
        )
        self.labels = [label2id[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx], dtype=torch.long)
                for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    return evaluate(labels, np.argmax(logits, axis=-1), labels=list(range(logits.shape[-1])))


def make_training_arguments(output_dir, *, use_cpu=False):
    return TrainingArguments(
        output_dir=str(Path(output_dir).resolve()),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="no",  # 使用完成第 3 轮后的模型，随后显式保存。
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        optim="adamw_torch",
        seed=SEED,
        data_seed=SEED,
        full_determinism=True,
        dataloader_num_workers=0,
        report_to=[],
        use_cpu=use_cpu,
    )


def main():
    set_seed(SEED)  # 在创建随机初始化的分类头之前固定 Python/NumPy/PyTorch 种子。
    splits = load_nyt_splits()
    labels = splits["metadata"]["label_order"]
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    experiment = Experiment("task3_bert", splits, {
        "model": MODEL_NAME, "max_length": MAX_LENGTH, "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE, "learning_rate": 2e-5, "weight_decay": 0.01,
        "seed": SEED, "full_determinism": True, "label2id": label2id,
        "tokenizer": "pretrained_bert_uncased",
        "classifier_fit_on": "train", "checkpoint_selection": "final_epoch_3",
        "hyperparameter_selection": "fixed_in_advance",
    })
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id,
        attn_implementation="eager",
    )
    experiment.config["resolved_model_revision"] = getattr(model.config, "_commit_hash", None)
    datasets = {name: NYTDataset(splits[name]["text"], splits[name]["label"], tokenizer, label2id)
                for name in ("train", "val", "test")}
    training_args = make_training_arguments(experiment.detail_dir / "trainer")
    write_json(experiment.detail_dir / "training_arguments.json", training_args.to_dict())
    trainer = Trainer(
        model=model, args=training_args, train_dataset=datasets["train"],
        eval_dataset=datasets["val"], compute_metrics=compute_metrics,
    )
    print(f"BERT device: {training_args.device}")
    fit_start = time.perf_counter()
    train_output = trainer.train()
    training_seconds = time.perf_counter() - fit_start
    # 显式评价最终模型，验证集与测试集均保存完整预测及指标。
    val_output = trainer.predict(datasets["val"], metric_key_prefix="val")
    test_output = trainer.predict(datasets["test"], metric_key_prefix="test")
    val_predictions = [id2label[int(i)] for i in np.argmax(val_output.predictions, axis=-1)]
    test_predictions = [id2label[int(i)] for i in np.argmax(test_output.predictions, axis=-1)]
    trainer.save_model(str(experiment.model_dir))
    tokenizer.save_pretrained(str(experiment.model_dir))
    write_json(experiment.detail_dir / "training_log.json", trainer.state.log_history)
    experiment.finish(val_predictions, test_predictions, extra={
        "training_seconds": training_seconds,
        "train_metrics": train_output.metrics,
        "completed_epochs": trainer.state.epoch,
        "device": str(training_args.device),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })


if __name__ == "__main__":
    main()
