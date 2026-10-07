# -*- coding: utf-8 -*-
"""Task 2.1：GloVe 6B 100 维有效词均值 + Logistic Regression。

预训练文件放在 data/glove/glove.6B.100d.txt。
模型保存分类器及外部词向量路径、校验值；推理时复用相同的 GloVe 文件。
"""
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.embeddings import build_doc_matrix, doc_vector
from src.experiment import Experiment, portable_path
from src.utils import (
    DATA_DIR, SPLIT_NAMES, TOKENIZER_CONFIG, file_sha256,
    fit_logistic_regression, load_nyt_splits, set_seed, tokenize_corpus,
)

EMBED_DIM = 100
GLOVE_PATH = DATA_DIR / "glove" / "glove.6B.100d.txt"


def load_glove_vectors(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到 {path}；请下载 https://nlp.stanford.edu/data/glove.6B.zip 并解压。")
    vectors = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            parts = line.split()
            if len(parts) != EMBED_DIM + 1:
                raise ValueError(f"GloVe 第 {line_number} 行格式错误，应为单词加 100 个数值。")
            try:
                vector = np.asarray(parts[1:], dtype=np.float32)
            except ValueError as error:
                raise ValueError(f"GloVe 第 {line_number} 行含无效数值。") from error
            if not np.isfinite(vector).all():
                raise ValueError(f"GloVe 第 {line_number} 行含非有限数值。")
            vectors[parts[0]] = vector
    if not vectors:
        raise ValueError("GloVe 文件为空。")
    return vectors


def main():
    set_seed()
    splits = load_nyt_splits()
    experiment = Experiment("task2_glove", splits, {
        "embedding_dim": EMBED_DIM, "pooling": "mean_of_known_token_occurrences",
        "tokenizer": TOKENIZER_CONFIG, "classifier_fit_on": "train",
        "hyperparameter_selection": "fixed_in_advance",
        "embedding_file": portable_path(GLOVE_PATH),
    })
    feature_start = time.perf_counter()
    embeddings = load_glove_vectors(GLOVE_PATH)
    experiment.config["embedding_sha256"] = file_sha256(GLOVE_PATH)
    experiment.config["embedding_vocab_size"] = len(embeddings)
    matrices, coverage = {}, {}
    for name in SPLIT_NAMES:
        matrices[name], coverage[name] = build_doc_matrix(
            tokenize_corpus(splits[name]["text"]), embeddings, EMBED_DIM)
    feature_seconds = time.perf_counter() - feature_start
    fit_start = time.perf_counter()
    classifier, convergence = fit_logistic_regression(matrices["train"], splits["train"]["label"])
    training_seconds = time.perf_counter() - fit_start
    experiment.config["classifier"] = convergence["parameters"]
    experiment.save_model({
        "classifier": classifier, "config": experiment.config,
        "label_order": splits["metadata"]["label_order"],
    }, "classifier.joblib")
    experiment.finish(classifier.predict(matrices["val"]), classifier.predict(matrices["test"]),
                      extra={"coverage": coverage, "convergence": convergence,
                             "timings": {"features_seconds": feature_seconds,
                                         "training_seconds": training_seconds}})


if __name__ == "__main__":
    main()
