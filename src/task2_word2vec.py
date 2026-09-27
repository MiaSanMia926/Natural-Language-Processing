# -*- coding: utf-8 -*-
"""Task 2.2/2.3：AG News 或 NYT Train 上训练 100 维 Word2Vec。"""
import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import build_doc_matrix, doc_vector
from src.experiment import Experiment
from src.utils import (
    AG_CSV, RANDOM_SEED, SPLIT_NAMES, TOKENIZER_CONFIG, file_sha256,
    fit_logistic_regression, load_ag_texts, load_nyt_splits, set_seed,
    stable_word_hash, tokenize_corpus,
)

EMBED_DIM = 100
WORD2VEC_PARAMS = {
    "vector_size": EMBED_DIM, "window": 5, "min_count": 5, "sg": 1,
    "epochs": 10, "seed": RANDOM_SEED, "workers": 1,
    "negative": 5, "hs": 0, "sample": 1e-3, "alpha": 0.025,
    "min_alpha": 0.0001, "ns_exponent": 0.75, "sorted_vocab": 1,
    "shrink_windows": True, "batch_words": 10000,
}


def train_word2vec(tokenized_corpus, **overrides):
    from gensim.models import Word2Vec

    params = {**WORD2VEC_PARAMS, **overrides}
    # 稳定哈希替代 Python 随机化的 hash；单线程消除线程调度差异。
    return Word2Vec(sentences=tokenized_corpus, hashfxn=stable_word_hash, **params)


def run(corpus_name):
    if corpus_name not in {"ag", "nyt"}:
        raise ValueError("corpus 必须为 ag 或 nyt。")
    set_seed()
    splits = load_nyt_splits()
    experiment = Experiment(f"task2_word2vec_{corpus_name}", splits, {
        "embedding_dim": EMBED_DIM, "word2vec": WORD2VEC_PARAMS.copy(),
        "word_hash": "sha256_first_32_bits", "tokenizer": TOKENIZER_CONFIG,
        "pooling": "mean_of_known_token_occurrences", "classifier_fit_on": "train",
        "hyperparameter_selection": "fixed_in_advance",
        "embedding_fit_on": "AG News" if corpus_name == "ag" else "NYT train only",
    })
    tokenized = {name: tokenize_corpus(splits[name]["text"]) for name in SPLIT_NAMES}
    if corpus_name == "ag":
        train_corpus = tokenize_corpus(load_ag_texts())
        experiment.config["ag_sha256"] = file_sha256(AG_CSV)
    else:
        train_corpus = tokenized["train"]
    experiment.config["embedding_training_documents"] = len(train_corpus)
    fit_start = time.perf_counter()
    model = train_word2vec(train_corpus)
    embedding_seconds = time.perf_counter() - fit_start
    experiment.config["embedding_vocab_size"] = len(model.wv)
    matrices, coverage = {}, {}
    feature_start = time.perf_counter()
    for name in SPLIT_NAMES:
        matrices[name], coverage[name] = build_doc_matrix(tokenized[name], model.wv, EMBED_DIM)
    feature_seconds = time.perf_counter() - feature_start
    fit_start = time.perf_counter()
    classifier, convergence = fit_logistic_regression(matrices["train"], splits["train"]["label"])
    classifier_seconds = time.perf_counter() - fit_start
    experiment.config["classifier"] = convergence["parameters"]
    model.save(str(experiment.model_dir / "word2vec.model"))
    experiment.save_model({
        "classifier": classifier, "config": experiment.config,
        "label_order": splits["metadata"]["label_order"],
    }, "classifier.joblib")
    return experiment.finish(
        classifier.predict(matrices["val"]), classifier.predict(matrices["test"]),
        extra={"coverage": coverage, "convergence": convergence,
               "timings": {"embedding_training_seconds": embedding_seconds,
                           "features_seconds": feature_seconds,
                           "classifier_training_seconds": classifier_seconds}},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["ag", "nyt"], required=True)
    run(parser.parse_args().corpus)


if __name__ == "__main__":
    main()
