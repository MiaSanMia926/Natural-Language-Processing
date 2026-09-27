# -*- coding: utf-8 -*-
"""Task 1：Binary BoW / Word Frequency + Logistic Regression。"""
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from src.experiment import Experiment
from src.utils import (
    TOKENIZER_CONFIG, fit_logistic_regression, load_nyt_splits, set_seed, tokenize,
)


def run_bow(binary, splits, vectorizer_vocab=None):
    feature_start = time.perf_counter()
    vectorizer = CountVectorizer(
        tokenizer=tokenize, lowercase=False, token_pattern=None,
        binary=binary, vocabulary=vectorizer_vocab,
    )
    X_train = vectorizer.fit_transform(splits["train"]["text"])
    X_val = vectorizer.transform(splits["val"]["text"])
    X_test = vectorizer.transform(splits["test"]["text"])
    feature_seconds = time.perf_counter() - feature_start
    fit_start = time.perf_counter()
    classifier, convergence = fit_logistic_regression(X_train, splits["train"]["label"])
    training_seconds = time.perf_counter() - fit_start
    return {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "val_predictions": classifier.predict(X_val),
        "test_predictions": classifier.predict(X_test),
        "convergence": convergence,
        "timings": {"features_seconds": feature_seconds, "training_seconds": training_seconds},
    }


def main():
    set_seed()
    splits, vocabulary = load_nyt_splits(), None
    for binary, name in ((True, "task1_binary_bow"), (False, "task1_word_frequency")):
        experiment = Experiment(name, splits, {
            "representation": "binary" if binary else "raw_counts",
            "tokenizer": TOKENIZER_CONFIG,
            "vocabulary_fit_on": "train",
            "classifier_fit_on": "train",
            "hyperparameter_selection": "fixed_in_advance",
        })
        result = run_bow(binary, splits, vocabulary)
        vocabulary = result["vectorizer"].vocabulary_
        experiment.config["vocab_size"] = len(vocabulary)
        experiment.config["classifier"] = result["convergence"]["parameters"]
        # Pipeline 包含可序列化的分词器与训练词表，可直接 predict 原始文本。
        model = Pipeline([("vectorizer", result["vectorizer"]),
                          ("classifier", result["classifier"])])
        experiment.save_model(model)
        print(f"Vocabulary: {len(vocabulary)}")
        experiment.finish(result["val_predictions"], result["test_predictions"], extra={
            "convergence": result["convergence"], "timings": result["timings"],
        })


if __name__ == "__main__":
    main()
