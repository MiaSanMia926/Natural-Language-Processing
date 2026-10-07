# -*- coding: utf-8 -*-
"""所有任务共用的数据协议、固定分词、评价与 Logistic Regression。"""
import hashlib
import json
import random
import re
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
NYT_CSV = DATA_DIR / "nyt.csv"
AG_CSV = DATA_DIR / "ag.csv"
RANDOM_SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
SPLIT_NAMES = ("train", "val", "test")
TOKEN_PATTERN = r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+"
TOKENIZER_CONFIG = {"name": "regex_v1", "lowercase": True, "pattern": TOKEN_PATTERN}
_TOKEN_RE = re.compile(TOKEN_PATTERN)


def tokenize(text: str):
    """固定的小写化 + 正则分词；不依赖 NLTK 或本地语言资源。"""
    if not isinstance(text, str):
        raise TypeError("tokenize 的输入必须是字符串。")
    return _TOKEN_RE.findall(text.lower())


def tokenize_corpus(texts):
    return [tokenize(text) for text in texts]


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def stable_word_hash(text):
    """Word2Vec 使用稳定哈希，避免依赖 Python 进程的随机哈希种子。"""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path, payload):
    """临时文件写完后替换，避免中断留下半个 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # 不同任务可能同时写同一份划分清单，每个写入者使用独立临时文件。
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temp_path = Path(stream.name)
        stream.write(serialized)
    try:
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    return path


def _read_csv(path, columns):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到数据文件：{path}")
    df = pd.read_csv(path, dtype=str)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少列：{sorted(missing)}")
    df["source_row"] = np.arange(len(df))  # CSV 数据行，从 0 开始，不含表头。
    invalid = df[list(columns)].isna().any(axis=1)
    for column in columns:
        invalid |= df[column].fillna("").str.strip().eq("")
    if invalid.any():
        rows = df.loc[invalid, "source_row"].tolist()[:10]
        raise ValueError(f"{path.name} 含空文本/标签，数据行示例：{rows}；请先修复数据。")
    return df


def load_nyt_splits(seed=RANDOM_SEED, *, path=None, verbose=True):
    """完全相同的文本先去重，再随机打乱并按 80/10/10 划分。

    原 CSV 不变。相同文本的标签若冲突则拒绝运行。metadata 包含
    原始行号、文件哈希、类别分布与划分标识，供全部任务共同使用。
    """
    path = Path(path) if path is not None else NYT_CSV
    df = _read_csv(path, ("text", "label"))
    original_size = len(df)
    conflicts = df.groupby("text", sort=False)["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("NYT 中存在相同文本对应不同标签的记录；请先核实标签。")
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    duplicates_removed = original_size - len(df)
    label_order = sorted(df["label"].unique().tolist())
    if len(label_order) < 2:
        raise ValueError("分类任务至少需要两个类别。")
    df = df.iloc[np.random.RandomState(seed).permutation(len(df))]
    n_train, n_val = int(len(df) * TRAIN_RATIO), int(len(df) * VAL_RATIO)
    frames = {
        "train": df.iloc[:n_train],
        "val": df.iloc[n_train:n_train + n_val],
        "test": df.iloc[n_train + n_val:],
    }
    if any(frame.empty for frame in frames.values()):
        raise ValueError("数据量不足，80/10/10 划分后出现空集合。")
    if set(frames["train"]["label"]) != set(label_order):
        raise ValueError("训练集缺少某些类别，请检查样本量与划分设置。")
    splits = {
        name: {"text": frame["text"].tolist(), "label": frame["label"].tolist(),
               "source_row": frame["source_row"].astype(int).tolist()}
        for name, frame in frames.items()
    }
    text_sets = {name: set(splits[name]["text"]) for name in SPLIT_NAMES}
    overlaps = {f"{a}_{b}": len(text_sets[a] & text_sets[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlaps.values()):
        raise RuntimeError("划分中仍存在重复文本交叉，停止实验。")
    split_spec = {
        "dataset_sha256": file_sha256(path),
        "seed": seed,
        "ratios": [TRAIN_RATIO, VAL_RATIO, TEST_RATIO],
        "deduplication": "exact_text_keep_first_reject_conflicting_labels",
        "row_indices": {name: splits[name]["source_row"] for name in SPLIT_NAMES},
    }
    split_id = object_sha256(split_spec)
    splits["metadata"] = {
        **split_spec,
        "dataset_file": path.name,
        "split_id": split_id,
        "protocol_id": object_sha256({"version": 2, "split_id": split_id,
                                      "traditional_tokenizer": TOKENIZER_CONFIG}),
        "original_samples": original_size,
        "duplicates_removed": duplicates_removed,
        "samples_after_deduplication": len(df),
        "label_order": label_order,
        "sizes": {name: len(splits[name]["text"]) for name in SPLIT_NAMES},
        "class_counts": {name: {label: int(Counter(splits[name]["label"])[label])
                                for label in label_order} for name in SPLIT_NAMES},
        "cross_split_duplicate_texts": overlaps,
        "traditional_tokenizer": TOKENIZER_CONFIG,
    }
    if verbose:
        print(f"NYT: {original_size} 条，删除重复文本 {duplicates_removed} 条。")
        print(f"Train/Validation/Test: {splits['metadata']['sizes']}")
        print(f"Split ID: {split_id[:16]}")
    return splits


def load_ag_texts(*, path=None):
    """AG News 是外部无监督训练语料，保留其全部有效文本。"""
    df = _read_csv(path if path is not None else AG_CSV, ("text",))
    return df["text"].tolist()


def evaluate(y_true, y_pred, *, labels=None):
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("真实标签与预测标签必须非空且长度一致。")
    if labels is not None and not (set(y_true) | set(y_pred)).issubset(set(labels)):
        raise ValueError("评价数据中出现未知类别。")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
    }


def fit_logistic_regression(X, labels, *, max_iter=2000):
    """固定基线配置；显式报告收敛状态，不声称迭代上限保证收敛。"""
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=max_iter,
                             class_weight=None, random_state=RANDOM_SEED)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X, labels)
    convergence_warnings = [str(w.message) for w in captured
                            if issubclass(w.category, ConvergenceWarning)]
    for warning in captured:
        warnings.warn(str(warning.message), warning.category, stacklevel=2)
    info = {
        "parameters": clf.get_params(),
        "n_iter": clf.n_iter_.tolist(),
        "converged": not convergence_warnings and int(clf.n_iter_.max()) < max_iter,
        "warnings": convergence_warnings,
    }
    return clf, info
