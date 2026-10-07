"""离线小样本检查；全部输出进入临时目录，不会覆盖正式结果。

运行：python -m unittest discover -s tests -v
Word2Vec/BERT 检查在依赖安装后自动启用；不下载预训练权重。
"""
from functools import partial
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.embeddings import build_doc_matrix, doc_vector
from src.experiment import Experiment, load_results
from src.predict import predict_saved
from src.task1_bow import main as bow_main, run_bow
from src.task2_glove import load_glove_vectors
from src.utils import (
    PROJECT_ROOT, evaluate, fit_logistic_regression, load_nyt_splits,
    tokenize, write_json,
)


def make_csv(path):
    topics = {"sports": "team match ball", "politics": "election vote government",
              "business": "company profit market"}
    rows = [{"text": f"{words} article {index}", "label": label}
            for index in range(30) for label, words in topics.items()]
    pd.DataFrame(rows + [rows[0], rows[1]]).to_csv(path, index=False)
    return rows


class DataProtocolTests(unittest.TestCase):
    def test_deduplication_consistent_disjoint_splits_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nyt.csv"
            make_csv(path)
            first = load_nyt_splits(path=path, verbose=False)
            second = load_nyt_splits(path=path, verbose=False)
            self.assertEqual(first, second)
            meta = first["metadata"]
            self.assertEqual(meta["duplicates_removed"], 2)
            self.assertEqual(meta["sizes"], {"train": 72, "val": 9, "test": 9})
            self.assertTrue(all(count == 0 for count in meta["cross_split_duplicate_texts"].values()))
            rows = [row for name in ("train", "val", "test") for row in first[name]["source_row"]]
            self.assertEqual(len(rows), len(set(rows)))
            self.assertEqual(len(rows), 90)
            self.assertNotEqual(meta["split_id"],
                                load_nyt_splits(seed=43, path=path, verbose=False)["metadata"]["split_id"])
            self.assertEqual(len(pd.read_csv(path)), 92)  # 原始数据不被修改。

    def test_conflicting_duplicate_labels_and_blank_text_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nyt.csv"
            rows = make_csv(path)
            pd.DataFrame(rows + [{**rows[0], "label": "other"}]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "不同标签"):
                load_nyt_splits(path=path, verbose=False)
            rows[0]["text"] = " "
            pd.DataFrame(rows).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "空文本"):
                load_nyt_splits(path=path, verbose=False)

    def test_tokenizer_does_not_depend_on_nltk(self):
        text = "U.S. stocks aren't down 3.5%."
        with patch.dict(sys.modules, {"nltk": None}):
            tokens = tokenize(text)
        self.assertEqual(tokens, ["u", "s", "stocks", "aren't", "down", "3", "5"])

    def test_macro_f1_includes_all_declared_classes(self):
        scores = evaluate(["a", "a"], ["a", "a"], labels=["a", "b", "c"])
        self.assertEqual(scores["accuracy"], 1.0)
        self.assertAlmostEqual(scores["macro_f1"], 1 / 3)
        with self.assertRaises(ValueError):
            evaluate(["a"], ["unknown"], labels=["a", "b"])


class ModelAndArtifactTests(unittest.TestCase):
    def test_binary_counts_and_train_only_vocabulary(self):
        splits = {
            "train": {"text": ["apple apple fruit", "banana fruit", "team team ball", "match ball"],
                      "label": ["food", "food", "sport", "sport"]},
            "val": {"text": ["apple validationonly"], "label": ["food"]},
            "test": {"text": ["team testonly"], "label": ["sport"]},
        }
        binary = run_bow(True, splits)
        vectorizer = binary["vectorizer"]
        self.assertNotIn("testonly", vectorizer.vocabulary_)
        self.assertNotIn("validationonly", vectorizer.vocabulary_)
        self.assertEqual(vectorizer.transform(["apple apple"])[0, vectorizer.vocabulary_["apple"]], 1)
        counts = run_bow(False, splits, vectorizer.vocabulary_)["vectorizer"]
        self.assertEqual(counts.transform(["apple apple"])[0, counts.vocabulary_["apple"]], 2)
        self.assertEqual(vectorizer.vocabulary_, counts.vocabulary_)

    def test_embedding_mean_counts_occurrences_ignores_oov_and_handles_empty(self):
        embeddings = {"a": np.array([1, 0]), "b": np.array([0, 3])}
        matrix, coverage = build_doc_matrix([["a", "a", "missing", "b"], []], embeddings, 2)
        np.testing.assert_allclose(matrix[0], [2 / 3, 1], rtol=1e-6)
        np.testing.assert_array_equal(matrix[1], [0, 0])
        self.assertEqual(coverage["known_tokens"], 3)
        self.assertEqual(coverage["empty_documents"], 1)
        self.assertEqual(coverage["coverage"], 0.75)
        with self.assertRaises(ValueError):
            doc_vector(["a"], {"a": [float("nan"), 0]}, 2)

    def test_glove_validates_dimensions_and_finite_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "vectors.txt"
            path.write_text("word " + " ".join(["0.5"] * 100) + "\n", encoding="utf-8")
            self.assertEqual(load_glove_vectors(path)["word"].shape, (100,))
            path.write_text("word 1 2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_glove_vectors(path)

    def test_convergence_is_reported_instead_of_assumed(self):
        X = np.array([[1, 0], [2, 0], [0, 1], [0, 2]], dtype=float)
        with self.assertWarns(ConvergenceWarning):
            _, diagnostics = fit_logistic_regression(X, ["a", "a", "b", "b"], max_iter=1)
        self.assertFalse(diagnostics["converged"])
        self.assertTrue(diagnostics["warnings"])

    def test_glove_main_exports_and_external_vectors_reload(self):
        from src.task2_glove import main as glove_main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_path, vector_path = root / "nyt.csv", root / "glove.txt"
            make_csv(data_path)
            splits = load_nyt_splits(path=data_path, verbose=False)
            words = ["team", "match", "ball", "election", "vote", "government",
                     "company", "profit", "market"]
            lines = []
            for index, word in enumerate(words):
                vector = np.zeros(100)
                vector[index // 3] = 1
                lines.append(word + " " + " ".join(str(value) for value in vector))
            vector_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            factory = partial(Experiment, results_dir=root / "results", models_dir=root / "models")
            with patch("src.task2_glove.load_nyt_splits", return_value=splits), \
                    patch("src.task2_glove.GLOVE_PATH", vector_path), \
                    patch("src.task2_glove.Experiment", side_effect=factory):
                glove_main()
            data = load_results(root / "results")["task2_glove"]
            expected = pd.read_csv(Path(data["details_dir"]) / "test_predictions.csv")["predicted_label"].tolist()
            self.assertEqual(predict_saved(data["model_dir"], splits["test"]["text"]), expected)
            with vector_path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "训练时不一致"):
                predict_saved(data["model_dir"], splits["test"]["text"])

    def test_bow_main_exports_and_models_reload_in_a_new_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_path = root / "nyt.csv"
            make_csv(data_path)
            splits = load_nyt_splits(path=data_path, verbose=False)
            factory = partial(Experiment, results_dir=root / "results", models_dir=root / "models")
            with patch("src.task1_bow.load_nyt_splits", return_value=splits), \
                    patch("src.task1_bow.Experiment", side_effect=factory):
                bow_main()
            records = load_results(root / "results", expected_protocol_id=splits["metadata"]["protocol_id"])
            self.assertEqual(len(records), 2)
            for name, data in records.items():
                model_dir, detail_dir = Path(data["model_dir"]), Path(data["details_dir"])
                expected = pd.read_csv(detail_dir / "test_predictions.csv")["predicted_label"].tolist()
                self.assertEqual(predict_saved(model_dir, splits["test"]["text"]), expected)
                matrix = pd.read_csv(detail_dir / "test_confusion_matrix.csv", index_col=0)
                self.assertEqual(int(matrix.to_numpy().sum()), len(expected))
                self.assertTrue((detail_dir / "test_confusion_matrix.png").is_file())
                self.assertTrue((detail_dir / "test_error_examples.json").is_file())
                self.assertTrue((detail_dir / "requirements-resolved.txt").is_file())
                proc = subprocess.run(
                    [sys.executable, str(PROJECT_ROOT / "src" / "predict.py"),
                     "--experiment", name, "--model-dir", str(model_dir),
                     "--text", splits["test"]["text"][0]],
                    cwd=temp, capture_output=True, text=True, encoding="utf-8", check=True,
                )
                self.assertEqual(json.loads(proc.stdout)["prediction"], expected[0])

    def test_error_exports_and_fixed_label_order(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nyt.csv"
            make_csv(path)
            splits = load_nyt_splits(path=path, verbose=False)
            exp = Experiment("task1_binary_bow", splits, {},
                             results_dir=Path(temp) / "results", models_dir=Path(temp) / "models")
            exp.save_model({"test_fixture": True})
            predictions = list(splits["test"]["label"])
            predictions[0] = next(label for label in exp.labels if label != predictions[0])
            data = exp.finish(splits["val"]["label"], predictions)
            errors = pd.read_csv(exp.detail_dir / "test_errors.csv")
            self.assertEqual(len(errors), 1)
            self.assertEqual(int(errors.iloc[0]["source_row"]), splits["test"]["source_row"][0])
            self.assertEqual(data["label_order"], sorted(set(splits["train"]["label"])))
            self.assertAlmostEqual(data["accuracy"], 8 / 9)

    def test_rejects_legacy_mixed_and_outdated_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "task1_binary_bow.json"
            write_json(first, {"accuracy": 1.0, "macro_f1": 1.0})
            with self.assertRaisesRegex(ValueError, "旧结果"):
                load_results(root)
            data = {"schema_version": 2, "experiment": "task1_binary_bow",
                    "protocol_id": "one", "accuracy": 1.0, "macro_f1": 1.0,
                    "val_accuracy": 1.0, "val_macro_f1": 1.0}
            write_json(first, data)
            with self.assertRaisesRegex(ValueError, "当前数据"):
                load_results(root, expected_protocol_id="two")
            write_json(root / "task1_word_frequency.json",
                       {**data, "experiment": "task1_word_frequency", "protocol_id": "two"})
            with self.assertRaisesRegex(ValueError, "不同数据划分"):
                load_results(root)


@unittest.skipUnless(importlib.util.find_spec("gensim"), "需要安装 gensim")
class Word2VecSmokeTests(unittest.TestCase):
    def test_both_corpora_export_and_reload(self):
        from gensim.models import Word2Vec
        from src.task2_word2vec import run
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_path = root / "nyt.csv"
            make_csv(data_path)
            splits = load_nyt_splits(path=data_path, verbose=False)
            ag_texts = [text + " agonly" for text in pd.read_csv(data_path)["text"].tolist()]
            ag_path = root / "ag.csv"
            pd.DataFrame({"text": ag_texts}).to_csv(ag_path, index=False)
            factory = partial(Experiment, results_dir=root / "results", models_dir=root / "models")
            with patch("src.task2_word2vec.load_nyt_splits", return_value=splits), \
                    patch("src.task2_word2vec.load_ag_texts", return_value=ag_texts), \
                    patch("src.task2_word2vec.AG_CSV", ag_path), \
                    patch("src.task2_word2vec.Experiment", side_effect=factory):
                for corpus in ("ag", "nyt"):
                    data = run(corpus)
                    model_dir = Path(data["model_dir"])
                    model = Word2Vec.load(str(model_dir / "word2vec.model"))
                    self.assertEqual("agonly" in model.wv, corpus == "ag")
                    count = len(ag_texts) if corpus == "ag" else len(splits["train"]["text"])
                    self.assertEqual(data["config"]["embedding_training_documents"], count)
                    expected = pd.read_csv(Path(data["details_dir"]) / "test_predictions.csv")["predicted_label"].tolist()
                    self.assertEqual(predict_saved(model_dir, splits["test"]["text"]), expected)

    def test_reproducible_training_and_model_reload(self):
        from gensim.models import Word2Vec
        from src.task2_word2vec import train_word2vec
        corpus = [["team", "wins", "match"], ["market", "company", "profit"]] * 10
        first = train_word2vec(corpus, min_count=1, epochs=3)
        second = train_word2vec(corpus, min_count=1, epochs=3)
        np.testing.assert_array_equal(first.wv.vectors, second.wv.vectors)
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "word2vec.model")
            first.save(path)
            restored = Word2Vec.load(path)
            np.testing.assert_array_equal(first.wv.vectors, restored.wv.vectors)


@unittest.skipUnless(all(importlib.util.find_spec(p) for p in ("torch", "transformers", "accelerate")),
                     "需要安装 torch、transformers、accelerate")
class BertSmokeTests(unittest.TestCase):
    def test_offline_tiny_bert_train_evaluate_save_reload(self):
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
        from src.task3_bert import NYTDataset, main as bert_main, make_training_arguments
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "vocab.txt").write_text(
                "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nteam\nmarket\nvote\n", encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(root / "vocab.txt"))
            data_path = root / "nyt.csv"
            make_csv(data_path)
            splits = load_nyt_splits(path=data_path, verbose=False)
            labels = {label: index for index, label in enumerate(splits["metadata"]["label_order"])}
            ds = NYTDataset(["team", "market", "vote"] * 3, list(labels) * 3, tokenizer, labels)
            self.assertEqual(tuple(ds[0]["input_ids"].shape), (64,))
            config = BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32, num_labels=3,
                                id2label={index: label for label, index in labels.items()},
                                label2id=labels)
            model = BertForSequenceClassification(config)
            factory = partial(Experiment, results_dir=root / "results", models_dir=root / "models")
            def training_arguments(path):
                arguments = make_training_arguments(path, use_cpu=True)
                arguments.disable_tqdm = True
                return arguments
            with patch("src.task3_bert.load_nyt_splits", return_value=splits), \
                    patch("src.task3_bert.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                    patch("src.task3_bert.AutoModelForSequenceClassification.from_pretrained", return_value=model), \
                    patch("src.task3_bert.Experiment", side_effect=factory), \
                    patch("src.task3_bert.make_training_arguments", side_effect=training_arguments):
                bert_main()
            result = load_results(root / "results")["task3_bert"]
            self.assertAlmostEqual(result["diagnostics"]["completed_epochs"], 3.0)
            expected = pd.read_csv(Path(result["details_dir"]) / "test_predictions.csv")["predicted_label"].tolist()
            self.assertEqual(predict_saved(result["model_dir"], splits["test"]["text"]), expected)


if __name__ == "__main__":
    unittest.main()
