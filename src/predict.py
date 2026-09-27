# -*- coding: utf-8 -*-
"""加载已保存模型进行单条预测，不重新训练。

例如：
python src/predict.py --experiment task1_binary_bow --text "The team won the match."
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from src.embeddings import build_doc_matrix
from src.experiment import EXPERIMENTS
from src.utils import PROJECT_ROOT, RESULTS_DIR, file_sha256, tokenize_corpus


def predict_saved(model_dir, texts):
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "experiment.json").read_text(encoding="utf-8"))
    name, config = metadata["experiment"], metadata["config"]
    if name.startswith("task1_"):
        return joblib.load(model_dir / "model.joblib").predict(texts).tolist()
    if name.startswith("task2_"):
        bundle = joblib.load(model_dir / "classifier.joblib")
        if name == "task2_glove":
            from src.task2_glove import load_glove_vectors
            embedding_path = PROJECT_ROOT / config["embedding_file"]
            if file_sha256(embedding_path) != config["embedding_sha256"]:
                raise ValueError("GloVe 文件与训练时不一致。")
            embeddings = load_glove_vectors(embedding_path)
        else:
            from gensim.models import Word2Vec
            embeddings = Word2Vec.load(str(model_dir / "word2vec.model")).wv
        matrix, _ = build_doc_matrix(tokenize_corpus(texts), embeddings, config["embedding_dim"])
        return bundle["classifier"].predict(matrix).tolist()
    if name == "task3_bert":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True, attn_implementation="eager")
        inputs = tokenizer(texts, padding="max_length", truncation=True,
                           max_length=config["max_length"], return_tensors="pt")
        model.eval()
        with torch.inference_mode():
            indices = model(**inputs).logits.argmax(dim=-1).tolist()
        return [model.config.id2label[index] for index in indices]
    raise ValueError(f"未知模型类型：{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=[item[0] for item in EXPERIMENTS], required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-dir", type=Path, help="可选；默认使用该实验最近一次成功运行的模型。")
    args = parser.parse_args()
    model_dir = args.model_dir
    if model_dir is None:
        result_path = RESULTS_DIR / f"{args.experiment}.json"
        if not result_path.is_file():
            raise SystemExit(f"尚无 {args.experiment} 的正式结果，请先完成训练。")
        data = json.loads(result_path.read_text(encoding="utf-8"))
        model_dir = PROJECT_ROOT / data["model_dir"]
    metadata = json.loads((model_dir / "experiment.json").read_text(encoding="utf-8"))
    if metadata["experiment"] != args.experiment:
        raise SystemExit("--experiment 与模型目录中的实验名称不一致。")
    print(json.dumps({"text": args.text, "prediction": predict_saved(model_dir, [args.text])[0]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
