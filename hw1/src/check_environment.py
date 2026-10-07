# -*- coding: utf-8 -*-
"""运行前检查依赖、数据、GPU 与统一划分；不会下载模型或开始训练。"""
import argparse
import importlib
from importlib import metadata
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "task1", "glove", "word2vec", "bert"], default="all")
    parser.add_argument(
        "--allow-version-differences", action="store_true",
        help="用于 Colab 等托管环境：检查依赖导入与运行接口，但允许已预装包偏离本地锁定版本。",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    base = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
            "scikit-learn": "sklearn", "joblib": "joblib", "matplotlib": "matplotlib",
            "packaging": "packaging"}
    required = dict(base)
    if args.task in ("all", "word2vec"):
        required["gensim"] = "gensim"
    if args.task in ("all", "bert"):
        required.update({"torch": "torch", "transformers": "transformers", "accelerate": "accelerate"})
    errors, loaded = [], {}
    print(f"Python {sys.version.split()[0]}")
    for package, module in required.items():
        try:
            version = metadata.version(package)
            loaded[package] = importlib.import_module(module)
            print(f"OK {package}=={version}")
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            errors.append(f"{package}: {error}")
    if "packaging" in loaded and not args.allow_version_differences:
        from packaging.requirements import Requirement
        for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            requirement = Requirement(line)
            if requirement.name in loaded:
                version = metadata.version(requirement.name)
                if not requirement.specifier.contains(version, prereleases=True):
                    errors.append(f"{requirement.name} 当前为 {version}，需要 {requirement.specifier}")
    elif "packaging" in loaded:
        print("使用托管环境现有包版本；已安装包的真实版本会记录在实验结果中。")
    if all(name in loaded for name in base):
        from src.utils import DATA_DIR, RESULTS_DIR, load_ag_texts, load_nyt_splits, write_json
        try:
            splits = load_nyt_splits()
            meta = splits["metadata"]
            print(f"类别分布：{meta['class_counts']}")
            print(f"跨划分重复文本：{meta['cross_split_duplicate_texts']}")
            write_json(RESULTS_DIR / "splits" / f"{meta['split_id']}.json", meta)
            if args.task in ("all", "word2vec"):
                print(f"AG News 文本数：{len(load_ag_texts())}")
            if args.task in ("all", "glove"):
                path = DATA_DIR / "glove" / "glove.6B.100d.txt"
                with path.open("r", encoding="utf-8") as stream:
                    if len(stream.readline().split()) != 101:
                        raise ValueError("GloVe 首行不符合 100 维格式。")
                print(f"GloVe 文件存在：{path.stat().st_size / 1024**2:.1f} MiB")
        except (OSError, ValueError, RuntimeError) as error:
            errors.append(f"数据检查：{error}")
    if args.task in ("all", "bert") and all(p in loaded for p in ("torch", "transformers", "accelerate")):
        try:
            from src.task3_bert import make_training_arguments
            # 验证真实 Trainer 导入、TrainingArguments API 和 Accelerate 兼容性。
            from transformers import Trainer
            arguments = make_training_arguments(root / "results" / "preflight")
            torch = loaded["torch"]
            print(f"BERT device: {arguments.device}")
            print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
            print("BERT 预训练权重将在正式运行时按需下载。")
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as error:
            errors.append(f"BERT 环境：{error}")
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        print("请在项目根目录运行：python -m pip install -r requirements.txt")
        raise SystemExit(1)
    print("检查通过。")


if __name__ == "__main__":
    main()
