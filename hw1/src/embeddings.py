"""有效词向量均值；GloVe、Word2Vec 共用完全相同的聚合规则。"""
import numpy as np


def doc_vector(tokens, embeddings, dim=100):
    vectors = [embeddings[word] for word in tokens if word in embeddings]
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dim or not np.isfinite(matrix).all():
        raise ValueError(f"词向量必须是有限数值且维度为 {dim}。")
    # 按词出现次数取平均，不去重；分母仅包含词表中存在的有效词。
    return matrix.mean(axis=0)


def build_doc_matrix(tokenized_list, embeddings, dim=100):
    matrix, token_count, known_count, empty_count = [], 0, 0, 0
    for tokens in tokenized_list:
        known = sum(word in embeddings for word in tokens)
        token_count += len(tokens)
        known_count += known
        empty_count += int(known == 0)
        matrix.append(doc_vector(tokens, embeddings, dim))
    if not matrix:
        raise ValueError("不能对空文档集构建特征。")
    return np.stack(matrix), {
        "documents": len(matrix),
        "empty_documents": empty_count,
        "tokens": token_count,
        "known_tokens": known_count,
        "oov_tokens": token_count - known_count,
        "coverage": known_count / token_count if token_count else 0.0,
    }
