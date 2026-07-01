"""
Embedding 模块 — 使用 bge-large-zh-v1.5
封装为单例。云端 Docker 构建时已预下载模型到 /root/.cache/huggingface，
运行时通过 local_files_only=True 从本地加载，秒级就绪。
"""

import os
import logging
from typing import Optional

log = logging.getLogger("kb-embedder")

# 运行时仅用本地缓存（模型在 Docker 构建阶段已下载）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_MODEL_NAME = "BAAI/bge-large-zh-v1.5"  # 1024-dim
_model_instance: Optional["Embedder"] = None


def _get_device() -> str:
    """选择计算设备。优先 MPS，出问题时回退 CPU。"""
    device = os.environ.get("KB_EMBEDDING_DEVICE", "")
    if device:
        return device
    # 服务器端默认用 CPU
    return "cpu"


class Embedder:
    """单例 Embedding 编码器"""

    def __init__(self, model_name: str = _MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or _get_device()
        self._model = None
        self._dim = None

    def _load(self):
        if self._model is not None:
            return
        log.info(f"加载 Embedding 模型: {self.model_name} (device={self.device}) ...")
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self.model_name, device=self.device, local_files_only=True
        )
        try:
            self._dim = self._model.get_embedding_dimension()
        except AttributeError:
            self._dim = self._model.get_sentence_embedding_dimension()
        log.info(f"Embedding 模型就绪，维度: {self._dim}")

    @property
    def dim(self) -> int:
        self._load()
        return self._dim

    def encode(self, text: str) -> list[float]:
        """对单段文本编码，返回浮点数列表"""
        self._load()
        # bge 模型 best practice: 查询需加前缀
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def encode_query(self, query: str) -> list[float]:
        """对查询文本编码（bge 模型查询需加 instruction prefix）"""
        self._load()
        embedding = self._model.encode(
            query, normalize_embeddings=True, prompt="为这个句子生成表示以用于检索相关文章："
        )
        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码"""
        self._load()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


def get_embedder() -> Embedder:
    """获取全局单例 Embedder"""
    global _model_instance
    if _model_instance is None:
        _model_instance = Embedder()
    return _model_instance
