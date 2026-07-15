from __future__ import annotations

import os
from typing import List, Sequence, TYPE_CHECKING, Union

from loguru import logger
import sentry_sdk
import onnxruntime as ort
from transformers import AutoTokenizer


if TYPE_CHECKING:
    import numpy as np


class GTETextEmbedder:
    """
    Embedder for gte-multilingual-base exported ONNX.

    Enforced layout:
      - model_path points to: .../models/gte-multilingual-base-onnx
      - ONNX file: ...<model_path>/gte-multilingual-base.onnx
      - tokenizer dir: ...<model_path>/tokenizer/
    """

    ONNX_FILENAME = "gte-multilingual-base.onnx"
    TOKENIZER_SUBDIR = "tokenizer"
    VECTOR_SIZE = 768

    def __init__(
        self,
        model_path: Union[str, os.PathLike],
        vector_size: int,
        document_prefix: str = "",
        query_prefix: str = "",
        max_length: int = 2048,
        batch_size: int = 32,
    ) -> None:
        self.model_path = str(model_path)
        self.vector_size = int(vector_size)
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.max_length = max_length
        self.batch_size = batch_size

        if self.vector_size != self.VECTOR_SIZE:
            raise ValueError(
                f"gte-multilingual-base embedder expects vector_size={self.VECTOR_SIZE}, got {self.vector_size}"
            )

        model_dir = os.path.expanduser(self.model_path)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"model_path must be an existing directory, got: {model_dir}")

        onnx_path = os.path.join(model_dir, self.ONNX_FILENAME)
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(
                f"Expected ONNX file not found: {onnx_path}\n"
                f"embedder2 is strict and only supports {self.ONNX_FILENAME} in the model_path directory."
            )

        tokenizer_dir = os.path.join(model_dir, self.TOKENIZER_SUBDIR)
        if not os.path.isdir(tokenizer_dir):
            raise FileNotFoundError(
                f"Expected tokenizer directory not found: {tokenizer_dir}\n"
                f"Place tokenizer files under: {tokenizer_dir}/"
            )

        required = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"]
        missing = [f for f in required if not os.path.isfile(os.path.join(tokenizer_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"Tokenizer directory is missing required files: {missing}\n"
                f"Tokenizer dir: {tokenizer_dir}"
            )

        logger.info(f"Initializing GTETextEmbedder ONNX: {onnx_path}")
        logger.info(f"Initializing GTETextEmbedder tokenizer: {tokenizer_dir}")
        logger.info(f"GTETextEmbedder config: max_length={self.max_length}, batch_size={self.batch_size}")

        try:
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.error("Failed to initialize GTETextEmbedder")
            sentry_sdk.capture_exception(e)
            raise

        input_names = [i.name for i in self.session.get_inputs()]
        if "input_ids" not in input_names or "attention_mask" not in input_names:
            raise RuntimeError(
                f"Unexpected ONNX inputs: {input_names}. Expected at least ['input_ids', 'attention_mask']."
            )

        logger.success("GTETextEmbedder initialized successfully")

    def prepare_document(self, document: str) -> str:
        return f"{self.document_prefix}{document}"

    def prepare_query(self, query: str) -> str:
        return f"{self.query_prefix}{query}"

    def embed(self, docs: Sequence[str]) -> List["np.ndarray"]:
        import numpy as np

        if not docs:
            return []

        results: List[np.ndarray] = []
        total = len(docs)

        for start in range(0, total, self.batch_size):
            batch_docs = list(docs[start: start + self.batch_size])

            enc = self.tokenizer(
                batch_docs,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            input_ids = enc.get("input_ids")
            attention_mask = enc.get("attention_mask")

            if input_ids is None or attention_mask is None:
                raise RuntimeError(
                    f"Tokenizer did not return required keys. Got: {sorted(enc.keys())}, "
                    f"expected: input_ids, attention_mask"
                )

            if input_ids.dtype != np.int64:
                input_ids = input_ids.astype(np.int64)
            if attention_mask.dtype != np.int64:
                attention_mask = attention_mask.astype(np.int64)

            out = self.session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
            emb = np.asarray(out[0])

            if emb.ndim != 2 or emb.shape[1] != self.vector_size:
                raise RuntimeError(
                    f"Unexpected embedding output shape {emb.shape}, expected [batch, {self.vector_size}]"
                )

            for i in range(emb.shape[0]):
                results.append(emb[i])

            logger.debug(f"Embedded batch {start}-{min(start + self.batch_size, total)} / {total}")

        return results

    def embed_one(self, doc: str) -> "np.ndarray":
        return self.embed([doc])[0]
