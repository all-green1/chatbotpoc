
from typing import Any
from chonkie import Pipeline

def chunk_it(size: int, dir: str) -> Any:
    docs = (Pipeline()
    .fetch_from("file", dir=dir, ext=[".md", ".txt"])
    .process_with("text")
    .chunk_with("recursive", chunk_size=size)
    .run())

    return docs
    