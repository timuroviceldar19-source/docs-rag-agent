from fastembed import TextEmbedding


class FastEmbedLocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model = TextEmbedding(model_name=model_name)
        # detect actual dimension by embedding one token
        sample = list(self._model.embed(["x"]))
        self.vector_size: int = len(sample[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]
