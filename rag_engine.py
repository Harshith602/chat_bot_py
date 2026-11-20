
import os
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGEngine:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.chunks = self.load_and_chunk()
        self.embeddings = self.model.encode(
            self.chunks,
            convert_to_numpy=True
        )

    def load_and_chunk(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Knowledge base not found: {self.file_path}"
            )

        with open(self.file_path, "r", encoding="utf-8") as file:
            text = file.read()

        words = text.split()
        chunk_size = 180
        overlap = 40
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    def retrieve(self, question, top_k=3):
        if not self.chunks:
            return []

        question_embedding = self.model.encode(
            [question],
            convert_to_numpy=True
        )[0]

        scores = self.embeddings @ question_embedding
        scores /= (
            np.linalg.norm(self.embeddings, axis=1)
            * np.linalg.norm(question_embedding)
            + 1e-10
        )

        best_indices = np.argsort(scores)[::-1][:top_k]

        return [self.chunks[i] for i in best_indices]
