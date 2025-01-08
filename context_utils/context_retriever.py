import faiss
import numpy as np
import pandas as pd


class ContextRetriever:
    def __init__(self, embedding_file, model, top_k=3):
        """
        초기화 시 모델을 외부에서 전달받도록 변경.
        """
        self.top_k = top_k
        self.knowledge_db = pd.read_pickle(embedding_file)
        self.embedding_dimension = len(self.knowledge_db['Question Embedding'][0])

        # FAISS Index 초기화
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        embeddings = np.array(self.knowledge_db['Question Embedding'].tolist(), dtype='float32')
        self.index.add(embeddings)

        # 외부에서 전달된 모델 사용
        self.model = model

    def get_related_contexts(self, question):
        """질문에 관련된 컨텍스트를 검색"""
        question_embedding = self.model.encode(question).reshape(1, -1).astype('float32')
        distances, indices = self.index.search(question_embedding, self.top_k)

        # 검색 결과 출력
        print("\n🔎 [Faiss 검색 결과] ------------------")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            print(f"Rank {i + 1}: Distance = {dist}\n  Question = {self.knowledge_db.iloc[idx]['question']}\n  Context = {self.knowledge_db.iloc[idx]['Additional Context'][:30]}")
        print("------------------\n")

        # 검색된 컨텍스트 추출 ( ### 현재는 1순위만 추출 하도록 함. ###)
        related_contexts = self.knowledge_db.iloc[indices[0][:1]]['Additional Context'].tolist()
        return "\n".join(related_contexts)