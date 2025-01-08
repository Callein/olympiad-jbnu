# import faiss
# import numpy as np
# import pandas as pd
#
# class VectorDatabase:
#     def __init__(self, embedding_file):
#         self.knowledge_db = pd.read_pickle(embedding_file)
#         self.dimension = len(self.knowledge_db.iloc[0]['Embedding'])
#         self.index = faiss.IndexFlatL2(self.dimension)
#         embeddings = np.array(self.knowledge_db['Embedding'].tolist(), dtype='float32')
#         self.index.add(embeddings)
#
#     def search(self, question_embedding, top_k=3):
#         """질문 임베딩으로 가장 유사한 컨텍스트 검색"""
#         distances, indices = self.index.search(np.array([question_embedding], dtype='float32'), top_k)
#         return self.knowledge_db.iloc[indices[0]]