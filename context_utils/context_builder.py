# import faiss
# import numpy as np
# import pandas as pd
#
# class ContextBuilder:
#     def __init__(self, embedding_file, top_k=3):
#         """
#         벡터 데이터베이스와 임베딩 데이터를 초기화합니다.
#         :param embedding_file: str, 저장된 임베딩 데이터 파일 경로 (.pkl)
#         :param top_k: int, 검색할 상위 유사도 결과 수
#         """
#         self.top_k = top_k
#         self.knowledge_db = pd.read_pickle(embedding_file)
#
#         # FAISS Index 초기화
#         embedding_dimension = len(self.knowledge_db['Embedding'][0])
#         self.index = faiss.IndexFlatL2(embedding_dimension)
#
#         # 벡터 추가
#         embeddings = np.array(self.knowledge_db['Embedding'].tolist(), dtype='float32')
#         self.index.add(embeddings)
#
#     def add_context(self, question):
#         """
#         질문 임베딩을 생성하고 벡터 데이터베이스에서 관련 컨텍스트를 검색.
#         :param question: str, 사용자 질문
#         :return: str, 질문에 추가된 컨텍스트
#         """
#         # 질문을 임베딩으로 변환 (Hugging Face 모델 사용)
#         question_embedding = self._generate_embedding(question)
#
#         # 벡터 데이터베이스에서 검색
#         distances, indices = self.index.search(np.array([question_embedding], dtype='float32'), self.top_k)
#
#         # 검색 결과 출력
#         print("\n[검색 결과] ------------------")
#         for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
#             print(f"Rank {i + 1}: Distance = {dist}, Context = {self.knowledge_db.iloc[idx]['Additional Context']}")
#         print("------------------\n")
#
#         # 컨텍스트 결합
#         related_contexts = self.knowledge_db.iloc[indices[0]]['Additional Context'].tolist()
#         combined_context = "\n".join(related_contexts)
#         return f"{question}\n\n<BEGIN CONTEXT>\n{combined_context}\n<END CONTEXT>"
#
#     @staticmethod
#     def _generate_embedding(text):
#         """
#         질문 텍스트를 임베딩으로 변환합니다.
#         :param text: str, 임베딩을 생성할 텍스트
#         :return: list, 임베딩 벡터
#         """
#         from sentence_transformers import SentenceTransformer
#         model = SentenceTransformer('all-MiniLM-L6-v2')
#         return model.encode(text).tolist()