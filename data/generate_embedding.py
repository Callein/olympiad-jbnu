from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle

def generate_embedding(text, model):
    """
    Hugging Face의 SentenceTransformer를 사용하여 텍스트 임베딩 생성.

    :param text: str, 임베딩을 생성할 텍스트
    :param model: SentenceTransformer, 사전 로드된 모델
    :return: list, 생성된 임베딩 벡터
    """
    return model.encode(text).tolist()

if __name__ == "__main__":
    # 모델 로드 (Hugging Face의 사전 학습된 모델)
    model_name = "all-MiniLM-L6-v2"  # 경량화된 무료 모델
    model = SentenceTransformer(model_name)

    # 데이터 로드
    knowledge_db_path = '../data/knowledge_db.xlsx'
    knowledge_db = pd.read_excel(knowledge_db_path)

    # NaN 처리
    knowledge_db['Additional Context'] = knowledge_db['Additional Context'].fillna("No context available")
    knowledge_db['question'] = knowledge_db['question'].fillna("")

    # question 임베딩 생성
    knowledge_db['Question Embedding'] = knowledge_db['question'].apply(
        lambda x: generate_embedding(x, model)
    )

    # 임베딩 저장
    output_path = '../data/embeddings.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(knowledge_db, f)

    print(f"임베딩 데이터가 {output_path}에 저장되었습니다!")