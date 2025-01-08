import os
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from context_utils.context_retriever import ContextRetriever


# 병렬 처리 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_data(file_path):
    """엑셀 파일을 읽어 DataFrame으로 반환하는 함수"""
    data = pd.read_excel(file_path)
    print("데이터 로드 완료!")
    print(data.head())
    return data

def process_with_openai(data, base_url, context_retriever):
    """OpenAI 클라이언트를 사용해 요청을 처리하는 함수"""
    results = []

    for index, row in data.iterrows():

        print(f"\n🔄 처리 중인 질문 (ID: {row['id']}): {row['question']}")

        # 기존 add_feature 함수 사용하여 메시지 구조 생성
        message_structure = add_feature(row["id"], row["question"], context_retriever)

        # OpenAI 클라이언트 설정
        client = OpenAI(
            base_url=base_url,
            api_key="dummy-key",
            default_headers={
                "Content-Type": "application/json",
                "Question-ID": str(row["id"])
            }
        )

        try:
            response = client.chat.completions.create(
                model="olympiad",
                messages=message_structure["message"],
                temperature=0.0,
                max_tokens=None,
                stream=False
            )

            # response를 dictionary로 변환
            response_dict = response.model_dump()
            result = response_dict.get('result', {})

            print(f"\n✅ ID {row['id']} 처리 완료")
            print(f"응답: {result.get('response', '')}")

            results.append({
                'id': row['id'],
                'question': row['question'],
                'prompt': result.get('prompt', ''),
                'context': result.get('context', ''),
                'response': result.get('response', ''),
                'score': result.get('score', 0),
                'reasoning': result.get('reasoning', '')
            })

        except Exception as e:
            print(f"ID {row['id']} 처리 중 에러 발생: {str(e)}")


    # 결과를 DataFrame으로 변환하고 엑셀로 저장
    results_df = pd.DataFrame(results)
    results_df.to_excel('response_results.xlsx', index=False)

def add_feature(id, question, context_retriever):
    """
    메시지 구조를 생성하는 함수
    - 이 함수는 학생들이 필요에 따라 시스템 메시지나 사용자 메시지를 추가로 정의하도록 설계되었습니다.
    - 아래의 system_prompt와 user_message를 수정하여 커스터마이즈하세요.

    :param id: int/str, ID
    :param question: str, 질문
    :return: dict, 메시지 JSON 구조
    """
    # -----------------수정--------------------#
    system_prompt = """
Persona:
You are an expert in artificial intelligence, machine learning, data analysis, and IT trends.

Instructions:
	1.	Base your response strictly on the provided question and sources. Keep it accurate and concise.
	2.	Do not include information outside the provided sources.
	3.	Exclude topics or details not explicitly mentioned in the sources.
	4.	If a specific format is requested, adhere to it strictly.
	5.	Include examples or cases only if explicitly mentioned in the sources.

Limit:
    1. Responses must not exceed 1,500 characters.
    2. Your response must be written in Korean.
    """

    # -----------------수정--------------------#
    # 사용자 메시지 생성 (add_rag 함수에서 추가 처리)
    user_message = add_rag(question, context_retriever)

    # 메시지 구조 반환
    message = {
        "id": id,
        "message": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    }
    return message


def add_rag(question, context_retriever):
    """
    질문에 추가 정보를 결합하는 함수
    - 이 함수는 학생들이 RAG(Retrieval-Augmented Generation)를 구현하거나, 질문에 추가 정보를 삽입하도록 설계되었습니다.
    - context를 활용해 질문에 정보를 추가하세요.

    :param question: str, 질문
    :return: str, 수정된 질문
    """
    # -----------------수정--------------------#
    """
    수정사항
    - 외부에서 전달받은 ContextRetriever를 사용합니다. (모델을 한번만 로드하기 위함)
    - FAISS를 사용해 질문과 관련된 컨텍스트를 검색하고 결합하도록 만들었습니다.
        - 질문의 벡터값과 컨텍스트 (배경지식)이 연결되어있음.
        - 질문이 들어오면, 해당 질문의 벡터값과 인덱싱되어있는 질문의 벡터값과 비교 후 해당 컨텍스트를 리턴
    """
    # ContextRetriever 초기화 (임베딩 파일 경로 지정)
    context = context_retriever.get_related_contexts(question)

    # -----------------수정--------------------#
    # 질문에 컨텍스트 추가 (예: "<BEGIN SOURCE>" 형식으로 데이터 결합)
    if context:
        # UI와 같은 형식 유지를 위한 변경 불가
        question = question + '\n\nYou may use the following sources if needed to answer the user\'s question. If you don\'t know the answer, say "I don\'t know."\n\n<BEGIN SOURCE>' + context
    return question


# 메인 실행 부분
if __name__ == "__main__":
    file_path = './data/problem.xlsx'
    embedding_file = './data/embeddings.pkl'
    base_url = "https://ryeon.elpai.org/submit/v1"

    # 데이터 로드
    data = load_data(file_path)

    # 모델 로드
    print("\n🔄 SentenceTransformer 모델 로드 중...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ 모델 로드 완료!")

    # ContextRetriever 초기화
    context_retriever = ContextRetriever(embedding_file, model)

    process_with_openai(data, base_url, context_retriever)