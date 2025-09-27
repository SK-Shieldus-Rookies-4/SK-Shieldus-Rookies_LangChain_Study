import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 🌟 Mac OS 환경에서 FAISS/OpenMP 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    # 0. API 키 로드
    load_dotenv()
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

    if not UPSTAGE_API_KEY:
        print("❌ 오류: UPSTAGE_API_KEY가 없습니다. .env 파일을 확인하세요.")
        return

    # 0단계: 문서 로드
    FILE_PATH = "../data/콘텐츠분쟁해결_사례.pdf"
    print("✅ 0단계: 문서 로드 시작...")
    loader = PyPDFLoader(FILE_PATH)
    documents = loader.load()
    print(f"   -> {len(documents)} 페이지 로드 완료")

    # 1단계: 문서 분할 설정
    print("\n✅ 1단계: 문서 분할 시작...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,       # 권장 범위: 1200~1800자
        chunk_overlap=300,     # 200~400자 권장
        separators=[
            "\n【사건개요】",
            "\n【쟁점사항】",
            "\n【처리경위】",
            "\n【처리결과】",
            "\n■", "\n\n", "\n", ".", " ", ""
        ]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"   -> 분할된 문서 조각 수: {len(split_docs)}")

    # 2단계: 임베딩 모델 설정
    print("\n✅ 2단계: 임베딩 모델 생성...")
    embeddings = UpstageEmbeddings(
        model="solar-embedding-1-large",
        upstage_api_key=UPSTAGE_API_KEY
    )
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    print("   -> 벡터스토어 생성 완료")

    # 3단계: 검색기 설정
    print("\n✅ 3단계: 검색기 설정...")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    print("   -> 검색기 설정 완료 (k=5)")

    # 4단계: LLM 설정
    print("\n✅ 4단계: LLM 설정...")
    llm = ChatUpstage(
        model="solar-pro",
        base_url="https://api.upstage.ai/v1",
        temperature=0.2,
        upstage_api_key=UPSTAGE_API_KEY
    )
    print("   -> LLM 설정 완료")

    # 5단계: 법률 자문 프롬프트 작성
    print("\n✅ 5단계: 프롬프트 작성...")
    prompt_template = """
당신은 콘텐츠 분야 전문 법률 자문가입니다. 
아래 분쟁조정 사례들을 바탕으로 정확하고 전문적인 법률 조언을 제공해주세요.

관련 분쟁사례:
{context}

상담 내용: {query}

답변 가이드라인:
1. 제시된 사례들을 근거로 답변하세요
2. 관련 법령이나 조항이 있다면 명시하세요
3. 비슷한 사례의 처리경위와 결과를 참고하여 설명하세요
4. 실무적 해결방안을 단계별로 제시하세요
5. 사례에 없는 내용은 "제시된 사례집에서는 확인할 수 없습니다"라고 명시하세요

전문 법률 자문:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "query"]
    )
    print("   -> 프롬프트 작성 완료")

    # 6단계: QA 체인 생성
    print("\n✅ 6단계: QA 체인 생성...")
    

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        input_key="query",   # 외부에서 넣는 키
        chain_type_kwargs={
            "question_key": "query"   # 내부 combine_documents_chain에서 쓸 키
        }
    )
    print("   -> QA 체인 생성 완료")
    print("QA Chain input keys:", qa_chain.input_keys)

    # 7단계: 테스트 질문
    test_questions = [
        "온라인 게임에서 시스템 오류로 아이템이 사라졌는데, 게임회사가 복구를 거부하고 있습니다. 어떻게 해결할 수 있나요?",
        "인터넷 강의를 중도 해지하려고 하는데 과도한 위약금을 요구받고 있습니다. 정당한가요?",
        "무료체험 후 자동으로 유료전환되어 요금이 청구되었습니다. 환불 가능한가요?",
        "미성년자가 부모 동의 없이 게임 아이템을 구매했습니다. 환불받을 수 있는 방법이 있나요?",
        "온라인 교육 서비스가 광고와 다르게 제공되어 계약을 해지하고 싶습니다. 가능한가요?"
    ]

    print("\n✅ 7단계: 테스트 질문 실행...")
    for q in test_questions:
        print("\n===================================================")
        print(f"🔍 질문: {q}")
        print("===================================================")
        result = qa_chain.invoke({"query": q})
        print("✅ [전문 법률 조언]")
        print(result["result"])

        print("\n📜 [참조 문서]")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"  {i}. 페이지: {doc.metadata.get('page', 'Unknown')}")


    # 8단계: 선택 - 분쟁 유형 분류 함수
    def classify_dispute_type(query):
        game_keywords = ["게임", "아이템", "계정", "캐릭터", "레벨", "길드", "온라인게임"]
        elearning_keywords = ["강의", "온라인교육", "이러닝", "수강", "환불", "화상교육"]
        web_keywords = ["웹사이트", "무료체험", "자동결제", "구독", "사이트"]

        query_lower = query.lower()

        if any(keyword in query_lower for keyword in game_keywords):
            return "게임"
        elif any(keyword in query_lower for keyword in elearning_keywords):
            return "이러닝"
        elif any(keyword in query_lower for keyword in web_keywords):
            return "웹콘텐츠"
        else:
            return "기타"

    # 분쟁 유형 테스트
    print("\n✅ 8단계: 분쟁 유형 분류 테스트...")
    for q in test_questions:
        print(f"'{q}' → {classify_dispute_type(q)}")


if __name__ == "__main__":
    main()