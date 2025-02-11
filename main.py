import os
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# 상단에 설정 상수 추가
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_SEPARATOR = "\n"

# 수정된 코드:
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
os.environ["OPENAI_API_KEY"] = api_key

# PDF 파일 로드
def load_pdf(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {file_path}")
            
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        if not text.strip():
            raise ValueError("PDF에서 텍스트를 추출할 수 없습니다.")
            
        return text
    except Exception as e:
        raise Exception(f"PDF 로딩 중 오류 발생: {str(e)}")

# 텍스트 분할 및 임베딩 생성
def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator=DEFAULT_SEPARATOR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    return knowledge_base

# 질문에 답변
def ask_question(knowledge_base, question):
    docs = knowledge_base.similarity_search(question)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.invoke(input={"input_documents": docs, "question": question})  # 변경된 부분
    return response

# 메인 함수
def main():
    # PDF 파일 경로
    pdf_path = "C:/langchain/aibook/pubtext.pdf"
    
    # PDF 파일 로드
    text = load_pdf(pdf_path)
    
    # 텍스트 처리 및 임베딩 생성
    knowledge_base = process_text(text)
    
    # 질문 예시
    question = "What is the main topic of this document?"
    
    # 질문에 답변
    answer = ask_question(knowledge_base, question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
