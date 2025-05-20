import hashlib
import time
import asyncio
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone
from tqdm import tqdm

# NLTK 필요 데이터 다운로드 (처음 한 번만 실행)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# API 키 설정
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

# 디렉토리 경로 설정
folder_list = ["충치", "치석", "치주염", "구강일반"]
directory_path = "./data/"

def generate_hash():
    """고유한 ID를 생성합니다."""
    return hashlib.md5(str(time.time()).encode()).hexdigest()

class StopwordsAwareEmbeddings(Embeddings):
    """불용어가 제거된 텍스트 임베딩 클래스"""
    def __init__(self, base_embeddings):
        self.base_embeddings = base_embeddings
    
    def embed_documents(self, docs):
        # Document 객체 리스트인 경우 메타데이터의 processed_text 사용
        if isinstance(docs[0], Document):
            texts_to_embed = [doc.metadata.get('processed_text', doc.page_content) for doc in docs]
        else:
            # 문자열 리스트인 경우 직접 전처리
            texts_to_embed = [preprocess_english_text(text) for text in docs]
        
        return self.base_embeddings.embed_documents(texts_to_embed)
    
    def embed_query(self, query):
        # 쿼리도 동일하게 전처리
        processed_query = preprocess_english_text(query)
        return self.base_embeddings.embed_query(processed_query)

def preprocess_english_text(text):
    """영어 텍스트에서 불용어 제거 및 기본 전처리"""
    # 특수 문자 및 숫자 제거 (알파벳과 공백만 유지)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 여러 공백을 하나로 치환
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 단어 토큰화
    tokens = word_tokenize(text)
    
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 표제어 추출 (Lemmatization)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # 토큰을 다시 텍스트로 결합
    processed_text = ' '.join(tokens)
    
    return processed_text

def preprocess_documents_with_stopwords(docs):
    """문서 리스트에서 간단히 불용어 제거 및 전처리"""
    processed_docs = []
    
    for doc in docs:
        # 텍스트 전처리
        processed_text = preprocess_english_text(doc.page_content)
        
        # 메타데이터 복사 후 전처리된 텍스트 추가
        metadata = doc.metadata.copy()
        metadata['processed_text'] = processed_text
        
        # 원본 텍스트 유지, 전처리 텍스트는 메타데이터에 저장
        processed_docs.append(Document(
            page_content=doc.page_content,
            metadata=metadata
        ))
    
    return processed_docs

def upsert_documents_to_pinecone(
    index,
    namespace,
    documents,
    sparse_encoder,
    embeddings,
    batch_size=32
):
    """문서를 Pinecone 인덱스에 업서트합니다."""
    # 텍스트 및 메타데이터 추출
    contents = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    # 전처리된 텍스트 가져오기 (BM25 인코딩용)
    processed_texts = [doc.metadata.get('processed_text', doc.page_content) for doc in documents]
    
    for i in tqdm(range(0, len(contents), batch_size), desc=f"Upserting to {namespace}"):
        i_end = min(i + batch_size, len(contents))
        
        # 배치 준비
        content_batch = contents[i:i_end]
        metadata_batch = metadatas[i:i_end]
        processed_text_batch = processed_texts[i:i_end]
        
        # ID 생성
        ids = [f"{namespace}_{generate_hash()}" for _ in range(i, i_end)]
        
        # Dense 임베딩 생성
        dense_embeds = embeddings.embed_documents(content_batch)
        
        # Sparse 임베딩 생성
        sparse_embeds = sparse_encoder.encode_documents(processed_text_batch)
        
        # 벡터 준비
        vectors = [
            {
                "id": _id,
                "values": dense,
                "sparse_values": sparse,
                "metadata": {**metadata, "text": content}
            }
            for _id, content, metadata, dense, sparse in zip(
                ids, content_batch, metadata_batch, dense_embeds, sparse_embeds
            )
        ]
        
        # 업서트
        index.upsert(vectors=vectors, namespace=namespace)
    
    print(f"{namespace}: 총 {len(contents)}개 문서를 하이브리드 벡터로 저장 완료")

async def main(folder):
    # 문서 로드
    print(f"{folder} 문서 로딩 중...")
    loader = DirectoryLoader(
        directory_path,
        glob=f"**/*{folder}*.docx",  # 폴더명을 포함한 .docx 파일 로드
        show_progress=True
    )
    
    all_docs = loader.load()
    print(f"{folder}: 총 {len(all_docs)}개 문서 로드 완료")
    
    if not all_docs:
        print(f"{folder}에 로드된 문서가 없습니다.")
        return
    
    # 간단한 불용어 처리
    print(f"{folder}: 불용어 처리 중...")
    preprocessed_docs = preprocess_documents_with_stopwords(all_docs)
    
    # 문서 분할
    print(f"{folder}: 문서 분할 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(preprocessed_docs)
    print(f"{folder}: 총 {len(splits)}개의 청크로 분할 완료")
    
    # 폴더 정보를 메타데이터에 추가
    for split in splits:
        split.metadata['folder'] = folder
    
    # Pinecone 인덱스 연결
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # 인덱스 존재 확인 및 생성 (모든 폴더에서 공통으로 사용할 인덱스)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"인덱스 '{PINECONE_INDEX_NAME}' 생성 중...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # OpenAI embedding 차원
            metric="cosine"
        )
        print(f"인덱스 생성됨. 준비 대기 중...")
        time.sleep(60)  # 인덱스 생성 대기
    
    # Pinecone 인덱스 연결
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # OpenAI Embeddings (Small 모델) - Dense Vector용
    print(f"{folder}: 임베딩 준비 중...")
    base_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    custom_embeddings = StopwordsAwareEmbeddings(base_embeddings)
    
    # BM25 인코더 준비
    print(f"{folder}: Sparse 벡터 인코더 생성 중...")
    processed_texts = [split.metadata.get('processed_text', split.page_content) for split in splits]
    bm25 = BM25Encoder(language="english")
    bm25.fit(processed_texts)
    
    # 문서를 Pinecone에 업서트
    print(f"{folder}: 문서를 Pinecone에 업서트 중...")
    upsert_documents_to_pinecone(
        index=index,
        namespace=folder,
        documents=splits,
        sparse_encoder=bm25,
        embeddings=custom_embeddings,
        batch_size=32
    )
    
    # 인덱스 통계 확인
    stats = index.describe_index_stats()
    print(f"인덱스 '{PINECONE_INDEX_NAME}' 통계:")
    print(f"- 전체 벡터 수: {stats['total_vector_count']}")
    print(f"- '{folder}' 네임스페이스 벡터 수: {stats['namespaces'].get(folder, {}).get('vector_count', 0)}")

async def process_all_folders():
    """모든 폴더를 비동기적으로 처리"""
    tasks = []
    for folder in folder_list:
        task = asyncio.create_task(main(folder))
        tasks.append(task)
    
    # 모든 태스크 완료 대기
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    # 모든 폴더 데이터 처리 및 저장
    asyncio.run(process_all_folders())
    