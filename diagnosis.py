"""
-치주염 부분 수정 필요
namespace:"pd 설정"의 vector store retrive를 통해 관련 문서를 참고해
pd(치주염 진행도(0~4))를 설정할건데 0~1은 양호 2~4는 병원방문을 요구해서 이 두 구간으로 나누는 것에 집중해야 함

프로세스는 
1. sparse vector를 통해 먼저 top_n개의 문서들을 뽑고
2. 원본 이미지를 prompt에 넣어 진행 정도를 text화 한다.
3. 그리고 해당 진행 정도가 어느 단계(0~4)에 속하는지 정하고
4. 이에 대한 피드백(치석과 충치의 경우와 같이)

+

각 충치, 치석 agent 실행 시 원본 이미지와 마스킹된 이미지가 전달되도록 해야할 듯
"""
import os
import json
import mysql.connector
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime, date
import asyncio
import base64
import cv2
import numpy as np

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from hsv import imageDiagnosis

# NLTK 데이터 확인
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# 환경 변수 설정
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_USER = os.environ.get("MYSQL_USER")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
MYSQL_DATABASE = "pet_dental_care"

class DentalDiagnosisState(TypedDict):
    """전체 상태 관리를 위한 State 클래스"""
    pet_id: int
    image_path: str
    messages: List[Dict[str, Any]]
    pet_info: Dict[str, Any]
    diagnosis_results: Dict[str, Any]
    masked_images: Dict[str, str]  # 마스킹된 이미지 경로들
    retrieved_contexts: Dict[str, List[str]]
    periodontitis_stage: int  # 치주염 진행도 (0-4)
    final_feedback: Dict[str, str]
    next_agent: str

def preprocess_english_text(text):
    """영어 텍스트에서 불용어 제거 및 기본 전처리 (preprocess.py와 동일)"""
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

def encode_image_to_base64(image_path: str) -> str:
    """이미지를 base64로 인코딩"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_image_array(image_array: np.ndarray, save_path: str) -> str:
    """이미지 배열을 파일로 저장하고 경로 반환"""
    try:
        cv2.imwrite(save_path, image_array)
        return save_path
    except Exception as e:
        print(f"이미지 저장 실패 ({save_path}): {e}")
        return ""

class MySQLAgent:
    """MySQL에서 펫 정보를 조회하는 Agent"""
    
    def __init__(self):
        self.connection_config = {
            'host': MYSQL_HOST,
            'user': MYSQL_USER,
            'password': MYSQL_PASSWORD,
            'database': MYSQL_DATABASE
        }
    
    def get_pet_info(self, pet_id: int) -> Dict[str, Any]:
        """펫 정보 조회"""
        try:
            conn = mysql.connector.connect(**self.connection_config)
            cursor = conn.cursor(dictionary=True)
            
            query = """
            SELECT breed, weight, birth_date, gender 
            FROM pets 
            WHERE pet_id = %s
            """
            
            cursor.execute(query, (pet_id,))
            result = cursor.fetchone()
            
            if result:
                # birth_date에서 나이 계산
                if result['birth_date']:
                    today = date.today()
                    birthdate = result['birth_date']
                    age_years = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
                    result['age_years'] = age_years
                
                return result
            else:
                return {}
                
        except mysql.connector.Error as e:
            print(f"MySQL 연결 오류: {e}")
            return {}
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

class ImageAnalysisAgent:    
    """이미지 분석을 수행하는 Agent (마스킹된 이미지 생성 포함)"""
    
    def analyze_dental_image(self, image_path: str) -> Dict[str, Any]:
        """치아 이미지 분석 수행 및 마스킹된 이미지 생성"""
        try:
            # imageDiagnosis 클래스 사용
            diagnosis = imageDiagnosis(image_path)
            
            # 전체 치아 영역 감지 및 이미지 저장
            teeth_result_img = diagnosis.detect_total_teeth()
            teeth_save_path = f"{image_path}_teeth_full_area.jpg"
            save_image_array(teeth_result_img, teeth_save_path)
            
            # 충치 감지 및 이미지 저장
            caries_result_img = diagnosis.detect_dental_caries()
            caries_save_path = f"{image_path}_caries_detected.jpg"
            save_image_array(caries_result_img, caries_save_path)
            
            # 치석 감지 및 이미지 저장
            calculus_result_img = diagnosis.detect_dental_calculus()
            calculus_save_path = f"{image_path}_calculus_detected.jpg"
            save_image_array(calculus_result_img, calculus_save_path)
            
            # 비율 계산
            results = diagnosis.calculate_ratios()
            
            # 마스킹된 이미지 경로들 생성
            masked_images = {
                "teeth_area": teeth_save_path,
                "caries": caries_save_path,
                "calculus": calculus_save_path
            }
            
            return {
                "total_tooth_area": results["total_tooth_area"],
                "caries_area": results["caries_area"],
                "calculus_area": results["calculus_area"],
                "caries_ratio": results["caries_ratio"],
                "calculus_ratio": results["calculus_ratio"],
                "masked_images": masked_images,
                "analysis_success": True
            }
            
        except Exception as e:
            print(f"이미지 분석 오류: {e}")
            return {
                "analysis_success": False,
                "error": str(e),
                "masked_images": {}
            }

class PineconeRetrievalAgent:
    """Pinecone에서 Sparse-first 하이브리드 검색을 수행하는 Agent"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # BM25 인코더 초기화
        self.bm25_encoder = BM25Encoder(language="english")
    
    def sparse_first_retrieve(self, query: str, namespace: str, top_k: int = 10) -> List[str]:
        """Sparse vector 우선 검색 (치주염 전용)"""
        try:
            # 쿼리 전처리
            processed_query = preprocess_english_text(query)
            
            # Sparse 임베딩 생성
            try:
                sparse_vector = self.bm25_encoder.encode_queries([processed_query])[0]
                
                # Sparse vector만으로 먼저 검색
                results = self.index.query(
                    sparse_vector=sparse_vector,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True,
                    alpha=0.0  # 완전 sparse 검색
                )
                
            except Exception as sparse_error:
                print(f"Sparse 벡터 검색 실패, Dense로 대체: {sparse_error}")
                # Sparse 실패시 Dense로 대체
                dense_vector = self.embeddings.embed_query(processed_query)
                results = self.index.query(
                    vector=dense_vector,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True
                )
            
            # 텍스트 추출
            contexts = []
            for match in results['matches']:
                if 'text' in match['metadata']:
                    contexts.append(match['metadata']['text'])
                    
            print(f"Sparse-first 검색 완료 ({namespace}): {len(contexts)}개 컨텍스트 발견")
            return contexts
            
        except Exception as e:
            print(f"Sparse-first 검색 오류 ({namespace}): {e}")
            return []
    
    def hybrid_retrieve(self, query: str, namespace: str, top_k: int = 3, alpha: float = 0.6) -> List[str]:
        """하이브리드 검색 (Dense + Sparse)"""
        try:
            # 쿼리 전처리
            processed_query = preprocess_english_text(query)
            
            # Dense 임베딩 생성
            dense_vector = self.embeddings.embed_query(processed_query)
            
            # Sparse 임베딩 생성
            try:
                sparse_vector = self.bm25_encoder.encode_queries([processed_query])[0]
                
                # 하이브리드 검색
                results = self.index.query(
                    vector=dense_vector,
                    sparse_vector=sparse_vector,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True,
                    alpha=alpha
                )
            except Exception as sparse_error:
                print(f"Sparse 벡터 생성 실패, Dense만 사용: {sparse_error}")
                results = self.index.query(
                    vector=dense_vector,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True
                )
            
            # 텍스트 추출
            contexts = []
            for match in results['matches']:
                if 'text' in match['metadata']:
                    contexts.append(match['metadata']['text'])
                    
            print(f"하이브리드 검색 완료 ({namespace}): {len(contexts)}개 컨텍스트 발견")
            return contexts
            
        except Exception as e:
            print(f"하이브리드 검색 오류 ({namespace}): {e}")
            return []

class PeriodontitisStageAgent:
    """치주염 진행도를 판정하는 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    def assess_periodontitis_stage(self, image_path: str, contexts: List[str], pet_info: Dict) -> int:
        """이미지와 컨텍스트를 기반으로 치주염 진행도 판정 (0-4)"""
        
        # 이미지를 base64로 인코딩
        image_base64 = encode_image_to_base64(image_path)
        
        # 관련 문서 텍스트 결합
        context_text = "\n\n".join(contexts[:3])  # 상위 3개 문서만 사용
        
        stage_prompt = ChatPromptTemplate.from_template(
            """You are a veterinary periodontitis specialist analyzing a pet's oral cavity image. Focus on VISUAL FEATURES to accurately assess the periodontitis progression stage (PD) from 0-4.

Pet Information:
- Breed: {breed}
- Age: {age} years old
- Weight: {weight} kg
- Gender: {gender}

Professional Reference Materials:
{context}

VISUAL ASSESSMENT CRITERIA - Look for these specific features in the image:

**Stage 0 (Healthy) - VISUAL MARKERS:**
LOOK FOR:
- GUM COLOR: Pale pink to coral pink, uniform coloration
- GUM TEXTURE: Firm, stippled surface like orange peel
- GUM MARGINS: Thin, knife-edge appearance around teeth
- TEETH: Clean, white/ivory colored surfaces
- PLAQUE/TARTAR: Minimal to none visible
- BLEEDING: No visible blood or dark spots

**Stage 1 (Gingivitis) - VISUAL MARKERS:**
LOOK FOR:
- GUM COLOR: Bright red or deep pink, especially at gum line
- GUM TEXTURE: Smooth, glossy appearance (loss of stippling)
- GUM MARGINS: Swollen, rounded, puffy appearance
- INFLAMMATION: Red line or band along tooth-gum junction
- PLAQUE: Yellow/white sticky deposits on teeth
- BLEEDING: May see dark red spots or blood traces

**Stage 2 (Early Periodontitis) - VISUAL MARKERS:**
LOOK FOR:
- GUM COLOR: Dark red, purple, or bluish discoloration
- GUM RECESSION: Gums pulling away from teeth, exposing tooth roots
- POCKET FORMATION: Dark spaces/gaps between teeth and gums
- TARTAR: Yellow-brown hard deposits, especially near gum line
- TEETH: May appear longer due to gum recession
- MOBILITY: Teeth may appear slightly displaced

**Stage 3 (Moderate Periodontitis) - VISUAL MARKERS:**
LOOK FOR:
- SEVERE RECESSION: Significant gum tissue loss, root exposure
- DEEP POCKETS: Large dark gaps between teeth and gums
- HEAVY TARTAR: Thick brown/black crusty deposits
- TOOTH MOBILITY: Visibly loose or tilted teeth
- ABSCESSES: Swollen bumps, pus, or dark fluid-filled areas
- BONE LOSS: Teeth appearing "long" with exposed roots

**Stage 4 (Advanced Periodontitis) - VISUAL MARKERS:**
LOOK FOR:
- EXTENSIVE TISSUE LOSS: Large areas of missing gum tissue
- SEVERE MOBILITY: Teeth obviously loose, displaced, or missing
- HEAVY CALCULUS: Thick, dark tartar covering most tooth surfaces
- ULCERATIONS: Open wounds, raw tissue, or white/yellow lesions
- INFECTIONS: Pus discharge, abscesses, severe swelling
- BONE EXPOSURE: Visible jawbone or deep tissue damage

IMAGE ANALYSIS INSTRUCTIONS:
1. Examine the ENTIRE visible oral cavity systematically
2. Focus on GUM COLOR and TEXTURE changes
3. Look for TARTAR/PLAQUE accumulation patterns  
4. Assess GUM-TOOTH JUNCTION carefully
5. Note any ASYMMETRY between different areas
6. Consider PET'S AGE and BREED-SPECIFIC factors

VISUAL SEVERITY SCALE:
- Stages 0-1: Healthy to mild visual changes (GOOD CONDITION)
- Stages 2-4: Moderate to severe visual pathology (VETERINARY VISIT REQUIRED)

Based on your VISUAL ANALYSIS of the image, respond ONLY in this format:

STAGE: [one number from 0, 1, 2, 3, 4]
VISUAL_FINDINGS: [describe what you specifically see in the image - colors, textures, deposits, etc.]
DESCRIPTION: [brief clinical interpretation in 2-3 lines]
RECOMMENDATION: [Good condition/Veterinary visit required]"""
        )
        
        try:
            # 이미지와 함께 프롬프트 전송
            response = self.llm.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": stage_prompt.format(
                        breed=pet_info.get('breed', '미상'),
                        age=pet_info.get('age_years', '미상'),
                        weight=pet_info.get('weight', '미상'),
                        gender=pet_info.get('gender', '미상'),
                        context=context_text
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ])
            ])
            
            response_text = response.content
            
            # STAGE 추출
            stage_match = re.search(r'STAGE:\s*(\d)', response_text)
            if stage_match:
                stage = int(stage_match.group(1))
                if 0 <= stage <= 4:
                    # VISUAL_FINDINGS 추출 (선택적)
                    visual_findings_match = re.search(r'VISUAL_FINDINGS:\s*([^\n]+)', response_text)
                    visual_findings = visual_findings_match.group(1) if visual_findings_match else "No specific findings noted"
                    
                    print(f"Periodontitis stage assessed: Stage {stage}")
                    print(f"Visual findings: {visual_findings}")
                    return stage
            
            print("Periodontitis stage assessment failed, returning default stage 2")
            return 2  # 기본값
            
        except Exception as e:
            print(f"Periodontitis stage assessment error: {e}")
            return 2  # 기본값

class FeedbackGenerationAgent:
    """최종 피드백을 생성하는 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    def generate_condition_feedback(self, condition: str, pet_info: Dict, diagnosis_results: Dict, 
                                  contexts: List[str], original_image: str, masked_image: str) -> str:
        """각 질환별 피드백 생성 (원본+마스킹 이미지 포함)"""
        
        # 마스킹된 이미지 파일이 존재하는지 확인
        if not os.path.exists(masked_image):
            print(f"Warning: Masked image not found: {masked_image}")
            masked_image = original_image  # 원본 이미지로 대체
        
        # 이미지들을 base64로 인코딩
        original_b64 = encode_image_to_base64(original_image)
        masked_b64 = encode_image_to_base64(masked_image)
        
        context_text = "\n".join(contexts[:2])
        
        if condition == "충치":
            ratio = diagnosis_results.get('caries_ratio', 0)
            prompt_text = f"""
            As a veterinary dental specialist, please provide feedback about dental caries in 3 lines.

Pet Information: Breed {pet_info.get('breed', 'Unknown')}, {pet_info.get('age_years', 'Unknown')} years old, {pet_info.get('weight', 'Unknown')}kg
Caries Ratio: {ratio:.2f}%

Professional Knowledge: {context_text}

Please provide 3 lines of practical advice based on the original image and caries detection image in Korean."""
            
        elif condition == "치석":
            ratio = diagnosis_results.get('calculus_ratio', 0)
            prompt_text = f"""
            As a veterinary dental specialist, please provide feedback about dental calculus in 3 lines.

Pet Information: Breed {pet_info.get('breed', 'Unknown')}, {pet_info.get('age_years', 'Unknown')} years old, {pet_info.get('weight', 'Unknown')}kg
Calculus Ratio: {ratio:.2f}%

Professional Knowledge: {context_text}

Please provide 3 lines of practical advice based on the original image and calculus detection image in Korean."""
        
        try:
            response = self.llm.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{masked_b64}"}}
                ])
            ])
            return response.content
            
        except Exception as e:
            print(f"{condition} feedback generation error: {e}")
            return f"{condition} 피드백을 생성할 수 없습니다."
    
    def generate_periodontitis_feedback(self, pet_info: Dict, stage: int, contexts: List[str]) -> str:
        """치주염 진행도별 피드백 생성"""
        
        context_text = "\n".join(contexts[:2])
        
        if stage <= 1:
            severity = "양호"
            action = "현재 상태 유지에 집중"
            detailed_action = "정기적인 홈케어와 예방관리를 통해 건강한 상태를 유지하세요."
        else:
            severity = "병원 방문 필요"
            action = "수의사 상담 권장"
            detailed_action = "전문적인 치료가 필요한 단계이므로 가능한 빨리 동물병원을 방문하세요."
        
        prompt = ChatPromptTemplate.from_template(
            """As a veterinary periodontitis specialist, please provide feedback about periodontitis progression in 3 lines.

Pet Information: Breed {breed}, {age} years old, {weight}kg
Periodontitis Stage: Stage {stage} ({severity})
Recommended Action: {action}
Detailed Guidance: {detailed_action}

Professional Knowledge:
{context}

Based on the periodontitis stage assessment, please provide 3 lines of clear and practical advice in Korean that addresses:
1. Current condition explanation
2. Immediate care recommendations  
3. Long-term management strategy"""
        )
        
        try:
            response = self.llm.invoke(
                prompt.format(
                    breed=pet_info.get('breed', '미상'),
                    age=pet_info.get('age_years', '미상'),
                    weight=pet_info.get('weight', '미상'),
                    stage=stage,
                    severity=severity,
                    action=action,
                    detailed_action=detailed_action,
                    context=context_text
                )
            )
            print(f"치주염 진행도 평가: {stage}단계 ({severity})")
            return response.content
            
        except Exception as e:
            print(f"치주염 피드백 생성 오류: {e}")
            return "치주염 피드백을 생성할 수 없습니다."

class DentalDiagnosisSupervisor:
    """전체 워크플로우를 관리하는 Supervisor"""
    
    def __init__(self):
        self.mysql_agent = MySQLAgent()
        self.image_agent = ImageAnalysisAgent()
        self.retrieval_agent = PineconeRetrievalAgent()
        self.periodontitis_agent = PeriodontitisStageAgent()
        self.feedback_agent = FeedbackGenerationAgent()
        
        # LangGraph 설정
        self.workflow = StateGraph(DentalDiagnosisState)
        self._setup_workflow()
    
    def _setup_workflow(self):
        """워크플로우 노드 및 엣지 설정"""
        
        # 노드 추가
        self.workflow.add_node("get_pet_info", self._get_pet_info_node)
        self.workflow.add_node("analyze_image", self._analyze_image_node)
        self.workflow.add_node("retrieve_contexts", self._retrieve_contexts_node)
        self.workflow.add_node("assess_periodontitis", self._assess_periodontitis_node)
        self.workflow.add_node("generate_feedback", self._generate_feedback_node)
        
        # 시작점 설정
        self.workflow.set_entry_point("get_pet_info")
        
        # 엣지 연결
        self.workflow.add_edge("get_pet_info", "analyze_image")
        self.workflow.add_edge("analyze_image", "retrieve_contexts")
        self.workflow.add_edge("retrieve_contexts", "assess_periodontitis")
        self.workflow.add_edge("assess_periodontitis", "generate_feedback")
        self.workflow.add_edge("generate_feedback", END)
        
        # 그래프 컴파일
        self.app = self.workflow.compile()
    
    def _get_pet_info_node(self, state: DentalDiagnosisState) -> DentalDiagnosisState:
        """펫 정보 조회 노드"""
        print(f"펫 정보 조회 중... (ID: {state['pet_id']})")
        
        pet_info = self.mysql_agent.get_pet_info(state['pet_id'])
        state['pet_info'] = pet_info
        
        print(f"펫 정보: {pet_info}")
        return state
    
    def _analyze_image_node(self, state: DentalDiagnosisState) -> DentalDiagnosisState:
        """이미지 분석 노드"""
        print(f"이미지 분석 중... ({state['image_path']})")
        
        diagnosis_results = self.image_agent.analyze_dental_image(state['image_path'])
        state['diagnosis_results'] = diagnosis_results
        state['masked_images'] = diagnosis_results.get('masked_images', {})
        
        print(f"진단 결과: 충치 {diagnosis_results.get('caries_ratio', 0):.1f}%, 치석 {diagnosis_results.get('calculus_ratio', 0):.1f}%")
        return state
    
    def _retrieve_contexts_node(self, state: DentalDiagnosisState) -> DentalDiagnosisState:
        """컨텍스트 검색 노드"""
        print("관련 전문 지식 검색 중...")
        
        contexts = {}
        pet_info = state['pet_info']
        diagnosis = state['diagnosis_results']
        
        # 품종과 나이를 포함한 기본 쿼리 (영어)
        breed = pet_info.get('breed', 'dog')
        age = pet_info.get('age_years', 'adult')
        base_query = f"{breed} {age} years old pet dog"
        
        # 충치 검색 (충치 비율이 있을 때만)
        if diagnosis.get('caries_ratio', 0) > 0:
            caries_query = f"{base_query} dental caries treatment management prevention cavity tooth decay"
            contexts['충치'] = self.retrieval_agent.hybrid_retrieve(caries_query, '충치', top_k=3, alpha=0.6)
            print(f"충치 검색 완료: {len(contexts['충치'])}개 문서")
        
        # 치석 검색 (치석 비율이 있을 때만)
        if diagnosis.get('calculus_ratio', 0) > 0:
            calculus_query = f"{base_query} dental calculus tartar removal prevention scaling plaque"
            contexts['치석'] = self.retrieval_agent.hybrid_retrieve(calculus_query, '치석', top_k=3, alpha=0.6)
            print(f"치석 검색 완료: {len(contexts['치석'])}개 문서")
        
        # 치주염 검색 (항상 실행 - Sparse-first 방식)
        periodontitis_query = f"{base_query} periodontitis gum disease progression stage assessment"
        contexts['pd'] = self.retrieval_agent.sparse_first_retrieve(periodontitis_query, 'pd', top_k=10)
        print(f"치주염 검색 완료: {len(contexts['pd'])}개 문서")
        
        state['retrieved_contexts'] = contexts
        return state
    
    def _assess_periodontitis_node(self, state: DentalDiagnosisState) -> DentalDiagnosisState:
        """치주염 진행도 판정 노드"""
        print("치주염 진행도 판정 중...")
        
        pd_contexts = state['retrieved_contexts'].get('pd', [])
        stage = self.periodontitis_agent.assess_periodontitis_stage(
            state['image_path'], 
            pd_contexts, 
            state['pet_info']
        )
        
        state['periodontitis_stage'] = stage
        print(f"치주염 진행도: {stage}단계")
        return state
    
    def _generate_feedback_node(self, state: DentalDiagnosisState) -> DentalDiagnosisState:
        """피드백 생성 노드"""
        print("맞춤형 피드백 생성 중...")
        
        feedback = {}
        pet_info = state['pet_info']
        diagnosis = state['diagnosis_results']
        contexts = state['retrieved_contexts']
        masked_images = state['masked_images']
        original_image = state['image_path']
        
        # 충치 피드백
        if diagnosis.get('caries_ratio', 0) > 0 and '충치' in contexts:
            feedback['충치'] = self.feedback_agent.generate_condition_feedback(
                '충치', pet_info, diagnosis, contexts['충치'], 
                original_image, masked_images.get('caries', '')
            )
        
        # 치석 피드백  
        if diagnosis.get('calculus_ratio', 0) > 0 and '치석' in contexts:
            feedback['치석'] = self.feedback_agent.generate_condition_feedback(
                '치석', pet_info, diagnosis, contexts['치석'],
                original_image, masked_images.get('calculus', '')
            )
        
        # 치주염 피드백 (진행도 기반)
        stage = state['periodontitis_stage']
        if 'pd' in contexts:
            feedback['치주염'] = self.feedback_agent.generate_periodontitis_feedback(
                pet_info, stage, contexts['pd']
            )
        
        state['final_feedback'] = feedback
        print("피드백 생성 완료")
        return state
    
    async def diagnose(self, pet_id: int, image_path: str) -> Dict[str, Any]:
        """전체 진단 프로세스 실행"""
        
        initial_state = DentalDiagnosisState(
            pet_id=pet_id,
            image_path=image_path,
            messages=[],
            pet_info={},
            diagnosis_results={},
            masked_images={},
            retrieved_contexts={},
            periodontitis_stage=0,
            final_feedback={},
            next_agent=""
        )
        
        # 워크플로우 실행
        final_state = await self.app.ainvoke(initial_state)
        
        # 결과 정리
        result = {
            "pet_info": final_state['pet_info'],
            "diagnosis_results": final_state['diagnosis_results'],
            "periodontitis_stage": final_state['periodontitis_stage'],
            "masked_images": final_state['masked_images'],
            "feedback": final_state['final_feedback']
        }
        
        return result

# 사용 예시
async def main():
    # Supervisor 초기화
    supervisor = DentalDiagnosisSupervisor()
    
    # 진단 실행
    pet_id = 1  # 실제 펫 ID
    image_path = "teeth3.jpg"  # 실제 이미지 경로
    
    try:
        result = await supervisor.diagnose(pet_id, image_path)
        
        print("\n" + "="*60)
        print("개선된 치아 진단 결과 (이미지 기반 치주염 진행도 포함)")
        print("="*60)
        
        # 펫 정보 출력
        pet_info = result['pet_info']
        print(f"\n 반려동물 정보:")
        print(f"   품종: {pet_info.get('breed', '미상')}")
        print(f"   나이: {pet_info.get('age_years', '미상')}세")
        print(f"   체중: {pet_info.get('weight', '미상')}kg")
        print(f"   성별: {pet_info.get('gender', '미상')}")
        
        # 진단 결과 출력
        diagnosis = result['diagnosis_results']
        stage = result['periodontitis_stage']
        print(f"\n 진단 결과:")
        print(f"   충치 비율: {diagnosis.get('caries_ratio', 0):.2f}%")
        print(f"   치석 비율: {diagnosis.get('calculus_ratio', 0):.2f}%")
        print(f"   치주염 진행도: {stage}단계 ({'양호' if stage <= 1 else '병원방문 권장'})")
        
        # 생성된 마스킹 이미지들
        masked_imgs = result['masked_images']
        print(f"\n 생성된 분석 이미지:")
        for img_type, path in masked_imgs.items():
            if os.path.exists(path):
                print(f" {img_type}: {path}")
            else:
                print(f" {img_type}: {path} (파일 없음)")
        
        # 피드백 출력
        feedback = result['feedback']
        print(f"\n 전문가 피드백:")
        for condition, advice in feedback.items():
            print(f"\n   [{condition}]")
            print(f"   {advice}")
        
    except Exception as e:
        print(f"진단 중 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(main())