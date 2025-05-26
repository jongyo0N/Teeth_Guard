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
    guardian_id: int
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
    
    def get_pet_info(self, guardian_id: int, pet_id: int) -> Dict[str, Any]:
        """펫 정보 조회 (guardian_id와 pet_id 모두 사용)"""
        try:
            conn = mysql.connector.connect(**self.connection_config)
            cursor = conn.cursor(dictionary=True)
            
            query = """
            SELECT p.name, p.breed, p.weight, p.birth_date, p.gender,
                   g.name as guardian_name, g.experience_level
            FROM pets p
            JOIN guardians g ON p.guardian_id = g.guardian_id
            WHERE p.guardian_id = %s AND p.pet_id = %s
            """
            
            cursor.execute(query, (guardian_id, pet_id))
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
            # 이미지 파일 존재 확인
            if not os.path.exists(image_path):
                print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
                return {
                    "analysis_success": False,
                    "error": f"Image file not found: {image_path}",
                    "masked_images": {},
                    "caries_ratio": 0,
                    "calculus_ratio": 0
                }
            
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
                "masked_images": {},
                "caries_ratio": 0,
                "calculus_ratio": 0
            }

class DenseOnlyRetrievalAgent:
    """Pinecone에서 Dense 검색만 수행하는 Agent (BM25/Sparse 완전 제거)"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("Dense 전용 검색 에이전트 초기화 완료")
    
    def retrieve_documents(self, query: str, namespace: str, top_k: int = 3) -> List[str]:
        """Dense 벡터 검색만 수행"""
        try:
            print(f"Dense 검색 시작 - Query: '{query[:50]}...', Namespace: {namespace}")
            
            # 쿼리 전처리
            processed_query = preprocess_english_text(query)
            print(f"전처리된 쿼리: '{processed_query[:50]}...'")
            
            # Dense 임베딩 생성
            dense_vector = self.embeddings.embed_query(processed_query)
            print(f"Dense 벡터 생성 완료 (차원: {len(dense_vector)})")
            
            # Dense 검색만 수행 (Sparse 없음)
            results = self.index.query(
                vector=dense_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            
            # 텍스트 추출
            contexts = []
            for i, match in enumerate(results['matches']):
                if 'text' in match['metadata']:
                    contexts.append(match['metadata']['text'])
                    print(f"문서 {i+1} - 유사도: {match['score']:.4f}")
                    
            print(f"Dense 검색 완료 ({namespace}): {len(contexts)}개 문서 발견")
            return contexts
            
        except Exception as e:
            print(f"Dense 검색 오류 ({namespace}): {e}")
            return []

class PeriodontitisStageAgent:
    """치주염 진행도를 판정하는 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    def assess_periodontitis_stage(self, image_path: str, contexts: List[str], pet_info: Dict) -> int:
        """이미지와 컨텍스트를 기반으로 치주염 진행도 판정 (0-4)"""
        
        # 이미지 파일 존재 확인
        if not os.path.exists(image_path):
            print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            return 2  # 기본값
        
        # 이미지를 base64로 인코딩
        image_base64 = encode_image_to_base64(image_path)
        
        # 관련 문서 텍스트 결합
        context_text = "\n\n".join(contexts[:3])  # 상위 3개 문서만 사용
        
        stage_prompt = ChatPromptTemplate.from_template(
            """You are a veterinary periodontitis(PD) specialist analyzing a pet's oral cavity image. Assess the periodontitis progression stage from 0-4 based on visual findings.

Pet Information:
- Breed: {breed}
- Age: {age} years old
- Weight: {weight} kg
- Gender: {gender}

Professional Reference Materials:
{context}

VISUAL ASSESSMENT CRITERIA:

**Stage 0 (Healthy):**
- Gum color: Pale pink to coral pink, uniform
- Gum texture: Firm, stippled surface 
- Teeth: Clean, white/ivory colored
- Plaque/tartar: Minimal to none
- No bleeding or inflammation signs

**Stage 1 (Gingivitis):**
- Gum color: Bright red, especially at gum line
- Gum texture: Smooth, glossy (loss of stippling)
- Mild plaque: Yellow/white deposits
- Slight swelling along gum margins

**Stage 2 (Early Periodontitis):**
- Gum color: Dark red, purple discoloration
- Mild gum recession exposing tooth roots
- Tartar: Yellow-brown deposits
- Small pockets between teeth and gums

**Stage 3 (Moderate Periodontitis):**
- Significant gum recession and root exposure
- Heavy tartar: Thick brown/black deposits
- Visible tooth mobility or displacement
- Deep pockets between teeth and gums

**Stage 4 (Advanced Periodontitis):**
- Extensive tissue loss and root exposure
- Heavy calculus covering most tooth surfaces
- Severe tooth mobility or missing teeth
- Visible bone loss or deep tissue damage

EVALUATION FOCUS:
- Examine entire visible oral cavity systematically
- Consider pet's age and breed-specific factors
- Focus on gum color, texture, and tartar accumulation
- Assess tooth-gum junction carefully

Respond in this exact format:

STAGE: [0, 1, 2, 3, or 4]
VISUAL_FINDINGS: [describe specific observations from the image]
CLINICAL_NOTES: [brief 2-3 line assessment]
MANAGEMENT_TYPE: [Home care suitable / Professional care recommended]"""
        )
        
        try:
            # 이미지와 함께 프롬프트 전송
            response = self.llm.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": stage_prompt.format(
                        breed=pet_info.get('breed', 'Unknown'),
                        age=pet_info.get('age_years', 'Unknown'),
                        weight=pet_info.get('weight', 'Unknown'),
                        gender=pet_info.get('gender', 'Unknown'),
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
                    findings_match = re.search(r'VISUAL_FINDINGS:\s*([^\n]+)', response_text)
                    findings = findings_match.group(1) if findings_match else "No specific findings noted"
                    
                    print(f"치주염 진행도 평가: {stage}단계")
                    print(f"관찰소견: {findings}")
                    return stage
            
            print("치주염 진행도 평가 실패, 기본값 2단계로 설정")
            return 2  # 기본값
            
        except Exception as e:
            print(f"치주염 진행도 평가 오류: {e}")
            return 2  # 기본값

class ThreeSentenceFeedbackAgent:
    """3문장 제한 피드백 생성 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    def generate_condition_feedback(self, condition: str, pet_info: Dict, diagnosis_results: Dict, 
                                  contexts: List[str], original_image: str, masked_image: str) -> str:
        """각 질환별 피드백 생성 (정확히 3문장)"""
        
        # 이미지 파일 존재 확인
        if not os.path.exists(original_image):
            print(f"Warning: Original image not found: {original_image}")
            return f"{condition} 피드백을 생성할 수 없습니다. (원본 이미지 없음)"
        
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
You are a veterinary dental(cavity) specialist. Based on the information below, provide educational feedback about dental caries.

Pet Information:
- Breed: {pet_info.get('breed', 'Unknown')}
- Age: {pet_info.get('age_years', 'Unknown')} years old
- Weight: {pet_info.get('weight', 'Unknown')}kg
- Detected caries ratio: {ratio:.1f}%

Professional Knowledge Base:
{context_text}

CRITICAL REQUIREMENT: Respond with EXACTLY 3 sentences in Korean. No more, no less.

Feedback Structure (Each must be one complete sentence):
Explain the current caries(cavity) condition detected in this specific pet
Describe what causes dental caries(cavity) in pets and common symptoms(of canine cavity) owners should recognize
Provide one key consideration or insight for pet owners to understand about this condition(cavities and caries)


Response Guidelines:

- Write in natural, conversational Korean language
- Keep each sentence informative but concise
- Focus on education rather than treatment recommendations
- Maintain a reassuring but informative tone
- Each sentence should be complete and meaningful
- don't write numbers infront of sentencese

Analyze both the original image and caries detection image, then respond with exactly sequenced 3 sentences in Korean."""
            
        elif condition == "치석":
            ratio = diagnosis_results.get('calculus_ratio', 0)
            prompt_text = f"""
You are a veterinary dental(calculus,dental cleaning) specialist. Based on the information below, provide educational feedback about teeth cleaning and dental calculus.

Pet Information:
- Breed: {pet_info.get('breed', 'Unknown')}
- Age: {pet_info.get('age_years', 'Unknown')} years old
- Weight: {pet_info.get('weight', 'Unknown')}kg
- Detected calculus ratio: {ratio:.1f}%

Professional Knowledge Base:
{context_text}

CRITICAL REQUIREMENT: Respond with EXACTLY 3 sentences in Korean. No more, no less.

Feedback Structure (Each must be one complete sentence):
Explain the current calculus(tooth color, resorption),halitosis condition detected in this specific pet
Describe how dental calculus forms and progresses in pets
Provide one key consideration or insight for pet owners to understand about this condition

Response Guidelines:
- Write in natural, conversational Korean language
- Keep each sentence informative but concise
- Focus on education and treatment recommendations
- Maintain a reassuring but informative tone
- Each sentence should be complete and meaningful
- don't write numbers infront of sentencese

Analyze both the original image and calculus detection image, then respond with exactly sequenced 3 sentences in Korean."""
        
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
            print(f"{condition} 피드백 생성 오류: {e}")
            return f"{condition} 피드백을 생성할 수 없습니다."
    
    def generate_periodontitis_feedback(self, pet_info: Dict, stage: int, contexts: List[str]) -> str:
        """치주염 진행도별 피드백 생성 (정확히 3문장)"""
        
        context_text = "\n".join(contexts[:2])
        
        # 진행도에 따른 정보 제공 방향 설정
        if stage <= 1:
            condition_focus = "healthy to mild gingivitis condition"
            information_type = "preventive education and maintenance guidance"
        elif stage == 2:
            condition_focus = "early periodontitis stage"
            information_type = "progression understanding and management education"
        else:
            condition_focus = "moderate to advanced periodontitis stage"
            information_type = "comprehensive condition education and awareness"
        
        prompt_text = f"""
You are a veterinary periodontitis(gum) specialist. Based on the information below, provide educational feedback about gum health, dental plaque and periodontitis progression.

Pet Information:
- Breed: {pet_info.get('breed', 'Unknown')}
- Age: {pet_info.get('age_years', 'Unknown')} years old
- Weight: {pet_info.get('weight', 'Unknown')}kg
- Periodontitis stage: Stage {stage}

Condition Focus: {condition_focus}
Information Type: {information_type}

Professional Knowledge Base:
{context_text}

CRITICAL REQUIREMENT: Respond with EXACTLY 3 sentences in Korean. No more, no less.

Feedback Structure (Each must be one complete sentence):
Explain what Stage {stage} periodontitis,and color of dog's gum mean for this pet's oral health status
Describe how periodontitis develops and progresses in pets, including key factors
Provide one key consideration or insight,prevention for pet owners to understand about managing this condition

Response Guidelines:
- Write in natural, conversational Korean language
- Keep each sentence informative but concise
- Focus on education and treatment directives
- Maintain a reassuring but informative tone
- Each sentence should be complete and meaningful
- Avoid excessive medical terminology
- don't write numbers infront of sentencese

Respond with exactly sequenced 3 sentences in Korean."""
        
        try:
            response = self.llm.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": prompt_text}
                ])
            ])
            
            condition_desc = "양호한 상태" if stage <= 1 else "관리 필요 상태"
            print(f"치주염 진행도 평가: {stage}단계 ({condition_desc})")
            return response.content
            
        except Exception as e:
            print(f"치주염 피드백 생성 오류: {e}")
            return "치주염 피드백을 생성할 수 없습니다."
        
class DentalDiagnosisSupervisor:
    """전체 워크플로우를 관리하는 Supervisor (Dense 전용)"""
    
    def __init__(self):
        self.mysql_agent = MySQLAgent()
        self.image_agent = ImageAnalysisAgent()
        self.retrieval_agent = DenseOnlyRetrievalAgent()
        self.periodontitis_agent = PeriodontitisStageAgent()
        self.feedback_agent = ThreeSentenceFeedbackAgent()
        
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
        print(f"펫 정보 조회 중... (Guardian ID: {state['guardian_id']}, Pet ID: {state['pet_id']})")
        
        pet_info = self.mysql_agent.get_pet_info(state['guardian_id'], state['pet_id'])
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
        """컨텍스트 검색 노드 (Dense 전용)"""
        print("관련 전문 지식 검색 중... (Dense 전용)")
        
        contexts = {}
        pet_info = state['pet_info']
        diagnosis = state['diagnosis_results']
        
        # 품종과 나이를 포함한 기본 정보 (영어)
        breed = pet_info.get('breed', 'dog').lower()
        age = pet_info.get('age_years', 5)
        
        # 나이대별 구분
        if age < 2:
            age_group = "young puppy"
        elif age < 7:
            age_group = "adult dog"
        else:
            age_group = "senior dog"
        
        base_info = f"{breed} {age_group}"
        
        # 충치 검색 (충치 비율이 0.5% 이상일 때만)
        if diagnosis.get('caries_ratio', 0) >= 0.5:
            print("충치 관련 문서 검색 중...")
            caries_queries = [
                f"{base_info} dental caries tooth decay cavity treatment",
                f"canine dental caries causes symptoms progression {breed}"
            ]
            
            all_caries_contexts = []
            for i, query in enumerate(caries_queries):
                print(f"  충치 쿼리 {i+1}: {query}")
                contexts_batch = self.retrieval_agent.retrieve_documents(query, '충치', top_k=2)
                all_caries_contexts.extend(contexts_batch)
            
            # 중복 제거 및 상위 3개만 선택
            contexts['충치'] = list(dict.fromkeys(all_caries_contexts))[:3]
            print(f"충치 검색 완료: {len(contexts['충치'])}개 문서")
        
        # 치석 검색 (치석 비율이 1% 이상일 때만)
        if diagnosis.get('calculus_ratio', 0) >= 1.0:
            print("치석 관련 문서 검색 중...")
            calculus_queries = [
                f"{base_info} dental calculus tartar buildup removal",
                f"canine dental tartar plaque formation {breed}"
            ]
            
            all_calculus_contexts = []
            for i, query in enumerate(calculus_queries):
                print(f"  치석 쿼리 {i+1}: {query}")
                contexts_batch = self.retrieval_agent.retrieve_documents(query, '치석', top_k=2)
                all_calculus_contexts.extend(contexts_batch)
            
            # 중복 제거 및 상위 3개만 선택
            contexts['치석'] = list(dict.fromkeys(all_calculus_contexts))[:3]
            print(f"치석 검색 완료: {len(contexts['치석'])}개 문서")
        
        # 치주염 검색 (항상 실행)
        print("치주염 관련 문서 검색 중...")
        periodontitis_queries = [
            f"{base_info} periodontitis gum disease progression stages",
            f"canine periodontal disease assessment {breed} {age_group}",
            f"dog gum disease stages symptoms causes"
        ]
        
        # 치주염은 더 많은 컨텍스트 수집 (진행도 판정용)
        all_pd_contexts = []
        for i, query in enumerate(periodontitis_queries):
            print(f"  치주염 쿼리 {i+1}: {query}")
            # namespace 확인하여 적절한 값 사용
            try:
                contexts_batch = self.retrieval_agent.retrieve_documents(query, 'pd', top_k=2)
            except:
                # 'pd' namespace가 없을 경우 다른 namespace 시도
                try:
                    contexts_batch = self.retrieval_agent.retrieve_documents(query, '치주염', top_k=2)
                except:
                    contexts_batch = self.retrieval_agent.retrieve_documents(query, '구강일반', top_k=2)
            all_pd_contexts.extend(contexts_batch)
        
        # 중복 제거 및 상위 5개 선택 (치주염 진행도 판정을 위해 더 많은 컨텍스트)
        contexts['치주염'] = list(dict.fromkeys(all_pd_contexts))[:5]
        print(f"치주염 검색 완료: {len(contexts['치주염'])}개 문서")
        
        state['retrieved_contexts'] = contexts
        return state
    
    def _assess_periodontitis_node(self, state: DentalDiagnosisState) -> DentalDiagnosisState:
        """치주염 진행도 판정 노드"""
        print("치주염 진행도 판정 중...")
        
        pd_contexts = state['retrieved_contexts'].get('치주염', [])
        stage = self.periodontitis_agent.assess_periodontitis_stage(
            state['image_path'], 
            pd_contexts, 
            state['pet_info']
        )
        
        state['periodontitis_stage'] = stage
        print(f"치주염 진행도: {stage}단계")
        return state
    
    def _generate_feedback_node(self, state: DentalDiagnosisState) -> DentalDiagnosisState:
        """피드백 생성 노드 (3문장 제한)"""
        print("맞춤형 피드백 생성 중... (각 3문장)")
        
        feedback = {}
        pet_info = state['pet_info']
        diagnosis = state['diagnosis_results']
        contexts = state['retrieved_contexts']
        masked_images = state['masked_images']
        original_image = state['image_path']
        
        # 충치 피드백
        if diagnosis.get('caries_ratio', 0) > 0 and '충치' in contexts:
            print("  충치 피드백 생성 중...")
            feedback['충치'] = self.feedback_agent.generate_condition_feedback(
                '충치', pet_info, diagnosis, contexts['충치'], 
                original_image, masked_images.get('caries', '')
            )
        
        # 치석 피드백  
        if diagnosis.get('calculus_ratio', 0) > 0 and '치석' in contexts:
            print("  치석 피드백 생성 중...")
            feedback['치석'] = self.feedback_agent.generate_condition_feedback(
                '치석', pet_info, diagnosis, contexts['치석'],
                original_image, masked_images.get('calculus', '')
            )
        
        # 치주염 피드백 (진행도 기반)
        print("  치주염 피드백 생성 중...")
        stage = state['periodontitis_stage']
        if '치주염' in contexts:
            feedback['치주염'] = self.feedback_agent.generate_periodontitis_feedback(
                pet_info, stage, contexts['치주염']
            )
        
        state['final_feedback'] = feedback
        print("피드백 생성 완료")
        return state
    
    async def diagnose(self, guardian_id: int, pet_id: int, image_path: str) -> Dict[str, Any]:
        """전체 진단 프로세스 실행"""
        
        initial_state = DentalDiagnosisState(
            guardian_id=guardian_id,
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
            "guardian_id": final_state['guardian_id'],
            "pet_id": final_state['pet_id'],
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
    
    # 진단 실행 (guardian_id와 pet_id 모두 필요)
    guardian_id = 1  # 실제 guardian ID
    pet_id = 1  # 실제 pet ID
    image_path = "teeth3.jpeg"  # 실제 이미지 경로
    
    # 현재 디렉토리에서 이미지 파일 찾기
    possible_paths = [
        "teeth3.jpeg",
        "./teeth3.jpeg", 
        "images/teeth3.jpeg",
        "data/teeth3.jpeg"
    ]
    
    found_image = None
    for path in possible_paths:
        if os.path.exists(path):
            found_image = path
            break
    
    if not found_image:
        print("이미지 파일을 찾을 수 없습니다. 다음 위치들을 확인했습니다:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\n올바른 이미지 경로를 확인해주세요.")
        return
    
    print(f"이미지 파일 발견: {found_image}")
    
    try:
        result = await supervisor.diagnose(guardian_id, pet_id, found_image)
        
        print("\n" + "="*60)
        print("DENSE 전용 치아 진단 결과 (3문장 피드백)")
        print("="*60)
        
        # 기본 정보 출력
        print(f"\n진단 대상:")
        print(f"   Guardian ID: {result['guardian_id']}")
        print(f"   Pet ID: {result['pet_id']}")
        
        # 펫 정보 출력
        pet_info = result['pet_info']
        print(f"\n반려동물 정보:")
        print(f"   이름: {pet_info.get('name', '미상')}")
        print(f"   품종: {pet_info.get('breed', '미상')}")
        print(f"   나이: {pet_info.get('age_years', '미상')}세")
        print(f"   체중: {pet_info.get('weight', '미상')}kg")
        print(f"   성별: {pet_info.get('gender', '미상')}")
        print(f"   보호자: {pet_info.get('guardian_name', '미상')}")
        print(f"   보호자 경험도: {pet_info.get('experience_level', '미상')}")
        
        # 진단 결과 출력
        diagnosis = result['diagnosis_results']
        stage = result['periodontitis_stage']
        print(f"\n진단 결과:")
        print(f"   충치 비율: {diagnosis.get('caries_ratio', 0):.2f}%")
        print(f"   치석 비율: {diagnosis.get('calculus_ratio', 0):.2f}%")
        print(f"   치주염 진행도: {stage}단계 ({'양호' if stage <= 1 else '병원방문 권장'})")
        
        # 생성된 마스킹 이미지들
        masked_imgs = result['masked_images']
        print(f"\n생성된 분석 이미지:")
        for img_type, path in masked_imgs.items():
            if os.path.exists(path):
                print(f"   {img_type}: {path}")
            else:
                print(f"   {img_type}: {path} (파일 없음)")
        
        # 피드백 출력 (3문장 제한)
        feedback = result['feedback']
        print(f"\n전문가 피드백 (각 정확히 3문장):")
        for condition, advice in feedback.items():
            print(f"\n   [{condition}]")
            sentences = advice.split('.')
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    print(f"   {i+1}. {sentence.strip()}.")
        
    except Exception as e:
        print(f"진단 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())