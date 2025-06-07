"""
맞춤형 치아 관리 피드백 시스템 (백엔드 연동 버전)
- 이미지 분석 결과를 매개변수로 받음
- 진단 컨텍스트를 매개변수로 받음
- 양치 현황 분석 및 맞춤 가이드에 집중
"""
import os
import json
import mysql.connector
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime, date
import asyncio
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

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

class ManagementReportState(TypedDict):
    """관리 피드백 리포트를 위한 State 클래스"""
    guardian_id: int
    pet_id: int
    
    # 백엔드에서 받는 데이터
    diagnosis_results: Dict[str, Any]      # 이미지 분석 결과
    diagnosis_feedback: Dict[str, str]     # 진단별 피드백 텍스트
    # masked_images: Dict[str, str]        # 마스킹된 이미지 경로들 (삭제)
    
    # 조회하는 데이터
    pet_info: Dict[str, Any]
    brush_style_info: Dict[str, Any]
    
    # 검색하는 데이터
    custom_contexts: Dict[str, List[str]]  # 맞춤형 컨텍스트만
    
    # 분석 결과
    brushing_analysis: Dict[str, Any]
    customized_guide: Dict[str, str]
    final_feedback: Dict[str, str]
    
    messages: List[Dict[str, Any]]
    next_agent: str

def preprocess_english_text(text):
    """영어 텍스트에서 불용어 제거 및 기본 전처리 (치과 용어 보존)"""
    # 치과 관련 중요 용어들은 보존
    dental_terms = {
        'periodontal', 'periodontitis', 'gingivitis', 'plaque', 'tartar', 'calculus',
        'anesthesia', 'trauma', 'fractured', 'crowded', 'senior', 'puppy', 'small',
        'large', 'toy', 'breeds', 'gums', 'teeth', 'dental', 'oral', 'brushing',
        'resistance', 'frequency', 'duration', 'routine', 'habits', 'improvement'
    }
    
    # 특수 문자 및 숫자 제거 (알파벳과 공백만 유지)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 여러 공백을 하나로 치환
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 단어 토큰화
    tokens = word_tokenize(text.lower())
    
    # 불용어 제거 (단, 치과 관련 중요 용어는 보존)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words or word in dental_terms]
    
    # 표제어 추출 (Lemmatization) - 치과 용어는 원형 보존
    lemmatizer = WordNetLemmatizer()
    processed_tokens = []
    for word in tokens:
        if word in dental_terms:
            processed_tokens.append(word)  # 치과 용어는 원형 보존
        else:
            processed_tokens.append(lemmatizer.lemmatize(word))
    
    # 토큰을 다시 텍스트로 결합
    processed_text = ' '.join(processed_tokens)
    
    return processed_text

class EnhancedMySQLAgent:
    """확장된 MySQL Agent - 펫 정보 및 양치 현황 조회"""
    
    def __init__(self):
        self.connection_config = {
            'host': MYSQL_HOST,
            'user': MYSQL_USER,
            'password': MYSQL_PASSWORD,
            'database': MYSQL_DATABASE
        }
    
    def get_pet_info(self, guardian_id: int, pet_id: int) -> Dict[str, Any]:
        """펫 정보 조회"""
        try:
            conn = mysql.connector.connect(**self.connection_config)
            cursor = conn.cursor(dictionary=True)
            
            query = """
            SELECT breed, weight, birth_date, gender, name
            FROM pets 
            WHERE guardian_id = %s AND pet_id = %s
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
                
                # 크기 분류 (체중 기준)
                weight = result.get('weight', 0)
                if weight <= 10:
                    result['size_category'] = '소형'
                elif weight <= 25:
                    result['size_category'] = '중형'
                else:
                    result['size_category'] = '대형'
                
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
    
    def get_brush_style_info(self, guardian_id: int, pet_id: int) -> Dict[str, Any]:
        """양치 현황 정보 조회"""
        try:
            conn = mysql.connector.connect(**self.connection_config)
            cursor = conn.cursor(dictionary=True)
            
            query = """
            SELECT 
                bs.brushing_frequency,
                bs.brushing_duration,
                bs.brushing_time,
                bs.brushing_routine,
                bs.pet_resistance,
                bs.notes,
                bs.start_date,
                bp.product_name as brush_name,
                bp.brush_type,
                tp.product_name as paste_name,
                tp.paste_type
            FROM brush_style bs
            LEFT JOIN brush_product bp ON bs.brush_id = bp.product_id
            LEFT JOIN toothpaste_product tp ON bs.paste_id = tp.product_id
            WHERE bs.guardian_id = %s AND bs.pet_id = %s
            """
            
            cursor.execute(query, (guardian_id, pet_id))
            result = cursor.fetchone()
            
            return result if result else {}
                
        except mysql.connector.Error as e:
            print(f"MySQL 연결 오류: {e}")
            return {}
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

class CustomizedRetrievalAgent:
    """맞춤형 검색을 위한 Agent"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    def dense_retrieve(self, query: str, namespace: str, top_k: int = 5) -> List[str]:
        """Dense 검색 수행"""
        try:
            # 쿼리 전처리
            processed_query = preprocess_english_text(query)
            
            # Dense 임베딩 생성
            dense_vector = self.embeddings.embed_query(processed_query)
            
            # Dense 검색
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
                    
            print(f"Dense 검색 완료 ({namespace}): {len(contexts)}개 컨텍스트 발견")
            return contexts
            
        except Exception as e:
            print(f"Dense 검색 오류 ({namespace}): {e}")
            return []

class BrushingAnalysisAgent:
    """양치 현황 분석 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    def analyze_brushing_habits(self, pet_info: Dict, brush_info: Dict, 
                              diagnosis_feedback: Dict, contexts: List[str]) -> Dict[str, Any]:
        """양치 현황 분석 및 개선점 제안"""
        
        # 컨텍스트 결합
        context_text = "\n".join(contexts[:3]) if contexts else "전문 지식이 없습니다."
        
        # 저항도 점수화
        resistance_scores = {'없음': 1, '약간': 2, '보통': 3, '심함': 4}
        resistance_score = resistance_scores.get(brush_info.get('pet_resistance', '없음'), 1)
        
        # 빈도 평가
        frequency = brush_info.get('brushing_frequency', 0)
        duration = brush_info.get('brushing_duration', 0)
        
        prompt = ChatPromptTemplate.from_template(
            """As a canine dental care specialist, please analyze the current brushing habits and provide improvement recommendations.

Pet Information:
- Name: {pet_name}
- Breed: {breed}
- Age: {age} years old
- Weight: {weight}kg
- Size Category: {size_category}

Current Brushing Status:
- Frequency: {frequency} times per week
- Duration: {duration} minutes
- Timing: {brushing_time}
- Routine: {brushing_routine}
- Resistance Level: {pet_resistance} (Score: {resistance_score}/4)
- Toothbrush: {brush_name}
- Toothpaste: {paste_name}

Dental Diagnosis Feedback:
- Periodontitis: {periodontitis_feedback}
- Calculus: {calculus_feedback}
- Caries: {caries_feedback}

Professional Knowledge:
{context}

**Evaluation Criteria**:
- Frequency: Small dogs 5-7 times/week, Medium/Large dogs 3-5 times/week recommended
- Duration: Minimum 2-3 minutes, ideally 5 minutes
- Resistance: 1-2 points (good), 3 points (moderate, needs improvement), 4 points (severe, training needed)
- Age consideration: Senior dogs (7+ years) need especially careful management
- Size-specific: Small dogs have higher periodontal disease risk, Large dogs have higher dental trauma risk

Please analyze in the following format:

CURRENT_STATUS: [Overall assessment of current brushing habits in 2 lines]
IMPROVEMENT_POINTS: [3 specific improvement recommendations numbered]
PERSONALIZED_TIPS: [2 personalized tips especially helpful for this dog]
OVERALL_SCORE: [Current dental care management score out of 100]

Please respond in Korean with natural and professional language."""
        )
        
        try:
            response = self.llm.invoke(
                prompt.format(
                    pet_name=pet_info.get('name', '반려견'),
                    breed=pet_info.get('breed', '미상'),
                    age=pet_info.get('age_years', '미상'),
                    weight=pet_info.get('weight', '미상'),
                    size_category=pet_info.get('size_category', '미상'),
                    frequency=frequency,
                    duration=duration,
                    brushing_time=brush_info.get('brushing_time', '미상'),
                    brushing_routine=brush_info.get('brushing_routine', '특별한 루틴 없음'),
                    pet_resistance=brush_info.get('pet_resistance', '없음'),
                    resistance_score=resistance_score,
                    brush_name=brush_info.get('brush_name', '미상'),
                    paste_name=brush_info.get('paste_name', '미상'),
                    periodontitis_feedback=diagnosis_feedback.get('치주염', '해당없음'),
                    calculus_feedback=diagnosis_feedback.get('치석', '해당없음'),
                    caries_feedback=diagnosis_feedback.get('충치', '해당없음'),
                    context=context_text
                )
            )
            
            response_text = response.content
            
            # 점수 추출
            score_match = re.search(r'OVERALL_SCORE:\s*(\d+)', response_text)
            overall_score = int(score_match.group(1)) if score_match else 75
            
            return {
                "analysis_text": response_text,
                "overall_score": overall_score,
                "frequency_assessment": "적절" if 3 <= frequency <= 7 else "개선필요",
                "duration_assessment": "적절" if 2 <= duration <= 5 else "개선필요",
                "resistance_level": resistance_score
            }
            
        except Exception as e:
            print(f"양치 현황 분석 오류: {e}")
            return {
                "analysis_text": "양치 현황 분석을 완료할 수 없습니다.",
                "overall_score": 50,
                "frequency_assessment": "미상",
                "duration_assessment": "미상",
                "resistance_level": 1
            }

class CustomizedGuideAgent:
    """맞춤형 가이드 생성 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    def generate_age_size_guide(self, pet_info: Dict, size_contexts: List[str], 
                               age_contexts: List[str]) -> Dict[str, str]:
        """나이와 크기별 맞춤 가이드 생성"""
        
        age = pet_info.get('age_years', 0)
        size_category = pet_info.get('size_category', '중형')
        breed = pet_info.get('breed', '믹스')
        
        # 나이대 분류
        if age <= 1:
            age_group = "유아견"
        elif age <= 7:
            age_group = "성견"
        else:
            age_group = "노견"
        
        # 컨텍스트 결합
        size_context = "\n".join(size_contexts[:2]) if size_contexts else "크기별 전문 지식이 없습니다."
        age_context = "\n".join(age_contexts[:2]) if age_contexts else "나이별 전문 지식이 없습니다."
        
        prompt = ChatPromptTemplate.from_template(
            """As a canine oral health specialist, please provide customized guidance based on the dog's age and size.

Pet Information:
- Breed: {breed}
- Age: {age} years old ({age_group} stage)
- Size: {size_category} dog
- Weight: {weight}kg

Size-specific Professional Knowledge:
{size_context}

Age-specific Professional Knowledge:
{age_context}

Please refer to the following guidelines when writing:

**Small Dog Characteristics**: Teeth are densely arranged, making tartar buildup easier and increasing periodontal disease risk. Gentle brushing and frequent wiping are important.

**Large Dog Characteristics**: Higher risk of dental trauma from chewing hard objects and tooth fractures. Avoid overly hard toys and regular checkups are necessary.

**Puppy Stage**: Early tooth development phase where prevention is most important. Oral examination during vaccination is essential.

**Adult Stage**: Critical period for establishing oral care habits. Maintain lifelong health through consistent brushing and regular checkups.

**Senior Stage**: 7+ years old with increased gum disease risk due to deficiencies in calcium, phosphorus, vitamin B, and zinc. Management considering anesthesia risks is needed.

Please write exactly 4 lines of guidance in the following format:

SIZE_GUIDE: [2 lines of {size_category} dog oral health precautions]
AGE_GUIDE: [2 lines of {age_group} stage oral care key points]

Each guide should reflect the above characteristics and provide specific, actionable advice.

Please respond in Korean with natural and professional language."""
        )
        
        try:
            response = self.llm.invoke(
                prompt.format(
                    breed=breed,
                    age=age,
                    age_group=age_group,
                    size_category=size_category,
                    weight=pet_info.get('weight', '미상'),
                    size_context=size_context,
                    age_context=age_context
                )
            )
            
            response_text = response.content
            
            # SIZE_GUIDE와 AGE_GUIDE 추출
            size_guide_match = re.search(r'SIZE_GUIDE:\s*\[(.*?)\]', response_text, re.DOTALL)
            age_guide_match = re.search(r'AGE_GUIDE:\s*\[(.*?)\]', response_text, re.DOTALL)
            
            size_guide = size_guide_match.group(1).strip() if size_guide_match else f"{size_category}견에 맞는 구강관리가 필요합니다."
            age_guide = age_guide_match.group(1).strip() if age_guide_match else f"{age_group} 시기에 맞는 관리가 중요합니다."
            
            return {
                "size_guide": size_guide,
                "age_guide": age_guide,
                "full_response": response_text
            }
            
        except Exception as e:
            print(f"맞춤 가이드 생성 오류: {e}")
            return {
                "size_guide": f"{size_category}견에 맞는 구강관리를 실시하세요.",
                "age_guide": f"{age_group} 시기에 적합한 관리가 필요합니다.",
                "full_response": "맞춤 가이드를 생성할 수 없습니다."
            }

class ManagementReportSupervisor:
    """맞춤형 관리 피드백 리포트 Supervisor"""
    
    def __init__(self):
        self.mysql_agent = EnhancedMySQLAgent()
        self.retrieval_agent = CustomizedRetrievalAgent()
        self.brushing_agent = BrushingAnalysisAgent()
        self.guide_agent = CustomizedGuideAgent()
        
        # LangGraph 설정
        self.workflow = StateGraph(ManagementReportState)
        self._setup_workflow()
    
    def _setup_workflow(self):
        """관리 피드백 워크플로우 노드 및 엣지 설정"""
        
        # 노드 추가
        self.workflow.add_node("get_pet_info", self._get_pet_info_node)
        self.workflow.add_node("get_brush_info", self._get_brush_info_node)
        self.workflow.add_node("retrieve_custom_contexts", self._retrieve_custom_contexts_node)
        self.workflow.add_node("analyze_brushing", self._analyze_brushing_node)
        self.workflow.add_node("generate_custom_guide", self._generate_custom_guide_node)
        self.workflow.add_node("generate_final_feedback", self._generate_final_feedback_node)
        
        # 시작점 설정
        self.workflow.set_entry_point("get_pet_info")
        
        # 엣지 연결 (이미지 분석과 진단 컨텍스트 노드 제거)
        self.workflow.add_edge("get_pet_info", "get_brush_info")
        self.workflow.add_edge("get_brush_info", "retrieve_custom_contexts")
        self.workflow.add_edge("retrieve_custom_contexts", "analyze_brushing")
        self.workflow.add_edge("analyze_brushing", "generate_custom_guide")
        self.workflow.add_edge("generate_custom_guide", "generate_final_feedback")
        self.workflow.add_edge("generate_final_feedback", END)
        
        # 그래프 컴파일
        self.app = self.workflow.compile()
    
    def _get_pet_info_node(self, state: ManagementReportState) -> ManagementReportState:
        """펫 정보 조회 노드"""
        print(f"펫 정보 조회 중... (Guardian: {state['guardian_id']}, Pet: {state['pet_id']})")
        
        pet_info = self.mysql_agent.get_pet_info(state['guardian_id'], state['pet_id'])
        state['pet_info'] = pet_info
        
        print(f"펫 정보: {pet_info.get('name', '미상')} ({pet_info.get('breed', '미상')}, "
              f"{pet_info.get('age_years', '미상')}세, {pet_info.get('size_category', '미상')}견)")
        return state
    
    def _get_brush_info_node(self, state: ManagementReportState) -> ManagementReportState:
        """양치 현황 정보 조회 노드"""
        print("양치 현황 정보 조회 중...")
        
        brush_info = self.mysql_agent.get_brush_style_info(
            state['guardian_id'], state['pet_id']
        )
        state['brush_style_info'] = brush_info
        
        print(f"양치 현황: 주 {brush_info.get('brushing_frequency', 0)}회, "
              f"{brush_info.get('brushing_duration', 0)}분, 저항도: {brush_info.get('pet_resistance', '없음')}")
        return state
    
    def _retrieve_custom_contexts_node(self, state: ManagementReportState) -> ManagementReportState:
        """맞춤형 컨텍스트 검색 노드"""
        print("맞춤형 관리 지식 검색 중...")
        
        pet_info = state['pet_info']
        brush_info = state['brush_style_info']
        
        # 양치 습관 개선을 위한 검색 (더 구체적인 키워드)
        brush_resistance = brush_info.get('pet_resistance', '없음')
        brush_query = f"dog tooth brushing daily routine resistance training habits frequency duration improve dental hygiene management"
        brush_contexts = self.retrieval_agent.dense_retrieve(brush_query, 'custom_brush', top_k=5)
        
        # 크기별 가이드 검색 (문서 내용 기반 키워드 강화)
        size_category = pet_info.get('size_category', '중형')
        if size_category == '소형':
            # 소형견 관련 핵심 키워드들
            size_query = "small dogs toy breeds dental problems periodontal disease crowded mouth teeth close together plaque tartar buildup wiping brushing"
        elif size_category == '대형':
            # 대형견 관련 핵심 키워드들  
            size_query = "large dogs big breeds fractured teeth trauma chewing hard objects bones antlers oral cancer tooth damage prevention"
        else:
            # 중형견은 소형+대형 특성 혼합
            size_query = "medium dogs dental care periodontal disease tooth brushing plaque prevention routine cleaning management"
        
        size_contexts = self.retrieval_agent.dense_retrieve(size_query, 'custom_size', top_k=3)
        
        # 나이별 가이드 검색 (문서 내용 기반 키워드 강화)
        age = pet_info.get('age_years', 3)
        if age <= 1:
            # 유아견 관련 키워드
            age_query = "puppy young dog teeth development first vaccines dental examination mouth growth prevention early care training"
        elif age >= 7:
            # 성견 관련 핵심 키워드들
            age_query = "senior dog older aging seven years dental health deteriorate calcium phosphorus vitamins zinc antioxidants anesthesia risk management"
        else:
            # 성견 키워드
            age_query = "adult dog dental care routine maintenance healthy teeth gums prevention regular cleaning habits formation"
        
        age_contexts = self.retrieval_agent.dense_retrieve(age_query, 'custom_age', top_k=3)
        
        # 맞춤형 컨텍스트 저장
        state['custom_contexts'] = {
            'custom_brush': brush_contexts,
            'custom_size': size_contexts,
            'custom_age': age_contexts
        }
        
        print(f"맞춤 검색 완료: 양치 {len(brush_contexts)}개, "
              f"크기 {len(size_contexts)}개, 나이 {len(age_contexts)}개")
        return state
    
    def _analyze_brushing_node(self, state: ManagementReportState) -> ManagementReportState:
        """양치 현황 분석 노드"""
        print("양치 현황 분석 중...")
        
        brushing_analysis = self.brushing_agent.analyze_brushing_habits(
            state['pet_info'],
            state['brush_style_info'],
            state['diagnosis_feedback'],  # 백엔드에서 받은 진단 피드백 사용
            state['custom_contexts'].get('custom_brush', [])
        )
        
        state['brushing_analysis'] = brushing_analysis
        
        print(f"양치 분석 완료: 종합점수 {brushing_analysis.get('overall_score', 0)}점")
        return state
    
    def _generate_custom_guide_node(self, state: ManagementReportState) -> ManagementReportState:
        """맞춤 가이드 생성 노드"""
        print("나이/크기별 맞춤 가이드 생성 중...")
        
        custom_guide = self.guide_agent.generate_age_size_guide(
            state['pet_info'],
            state['custom_contexts'].get('custom_size', []),
            state['custom_contexts'].get('custom_age', [])
        )
        
        state['customized_guide'] = custom_guide
        
        print("맞춤 가이드 생성 완료")
        return state
    
    def _generate_final_feedback_node(self, state: ManagementReportState) -> ManagementReportState:
        """최종 피드백 생성 노드"""
        print("종합 피드백 생성 중...")
        
        # 모든 분석 결과를 종합
        final_feedback = {
            'brushing_analysis': state['brushing_analysis'].get('analysis_text', ''),
            'size_guide': state['customized_guide'].get('size_guide', ''),
            'age_guide': state['customized_guide'].get('age_guide', ''),
            'overall_score': state['brushing_analysis'].get('overall_score', 0),
            'diagnosis_summary': {
                'caries_ratio': state['diagnosis_results'].get('caries_ratio', 0),
                'calculus_ratio': state['diagnosis_results'].get('calculus_ratio', 0),
                'periodontitis_stage': state['diagnosis_results'].get('periodontitis_stage', 0)
            }
        }
        
        state['final_feedback'] = final_feedback
        
        print("종합 피드백 생성 완료")
        return state
    
    async def generate_management_report(self, 
                                       guardian_id: int, 
                                       pet_id: int,
                                       diagnosis_results: Dict[str, Any],
                                       diagnosis_feedback: Dict[str, str]):
        """맞춤형 관리 피드백 리포트 생성 (마스킹 이미지 제거)"""
        
        initial_state = ManagementReportState(
            guardian_id=guardian_id,
            pet_id=pet_id,
            diagnosis_results=diagnosis_results,
            diagnosis_feedback=diagnosis_feedback,
            pet_info={},
            brush_style_info={},
            custom_contexts={},
            brushing_analysis={},
            customized_guide={},
            final_feedback={},
            messages=[],
            next_agent=""
        )
        
        # 워크플로우 실행
        final_state = await self.app.ainvoke(initial_state)
        
        # 결과 정리
        result = {
            "pet_info": final_state['pet_info'],
            "brush_style_info": final_state['brush_style_info'],
            "diagnosis_results": final_state['diagnosis_results'],
            "diagnosis_feedback": final_state['diagnosis_feedback'],
            "brushing_analysis": final_state['brushing_analysis'],
            "customized_guide": final_state['customized_guide'],
            "final_feedback": final_state['final_feedback']
        }
        
        return result

# 사용 예시
async def main():
    # Supervisor 초기화
    supervisor = ManagementReportSupervisor()
    
    # 백엔드에서 받을 샘플 데이터
    diagnosis_results = {
        'caries_ratio': 15.5,
        'calculus_ratio': 8.2,
        'periodontitis_stage': 2,
        'total_tooth_area': 5000,
        'caries_area': 775,
        'calculus_area': 410
    }
    
    diagnosis_feedback = {
        '치주염': '2단계 치주염 소견으로 잇몸에 염증이 있습니다. 정기적인 양치와 병원 상담을 권장합니다.',
        '치석': '치석이 적당히 축적되어 있습니다. 정기적인 스케일링과 양치 빈도를 늘려주세요.',
        '충치': '초기 충치가 발견되었습니다. 양치 횟수를 늘리고 치아 관리에 더 신경써 주세요.'
    }
    
    try:
        result = await supervisor.generate_management_report(
            guardian_id=1,
            pet_id=1,
            diagnosis_results=diagnosis_results,
            diagnosis_feedback=diagnosis_feedback
        )
        
        print("\n" + "="*80)
        print("맞춤형 치아 관리 피드백 리포트 (백엔드 연동 버전)")
        print("="*80)
        
        # 반려견 정보
        pet_info = result['pet_info']
        print(f"\n 반려견 정보:")
        print(f"   이름: {pet_info.get('name', '미상')}")
        print(f"   품종: {pet_info.get('breed', '미상')}")
        print(f"   나이: {pet_info.get('age_years', '미상')}세")
        print(f"   크기: {pet_info.get('size_category', '미상')}견 ({pet_info.get('weight', '미상')}kg)")
        
        # 현재 양치 현황
        brush_info = result['brush_style_info']
        print(f"\n 현재 양치 현황:")
        print(f"   빈도: 주 {brush_info.get('brushing_frequency', 0)}회")
        print(f"   지속시간: {brush_info.get('brushing_duration', 0)}분")
        print(f"   시간대: {brush_info.get('brushing_time', '미상')}")
        print(f"   저항도: {brush_info.get('pet_resistance', '없음')}")
        print(f"   사용 칫솔: {brush_info.get('brush_name', '미상')}")
        print(f"   사용 치약: {brush_info.get('paste_name', '미상')}")
        
        # 진단 결과 (백엔드에서 받은 데이터)
        diagnosis = result['diagnosis_results']
        print(f"\n 구강 진단 결과 (백엔드 제공):")
        print(f"   충치 비율: {diagnosis.get('caries_ratio', 0):.1f}%")
        print(f"   치석 비율: {diagnosis.get('calculus_ratio', 0):.1f}%")
        print(f"   치주염 단계: {diagnosis.get('periodontitis_stage', 0)}단계")
        
        # 진단별 피드백 (백엔드에서 받은 데이터)
        diagnosis_fb = result['diagnosis_feedback']
        print(f"\n 구강 진단 피드백 (백엔드 제공):")
        for condition, feedback in diagnosis_fb.items():
            print(f"   [{condition}] {feedback}")
        
        # 양치 현황 분석
        brushing_analysis = result['brushing_analysis']
        print(f"\n 양치 현황 분석:")
        print(f"   종합 점수: {brushing_analysis.get('overall_score', 0)}점/100점")
        print(f"   빈도 평가: {brushing_analysis.get('frequency_assessment', '미상')}")
        print(f"   지속시간 평가: {brushing_analysis.get('duration_assessment', '미상')}")
        
        # 맞춤 가이드
        custom_guide = result['customized_guide']
        print(f"\n 맞춤형 관리 가이드:")
        print(f"\n   [{pet_info.get('size_category', '중')}형견 전용 가이드]")
        print(f"   {custom_guide.get('size_guide', '크기별 가이드가 없습니다.')}")
        
        age_group = "퍼피" if pet_info.get('age_years', 0) <= 1 else ("시니어" if pet_info.get('age_years', 0) > 7 else "성견")
        print(f"\n   [{age_group} 전용 가이드]")
        print(f"   {custom_guide.get('age_guide', '나이별 가이드가 없습니다.')}")
        
        # 종합 피드백
        final_feedback = result['final_feedback']
        print(f"\n 종합 양치 분석:")
        analysis_text = final_feedback.get('brushing_analysis', '')
        if analysis_text:
            # 분석 텍스트에서 주요 섹션들 추출해서 깔끔하게 출력
            sections = analysis_text.split('\n')
            for section in sections:
                if section.strip():
                    print(f"   {section.strip()}")
        
        print(f"\n 전체 관리 점수: {final_feedback.get('overall_score', 0)}점/100점")
        
    except Exception as e:
        print(f"리포트 생성 중 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(main())