"""
LangGraph 기반 치아 관리 챗봇 시스템
- 최근 dental_analysis 정보를 기반으로 시작
- 질문 분류 판단 후 해당 namespace에서 검색
- 한글 문서 대응
- MemorySaver를 사용한 대화 히스토리 관리
- thread_id 지원
"""

import os
import json
import mysql.connector
import re
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime, date
import asyncio

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

# 환경 변수 설정
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_USER = os.environ.get("MYSQL_USER")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
MYSQL_DATABASE = "pet_dental_care"

class ChatbotState(TypedDict):
    """챗봇 상태 관리를 위한 State 클래스"""
    guardian_id: int
    pet_id: int
    thread_id: str  # thread_id 추가
    
    # 최근 분석 정보
    recent_analysis: Dict[str, Any]
    pet_info: Dict[str, Any]
    product_recommendations: Dict[str, Any]
    
    # 현재 대화
    user_question: str
    selected_category: str  # "질병", "식습관", "양치법", "이상행동"
    
    # 분류 판단 결과
    is_valid_category: bool
    classification_confidence: float
    
    # 검색 결과
    retrieved_contexts: List[str]
    
    # 응답
    chatbot_response: str
    
    # 대화 히스토리 (MemorySaver가 자동 관리)
    conversation_history: List[Dict[str, str]]
    next_agent: str

class RecentAnalysisAgent:
    """최근 dental_analysis 정보를 조회하는 Agent"""
    
    def __init__(self):
        self.connection_config = {
            'host': MYSQL_HOST,
            'user': MYSQL_USER,
            'password': MYSQL_PASSWORD,
            'database': MYSQL_DATABASE
        }
    
    def get_recent_analysis(self, guardian_id: int, pet_id: int) -> Dict[str, Any]:
        """가장 최근 dental_analysis 정보 조회"""
        try:
            conn = mysql.connector.connect(**self.connection_config)
            cursor = conn.cursor(dictionary=True)
            
            # 가장 최근 분석 정보 조회
            analysis_query = """
            SELECT 
                analysis_date,
                caries_percentage,
                calculus_percentage,
                periodontal_level,
                recommend_brush_frequency,
                recommend_brush_id,
                recommend_toothpaste_id,
                recommend_other_product_id,
                recommend_routine,
                recommend_guide,
                total_score
            FROM dental_analysis 
            WHERE guardian_id = %s AND pet_id = %s
            ORDER BY analysis_date DESC
            LIMIT 1
            """
            
            cursor.execute(analysis_query, (guardian_id, pet_id))
            analysis_result = cursor.fetchone()
            
            if not analysis_result:
                return {"error": "최근 분석 정보가 없습니다."}
            
            # 추천 제품 정보 조회
            product_info = {}
            
            # 추천 칫솔 정보
            if analysis_result['recommend_brush_id']:
                brush_query = """
                SELECT product_name, brand, brush_type, size, material, description
                FROM brush_product 
                WHERE product_id = %s
                """
                cursor.execute(brush_query, (analysis_result['recommend_brush_id'],))
                brush_result = cursor.fetchone()
                if brush_result:
                    product_info['recommended_brush'] = brush_result
            
            # 추천 치약 정보
            if analysis_result['recommend_toothpaste_id']:
                paste_query = """
                SELECT product_name, brand, paste_type, flavor, volume, description
                FROM toothpaste_product 
                WHERE product_id = %s
                """
                cursor.execute(paste_query, (analysis_result['recommend_toothpaste_id'],))
                paste_result = cursor.fetchone()
                if paste_result:
                    product_info['recommended_toothpaste'] = paste_result
            
            # 추천 기타 제품 정보
            if analysis_result['recommend_other_product_id']:
                other_query = """
                SELECT product_name, brand, product_type, size, flavor, 
                       main_ingredient, effectiveness_area, usage_frequency, description
                FROM other_product 
                WHERE product_id = %s
                """
                cursor.execute(other_query, (analysis_result['recommend_other_product_id'],))
                other_result = cursor.fetchone()
                if other_result:
                    product_info['recommended_other'] = other_result
            
            # 펫 정보도 함께 조회
            pet_query = """
            SELECT p.name, p.breed, p.weight, p.birth_date, p.gender,
                   g.name as guardian_name, g.experience_level
            FROM pets p
            JOIN guardians g ON p.guardian_id = g.guardian_id
            WHERE p.guardian_id = %s AND p.pet_id = %s
            """
            cursor.execute(pet_query, (guardian_id, pet_id))
            pet_result = cursor.fetchone()
            
            # 나이 계산
            if pet_result and pet_result['birth_date']:
                today = date.today()
                birthdate = pet_result['birth_date']
                age_years = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
                pet_result['age_years'] = age_years
            
            return {
                "analysis": analysis_result,
                "products": product_info,
                "pet_info": pet_result or {}
            }
                
        except mysql.connector.Error as e:
            print(f"MySQL 연결 오류: {e}")
            return {"error": f"데이터베이스 오류: {e}"}
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

class QuestionValidatorAgent:
    """사용자가 선택한 분류에 질문이 적합한지 검증하는 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # 분류별 키워드 정의 (선택된 분류 적합성 판단용)
        self.category_keywords = {
            "질병": [
                "충치", "치주염", "잇몸", "염증", "통증", "아픔", "붓기", "피", "출혈",
                "치석", "플라그", "구취", "입냄새", "치아", "이빨", "병", "질환",
                "감염", "농양", "상처", "궤양", "검은점", "갈색", "변색", "흔들림"
            ],
            "식습관": [
                "먹이", "사료", "간식", "음식", "식이", "영양", "칼슘", "비타민",
                "단단한", "딱딱한", "뼈", "껌", "씹는", "섭취", "식단", "급여",
                "물", "우유", "달달한","당분", "설탕", "달콤한", "부드러운", "젖은", "습식", "건식"
            ],
            "양치법": [
                "양치", "칫솔", "치약", "닦기", "문지르기", "브러싱", "청소",
                "세정", "구강", "관리", "빈도", "횟수", "시간", "방법", "기법",
                "저항", "싫어", "거부", "훈련", "습관", "루틴", "스케일링", "병원"
            ],
            "이상행동": [
                "씹기", "물어뜯기", "핥기", "비비기", "긁기", "발톱", "앞발",
                "입", "혀", "침", "흘리기", "거품", "토하기", "먹지않음", "식욕",
                "행동", "변화", "이상", "평소", "갑자기", "계속", "자주", "반복"
            ]
        }
    
    def validate_question_category(self, question: str, selected_category: str) -> Dict[str, Any]:
        """사용자가 선택한 분류에 질문이 적합한지 검증"""
        
        # 1차: 키워드 기반 빠른 검증
        category_keywords = self.category_keywords.get(selected_category, [])
        keyword_matches = sum(1 for keyword in category_keywords if keyword in question)
        keyword_confidence = min(keyword_matches / max(len(category_keywords) * 0.1, 1), 1.0)
        
        # 2차: LLM 기반 정밀 검증
        prompt = ChatPromptTemplate.from_template(
            """당신은 반려동물 구강 건강 상담사입니다. 사용자가 선택한 분류에 질문이 적합한지 검증해주세요.

사용자가 선택한 분류: {category}

분류별 범위:
- 질병: 충치, 치주염, 잇몸염, 구취, 치아 질환, 감염 등 구강 질병 관련
- 식습관: 먹이, 간식, 영양소, 씹을거리, 식단 등 구강 건강에 영향을 주는 식이 관련
- 양치법: 양치 방법, 칫솔 사용법, 구강 관리 기법, 빈도 등 직접적인 관리법 관련
- 이상행동: 비정상적인 씹기, 핥기, 침흘리기 등 구강 관련 행동 변화

사용자 질문: "{question}"

검증 기준:
1. 질문이 선택된 '{category}' 분류와 직접적으로 관련이 있는가?
2. 질문 내용이 해당 분류의 범위 안에 포함되는가?
3. 다른 분류가 더 적합하지는 않은가?

다음 형식으로 응답하세요:

VALID: [YES/NO]
CONFIDENCE: [0-100 사이의 숫자]
REASON: [적합/부적합 판단 근거를 한 줄로]
SUGGESTION: [NO인 경우만] 더 적합한 분류 또는 질문 수정 제안

올바른 예시:
VALID: YES
CONFIDENCE: 90
REASON: 양치 방법에 대한 질문으로 '양치법' 분류에 정확히 해당합니다.
SUGGESTION: -

부적합 예시:
VALID: NO  
CONFIDENCE: 15
REASON: 산책 관련 질문으로 구강 건강과 관련이 없습니다.
SUGGESTION: 구강 건강 관련 질문으로 다시 작성해주세요.
"""
        )
        
        try:
            response = self.llm.invoke(
                prompt.format(category=selected_category, question=question)
            )
            
            response_text = response.content
            
            # 응답 파싱
            valid_match = re.search(r'VALID:\s*(YES|NO)', response_text, re.IGNORECASE)
            confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', response_text)
            reason_match = re.search(r'REASON:\s*([^\n]+)', response_text)
            suggestion_match = re.search(r'SUGGESTION:\s*([^\n]+)', response_text)
            
            is_valid = valid_match.group(1).upper() == "YES" if valid_match else False
            llm_confidence = int(confidence_match.group(1)) / 100.0 if confidence_match else 0.5
            reason = reason_match.group(1).strip() if reason_match else "판단 근거를 찾을 수 없습니다."
            suggestion = suggestion_match.group(1).strip() if suggestion_match else ""
            
            # 키워드와 LLM 신뢰도 결합 (가중평균)
            final_confidence = (keyword_confidence * 0.3) + (llm_confidence * 0.7)
            
            return {
                "is_valid": is_valid and final_confidence > 0.3,  # 최소 신뢰도 임계값
                "confidence": final_confidence,
                "reason": reason,
                "suggestion": suggestion,
                "keyword_matches": keyword_matches,
                "keyword_confidence": keyword_confidence,
                "llm_confidence": llm_confidence
            }
            
        except Exception as e:
            print(f"질문 적합성 검증 오류: {e}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "reason": "적합성 검증 중 오류가 발생했습니다.",
                "suggestion": "다시 시도해주세요.",
                "keyword_matches": 0,
                "keyword_confidence": 0.0,
                "llm_confidence": 0.0
            }

class KoreanChatRetrievalAgent:
    """한글 문서 검색을 위한 Agent"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 분류별 namespace 매핑
        self.category_namespaces = {
            "질병": "chat_disease",
            "식습관": "chat_habit", 
            "양치법": "chat_brush",
            "이상행동": "chat_behavior"
        }
    
    def preprocess_korean_query(self, query: str) -> str:
        """한글 쿼리 전처리"""
        # 특수문자 제거 및 공백 정리
        query = re.sub(r'[^\w\s가-힣]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        # 너무 짧은 쿼리는 확장
        if len(query) < 10:
            query = f"반려동물 강아지 {query} 관리 방법 증상 원인"
        
        return query
    
    def retrieve_documents(self, question: str, category: str, top_k: int = 5) -> List[str]:
        """해당 분류의 namespace에서 문서 검색"""
        try:
            namespace = self.category_namespaces.get(category)
            if not namespace:
                print(f"알 수 없는 분류: {category}")
                return []
            
            # 한글 쿼리 전처리
            processed_query = self.preprocess_korean_query(question)
            print(f"검색 쿼리: '{processed_query}' -> namespace: {namespace}")
            
            # 임베딩 생성
            query_embedding = self.embeddings.embed_query(processed_query)
            
            # Pinecone 검색
            results = self.index.query(
                vector=query_embedding,
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
            
            print(f"검색 완료 ({namespace}): {len(contexts)}개 문서 발견")
            return contexts
            
        except Exception as e:
            print(f"문서 검색 오류 ({category}): {e}")
            return []

class ChatResponseAgent:
    """챗봇 응답 생성 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    def generate_response(self, question: str, category: str, contexts: List[str], 
                         recent_analysis: Dict[str, Any], pet_info: Dict[str, Any],
                         conversation_history: List[Dict[str, str]] = None) -> str:
        """컨텍스트와 대화 히스토리를 기반으로 응답 생성"""
        
        # 최근 분석 정보 요약
        analysis = recent_analysis.get('analysis', {})
        products = recent_analysis.get('products', {})
        
        analysis_summary = f"""
최근 구강 검진 결과 (날짜: {analysis.get('analysis_date', '미상')}):
- 충치 비율: {analysis.get('caries_percentage', 0)}%
- 치석 비율: {analysis.get('calculus_percentage', 0)}%  
- 치주염 단계: {analysis.get('periodontal_level', 0)}단계
- 전체 구강 건강 점수: {analysis.get('total_score', 0)}점
- 권장 양치 빈도: 주 {analysis.get('recommend_brush_frequency', 0)}회
- 관리 가이드: {analysis.get('recommend_guide', '없음')}
"""
        
        # 추천 제품 정보
        product_info = ""
        if products.get('recommended_brush'):
            brush = products['recommended_brush']
            product_info += f"- 추천 칫솔: {brush.get('product_name', '')} ({brush.get('brand', '')})\n"
        
        if products.get('recommended_toothpaste'):
            paste = products['recommended_toothpaste']
            product_info += f"- 추천 치약: {paste.get('product_name', '')} ({paste.get('brand', '')})\n"
        
        if products.get('recommended_other'):
            other = products['recommended_other']
            product_info += f"- 추천 케어제품: {other.get('product_name', '')} ({other.get('brand', '')})\n"
        
        # 대화 히스토리 요약 (최근 3개만)
        history_summary = ""
        if conversation_history and len(conversation_history) > 1:
            recent_history = conversation_history[-5:]  # 최근 5개 대화만
            history_items = []
            for conv in recent_history[:-1]:  # 현재 질문 제외
                if not conv.get('validation_error'):  # 유효한 대화만
                    history_items.append(f"이전 질문({conv.get('category', '미상')}): {conv.get('user', '')[:50]}...")
            
            if history_items:
                history_summary = f"\n이전 대화 내용:\n" + "\n".join(history_items) + "\n"
        
        # 컨텍스트 결합
        context_text = "\n\n".join(contexts[:3]) if contexts else "관련 전문 지식을 찾을 수 없습니다."
        
        prompt = ChatPromptTemplate.from_template(
            """당신은 반려동물 구강 건강 전문 상담사입니다. 사용자의 질문에 대해 전문적이고 도움이 되는 답변을 제공해주세요.

반려동물 정보:
- 이름: {pet_name}
- 품종: {breed}
- 나이: {age}세
- 체중: {weight}kg
- 성별: {gender}

{analysis_summary}

추천 제품:
{product_info}

{history_summary}

질문 분류: {category}
현재 질문: "{question}"

전문 지식 자료:
{context}

답변 가이드라인:
1. 최근 구강 검진 결과를 참고하여 개인화된 조언을 제공하세요
2. 전문 지식 자료의 내용을 근거로 답변하세요
3. 이전 대화 내용이 있다면 연관성을 고려하여 일관된 조언을 하세요
4. 구체적이고 실용적인 조언을 포함하세요
5. 심각한 증상이나 응급상황에는 수의사 상담을 권하세요
6. 추천된 제품이 관련 있다면 언급하세요
7. 친근하고 이해하기 쉬운 언어로 답변하세요
8. 답변은 3-5문장으로 구성하세요

답변:"""
        )
        
        try:
            response = self.llm.invoke(
                prompt.format(
                    pet_name=pet_info.get('name', '반려견'),
                    breed=pet_info.get('breed', '미상'),
                    age=pet_info.get('age_years', '미상'),
                    weight=pet_info.get('weight', '미상'),
                    gender=pet_info.get('gender', '미상'),
                    analysis_summary=analysis_summary,
                    product_info=product_info if product_info else "추천 제품 정보가 없습니다.",
                    history_summary=history_summary,
                    category=category,
                    question=question,
                    context=context_text
                )
            )
            
            return response.content
            
        except Exception as e:
            print(f"응답 생성 오류: {e}")
            return "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다. 다시 시도해주세요."
    
    def generate_invalid_category_response(self, question: str, selected_category: str, 
                                         suggestion: str, confidence: float) -> str:
        """선택한 분류에 부적합한 질문일 때의 응답 생성"""
        
        if confidence < 0.2:
            return f"""
죄송합니다. 질문하신 내용 "{question[:50]}..." 이 선택하신 '{selected_category}' 분류와 관련이 없는 것 같습니다.

'{selected_category}' 분류에서는 다음과 같은 질문을 해주세요:

**질병**: "충치가 있는지 확인하는 방법은?", "잇몸이 부었어요", "입냄새가 심해요"
**식습관**: "치아에 좋은 간식은?", "딱딱한 뼈를 줘도 되나요?", "어떤 사료가 좋나요?"
**양치법**: "양치하는 방법을 알려주세요", "양치를 싫어해요", "얼마나 자주 해야 하나요?"
**이상행동**: "계속 입을 비벼요", "침을 많이 흘려요", "이상하게 씹어요"

'{selected_category}' 분류에 맞는 구체적인 질문을 다시 해주시면 정확한 답변을 드리겠습니다!
"""
        else:
            return f"""
질문해주신 내용이 '{selected_category}' 분류와 완전히 일치하지 않을 수 있습니다.

{suggestion if suggestion and suggestion != '-' else f"'{selected_category}' 분류에 더 적합한 질문으로 다시 작성해주세요."}

'{selected_category}' 분류에서 도움드릴 수 있는 주제들:

**질병**: 충치, 치주염, 잇몸염, 구취, 치아 질환 등
**식습관**: 치아에 좋은/나쁜 음식, 간식, 영양소, 씹을거리 등  
**양치법**: 양치 방법, 칫솔 선택, 관리 빈도, 저항 해결법 등
**이상행동**: 비정상적인 씹기, 핥기, 침흘리기, 입 비비기 등

더 구체적인 '{selected_category}' 관련 질문을 해주시면 정확한 답변을 드리겠습니다!
"""

class DentalChatbotSupervisor:
    """치아 관리 챗봇 워크플로우 관리 Supervisor (MemorySaver + thread_id)"""
    
    def __init__(self):
        self.analysis_agent = RecentAnalysisAgent()
        self.validator_agent = QuestionValidatorAgent()
        self.retrieval_agent = KoreanChatRetrievalAgent()
        self.response_agent = ChatResponseAgent()
        
        # MemorySaver 초기화 - 대화 히스토리 자동 관리
        self.memory = MemorySaver()
        
        # LangGraph 설정
        self.workflow = StateGraph(ChatbotState)
        self._setup_workflow()
    
    def _setup_workflow(self):
        """챗봇 워크플로우 노드 및 엣지 설정"""
        
        # 노드 추가
        self.workflow.add_node("get_recent_analysis", self._get_recent_analysis_node)
        self.workflow.add_node("validate_question", self._validate_question_node)
        self.workflow.add_node("retrieve_contexts", self._retrieve_contexts_node)
        self.workflow.add_node("generate_response", self._generate_response_node)
        self.workflow.add_node("generate_invalid_response", self._generate_invalid_response_node)
        
        # 시작점 설정
        self.workflow.set_entry_point("get_recent_analysis")
        
        # 엣지 연결
        self.workflow.add_edge("get_recent_analysis", "validate_question")
        
        # 조건부 엣지 - 분류 유효성에 따라 분기
        self.workflow.add_conditional_edges(
            "validate_question",
            self._decide_next_step,
            {
                "valid": "retrieve_contexts",
                "invalid": "generate_invalid_response"
            }
        )
        
        self.workflow.add_edge("retrieve_contexts", "generate_response")
        self.workflow.add_edge("generate_response", END)
        self.workflow.add_edge("generate_invalid_response", END)
        
        # 그래프 컴파일 (MemorySaver를 checkpointer로 추가)
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _get_recent_analysis_node(self, state: ChatbotState) -> ChatbotState:
        """최근 분석 정보 조회 노드"""
        print(f"🔍 최근 분석 정보 조회 중... (Guardian: {state['guardian_id']}, Pet: {state['pet_id']}, Thread: {state['thread_id']})")
        
        analysis_data = self.analysis_agent.get_recent_analysis(
            state['guardian_id'], state['pet_id']
        )
        
        if "error" in analysis_data:
            print(f"⚠️ 분석 정보 조회 실패: {analysis_data['error']}")
            state['recent_analysis'] = {}
            state['pet_info'] = {}
            state['product_recommendations'] = {}
        else:
            state['recent_analysis'] = analysis_data
            state['pet_info'] = analysis_data.get('pet_info', {})
            state['product_recommendations'] = analysis_data.get('products', {})
            
            print(f"✅ 분석 정보 로드 완료: {state['pet_info'].get('name', '미상')} "
                  f"(최근 검진: {analysis_data.get('analysis', {}).get('analysis_date', '미상')})")
        
        return state
    
    def _validate_question_node(self, state: ChatbotState) -> ChatbotState:
        """질문 적합성 검증 노드"""
        print(f"🤔 질문 적합성 검증 중... (선택한 분류: {state['selected_category']})")
        
        validation_result = self.validator_agent.validate_question_category(
            state['user_question'], state['selected_category']
        )
        
        state['is_valid_category'] = validation_result['is_valid']
        state['classification_confidence'] = validation_result['confidence']
        
        print(f"{'✅ 적합' if validation_result['is_valid'] else '❌ 부적합'} "
              f"(신뢰도: {validation_result['confidence']:.2f})")
        print(f"💡 판단 근거: {validation_result['reason']}")
        
        return state
    
    def _decide_next_step(self, state: ChatbotState) -> str:
        """다음 단계 결정 (분류 유효성 기반)"""
        return "valid" if state['is_valid_category'] else "invalid"
    
    def _retrieve_contexts_node(self, state: ChatbotState) -> ChatbotState:
        """컨텍스트 검색 노드"""
        print(f"🔍 관련 문서 검색 중... (분류: {state['selected_category']})")
        
        contexts = self.retrieval_agent.retrieve_documents(
            state['user_question'], 
            state['selected_category'],
            top_k=5
        )
        
        state['retrieved_contexts'] = contexts
        
        print(f"📚 검색 완료: {len(contexts)}개 문서")
        return state
    
    def _generate_response_node(self, state: ChatbotState) -> ChatbotState:
        """유효한 질문에 대한 응답 생성 노드"""
        print("🤖 챗봇 응답 생성 중...")
        
        response = self.response_agent.generate_response(
            state['user_question'],
            state['selected_category'],
            state['retrieved_contexts'],
            state['recent_analysis'],
            state['pet_info'],
            state.get('conversation_history', [])
        )
        
        state['chatbot_response'] = response
        
        # 대화 히스토리에 추가 (MemorySaver가 자동 관리)
        if 'conversation_history' not in state:
            state['conversation_history'] = []
        
        state['conversation_history'].append({
            "user": state['user_question'],
            "category": state['selected_category'],
            "assistant": response,
            "timestamp": datetime.now().isoformat(),
            "thread_id": state['thread_id']
        })
        
        print("응답 생성 완료")
        return state
    
    def _generate_invalid_response_node(self, state: ChatbotState) -> ChatbotState:
        """부적합한 질문에 대한 응답 생성 노드"""
        print(" 분류 부적합 응답 생성 중...")
        
        # 검증 결과에서 제안사항 가져오기
        validation_result = self.validator_agent.validate_question_category(
            state['user_question'], state['selected_category']
        )
        
        response = self.response_agent.generate_invalid_category_response(
            state['user_question'],
            state['selected_category'],
            validation_result.get('suggestion', ''),
            state['classification_confidence']
        )
        
        state['chatbot_response'] = response
        
        # 대화 히스토리에 추가
        if 'conversation_history' not in state:
            state['conversation_history'] = []
        
        state['conversation_history'].append({
            "user": state['user_question'],
            "category": state['selected_category'],
            "assistant": response,
            "timestamp": datetime.now().isoformat(),
            "thread_id": state['thread_id'],
            "validation_error": True
        })
        
        print("분류 부적합 응답 생성 완료")
        return state
    
    async def chat(self, guardian_id: int, pet_id: int, question: str, category: str, thread_id: str = None) -> Dict[str, Any]:
        """챗봇 대화 처리 (MemorySaver로 대화 히스토리 관리)"""
        
        # thread_id가 없으면 기본값 생성
        if not thread_id:
            thread_id = f"user_{guardian_id}_pet_{pet_id}"
        
        print(f"대화 시작 - Thread ID: {thread_id}")
        
        initial_state = ChatbotState(
            guardian_id=guardian_id,
            pet_id=pet_id,
            thread_id=thread_id,
            recent_analysis={},
            pet_info={},
            product_recommendations={},
            user_question=question,
            selected_category=category,
            is_valid_category=False,
            classification_confidence=0.0,
            retrieved_contexts=[],
            chatbot_response="",
            conversation_history=[],
            next_agent=""
        )
        
        # config에 thread_id 추가 - MemorySaver가 이를 사용해 대화 히스토리 관리
        config = {"configurable": {"thread_id": thread_id}}
        
        # 워크플로우 실행 (thread_id와 함께)
        final_state = await self.app.ainvoke(initial_state, config=config)
        
        # 결과 정리
        return {
            "response": final_state['chatbot_response'],
            "category": final_state['selected_category'],
            "is_valid_category": final_state['is_valid_category'],
            "confidence": final_state['classification_confidence'],
            "contexts_found": len(final_state['retrieved_contexts']),
            "pet_info": final_state['pet_info'],
            "recent_analysis": final_state['recent_analysis'],
            "conversation_history": final_state['conversation_history'],
            "thread_id": thread_id
        }
    
    async def get_conversation_history(self, guardian_id: int, pet_id: int, thread_id: str = None) -> List[Dict[str, Any]]:
        """대화 히스토리 조회 (MemorySaver에서 자동 관리되는 히스토리)"""
        
        if not thread_id:
            thread_id = f"user_{guardian_id}_pet_{pet_id}"
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # 해당 thread의 체크포인트 히스토리 조회
            history = []
            async for checkpoint in self.app.aget_state_history(config):
                if checkpoint.values and 'conversation_history' in checkpoint.values:
                    history.extend(checkpoint.values['conversation_history'])
            
            # 중복 제거 및 시간순 정렬
            unique_history = []
            seen = set()
            for conv in sorted(history, key=lambda x: x.get('timestamp', '')):
                conv_key = f"{conv.get('timestamp')}_{conv.get('user', '')[:50]}"
                if conv_key not in seen:
                    unique_history.append(conv)
                    seen.add(conv_key)
            
            return unique_history
        except Exception as e:
            print(f"대화 히스토리 조회 오류: {e}")
            return []
    
    async def get_current_state(self, thread_id: str) -> Dict[str, Any]:
        """현재 대화 상태 조회"""
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            state = await self.app.aget_state(config)
            return state.values if state else {}
        except Exception as e:
            print(f"현재 상태 조회 오류: {e}")
            return {}
    
    async def clear_conversation_history(self, guardian_id: int, pet_id: int, thread_id: str = None) -> str:
        """대화 히스토리 초기화 (새로운 thread_id 생성으로 우회)"""
        
        if not thread_id:
            thread_id = f"user_{guardian_id}_pet_{pet_id}"
        
        # MemorySaver는 직접 삭제가 제한적이므로 새로운 thread_id 생성
        new_thread_id = f"user_{guardian_id}_pet_{pet_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"대화 초기화: {thread_id} → {new_thread_id}")
        return new_thread_id

# 사용 예시
async def main():
    # 챗봇 초기화
    chatbot = DentalChatbotSupervisor()
    
    # 테스트 데이터
    guardian_id = 1
    pet_id = 1
    thread_id = f"test_user_{guardian_id}_pet_{pet_id}"
    
    # 시나리오별 테스트 (동일한 thread_id로 연속 대화)
    test_scenarios = [
        ("우리 강아지가 충치가 있는 것 같은데 어떻게 확인할 수 있나요?", "질병"),
        ("양치를 할 때 강아지가 너무 싫어해서 힘든데 어떻게 훈련시킬까요?", "양치법"),
        ("어떤 간식을 주면 치아에 좋을까요?", "식습관"),
        ("산책할 때 목줄은 어떤 걸 사용하는 게 좋나요?", "질병"),  # 부적합한 질문
        ("요즘 계속 입을 비비고 침을 많이 흘리는데 괜찮나요?", "이상행동"),
    ]
    
    print("="*80)
    print("LangGraph + MemorySaver 치아 관리 챗봇 테스트")
    print(f"Thread ID: {thread_id}")
    print("="*80)
    
    for i, (question, category) in enumerate(test_scenarios, 1):
        print(f"\n[테스트 {i}] 선택한 분류: {category}")
        print(f"질문: {question}")
        print("-" * 60)
        
        try:
            # 동일한 thread_id로 연속 대화 (대화 히스토리 유지)
            result = await chatbot.chat(guardian_id, pet_id, question, category, thread_id)
            
            print(f"분류 적합성: {'적합' if result['is_valid_category'] else '부적합'}")
            print(f"검증 신뢰도: {result['confidence']:.2f}")
            print(f"검색된 문서 수: {result['contexts_found']}개")
            print(f"Thread ID: {result['thread_id']}")
            
            if result['pet_info']:
                pet = result['pet_info']
                print(f"반려동물: {pet.get('name', '미상')} ({pet.get('breed', '미상')}, {pet.get('age_years', '미상')}세)")
            
            print(f"\n챗봇 응답:")
            print(result['response'])
            
        except Exception as e:
            print(f"오류 발생: {e}")
        
        print("\n" + "="*80)
    
    # 대화 히스토리 확인
    print(f"\n[대화 히스토리 조회] Thread ID: {thread_id}")
    print("-" * 60)
    try:
        history = await chatbot.get_conversation_history(guardian_id, pet_id, thread_id)
        print(f" 총 대화 기록: {len(history)}개")
        
        for i, conv in enumerate(history, 1):
            print(f"  {i}. [{conv.get('category', '미상')}] {conv.get('user', '')[:50]}...")
            print(f"      {conv.get('timestamp', '')}")
            if conv.get('validation_error'):
                print(f"     ⚠️ 검증 오류")
            print()
            
    except Exception as e:
        print(f" 히스토리 조회 오류: {e}")

if __name__ == "__main__":
    asyncio.run(main())