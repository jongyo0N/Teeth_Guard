"""
월 단위 피드백 생성 시스템 (LangGraph + Direct Generation)
- 양치/케어 실천률 조회
- 진단 점수 (total_score) 활용
- RAG 없이 직접 피드백 생성
- 각각 3문장 제한 피드백
"""
import os
import mysql.connector
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime, date, timedelta
import asyncio
import calendar

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 환경 변수 설정
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_USER = os.environ.get("MYSQL_USER")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
MYSQL_DATABASE = "pet_dental_care"

class MonthlyFeedbackState(TypedDict):
    """월 단위 피드백을 위한 State 클래스"""
    guardian_id: int
    pet_id: int
    target_year: int
    target_month: int
    
    # 조회 데이터
    pet_info: Dict[str, Any]
    monthly_stats: Dict[str, Any]  # 실천률 통계
    diagnosis_scores: List[Dict[str, Any]]  # 월간 진단 점수들
    
    # 분석 결과
    performance_analysis: Dict[str, Any]
    monthly_feedback: Dict[str, str]  # 양치/케어 각각 3문장
    
    messages: List[Dict[str, Any]]
    next_agent: str

class MonthlyStatsAgent:
    """월간 통계 조회 Agent"""
    
    def __init__(self):
        self.connection_config = {
            'host': MYSQL_HOST,
            'user': MYSQL_USER,
            'password': MYSQL_PASSWORD,
            'database': MYSQL_DATABASE
        }
    
    def get_pet_basic_info(self, guardian_id: int, pet_id: int) -> Dict[str, Any]:
        """펫 기본 정보 조회"""
        try:
            conn = mysql.connector.connect(**self.connection_config)
            cursor = conn.cursor(dictionary=True)
            
            query = """
            SELECT name, breed, weight, birth_date, gender
            FROM pets 
            WHERE guardian_id = %s AND pet_id = %s
            """
            
            cursor.execute(query, (guardian_id, pet_id))
            result = cursor.fetchone()
            
            if result:
                # 나이 계산
                if result['birth_date']:
                    today = date.today()
                    birthdate = result['birth_date']
                    age_years = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
                    result['age_years'] = age_years
                
                # 크기 분류
                weight = result.get('weight', 0)
                if weight <= 10:
                    result['size_category'] = '소형'
                elif weight <= 25:
                    result['size_category'] = '중형'
                else:
                    result['size_category'] = '대형'
                
                return result
            return {}
                
        except mysql.connector.Error as e:
            print(f"MySQL 연결 오류: {e}")
            return {}
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
    
    def get_monthly_completion_rates(self, guardian_id: int, pet_id: int, 
                                   year: int, month: int) -> Dict[str, Any]:
        """월간 양치/케어 실천률 조회"""
        try:
            conn = mysql.connector.connect(**self.connection_config)
            cursor = conn.cursor(dictionary=True)
            
            # 해당 월의 첫째 날과 마지막 날
            first_day = date(year, month, 1)
            last_day = date(year, month, calendar.monthrange(year, month)[1])
            
            # 양치 실천률 조회
            brush_query = """
            SELECT 
                COUNT(*) as total_scheduled,
                SUM(CASE WHEN is_completed = 1 THEN 1 ELSE 0 END) as completed_count,
                ROUND(
                    (SUM(CASE WHEN is_completed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 
                    1
                ) as completion_rate
            FROM schedule_dates sd
            WHERE sd.guardian_id = %s 
            AND sd.pet_id = %s 
            AND sd.schedule_type = '양치'
            AND sd.scheduled_date BETWEEN %s AND %s
            """
            
            cursor.execute(brush_query, (guardian_id, pet_id, first_day, last_day))
            brush_result = cursor.fetchone()
            
            # 케어 실천률 조회
            care_query = """
            SELECT 
                COUNT(*) as total_scheduled,
                SUM(CASE WHEN is_completed = 1 THEN 1 ELSE 0 END) as completed_count,
                ROUND(
                    (SUM(CASE WHEN is_completed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 
                    1
                ) as completion_rate
            FROM schedule_dates sd
            WHERE sd.guardian_id = %s 
            AND sd.pet_id = %s 
            AND sd.schedule_type = '케어'
            AND sd.scheduled_date BETWEEN %s AND %s
            """
            
            cursor.execute(care_query, (guardian_id, pet_id, first_day, last_day))
            care_result = cursor.fetchone()
            
            return {
                "brush_stats": {
                    "total_scheduled": brush_result['total_scheduled'] or 0,
                    "completed_count": brush_result['completed_count'] or 0,
                    "completion_rate": brush_result['completion_rate'] or 0.0
                },
                "care_stats": {
                    "total_scheduled": care_result['total_scheduled'] or 0,
                    "completed_count": care_result['completed_count'] or 0,
                    "completion_rate": care_result['completion_rate'] or 0.0
                },
                "target_period": f"{year}년 {month}월"
            }
                
        except mysql.connector.Error as e:
            print(f"MySQL 연결 오류: {e}")
            return {}
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
    
    def get_monthly_diagnosis_scores(self, guardian_id: int, pet_id: int,
                                   year: int, month: int) -> List[Dict[str, Any]]:
        """월간 진단 점수들 조회"""
        try:
            conn = mysql.connector.connect(**self.connection_config)
            cursor = conn.cursor(dictionary=True)
            
            # 해당 월의 첫째 날과 마지막 날
            first_day = date(year, month, 1)
            last_day = date(year, month, calendar.monthrange(year, month)[1])
            
            query = """
            SELECT 
                DATE(analysis_date) as diagnosis_date,
                total_score,
                caries_percentage,
                calculus_percentage,
                periodontal_level
            FROM dental_analysis
            WHERE guardian_id = %s AND pet_id = %s
            AND DATE(analysis_date) BETWEEN %s AND %s
            ORDER BY analysis_date DESC
            """
            
            cursor.execute(query, (guardian_id, pet_id, first_day, last_day))
            results = cursor.fetchall()
            
            return results
                
        except mysql.connector.Error as e:
            print(f"MySQL 연결 오류: {e}")
            return []
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

class PerformanceAnalysisAgent:
    """실천률 및 진단 점수 분석 Agent"""
    
    def analyze_monthly_performance(self, monthly_stats: Dict[str, Any], 
                                  diagnosis_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """월간 성과 분석"""
        
        # 실천률 분석
        brush_rate = monthly_stats.get('brush_stats', {}).get('completion_rate', 0)
        care_rate = monthly_stats.get('care_stats', {}).get('completion_rate', 0)
        
        # 실천률 등급 매기기
        def get_performance_grade(rate: float) -> str:
            if rate >= 90: return "최우수"
            elif rate >= 80: return "우수"
            elif rate >= 70: return "양호"
            elif rate >= 50: return "보통"
            else: return "개선필요"
        
        brush_grade = get_performance_grade(brush_rate)
        care_grade = get_performance_grade(care_rate)
        
        # 진단 점수 분석
        diagnosis_analysis = {}
        if diagnosis_scores:
            # 최신 점수와 평균 점수
            latest_score = diagnosis_scores[0].get('total_score', 0) if diagnosis_scores else 0
            avg_score = sum(float(d.get('total_score', 0)) for d in diagnosis_scores) / len(diagnosis_scores)
            
            # 점수 변화 추이 (최신 vs 이전)
            score_trend = "유지"
            if len(diagnosis_scores) > 1:
                prev_score = diagnosis_scores[1].get('total_score', 0)
                if latest_score > prev_score:
                    score_trend = "개선"
                elif latest_score < prev_score:
                    score_trend = "악화"
            
            diagnosis_analysis = {
                "latest_score": float(latest_score),
                "average_score": round(avg_score, 2),
                "score_trend": score_trend,
                "diagnosis_count": len(diagnosis_scores)
            }
        else:
            diagnosis_analysis = {
                "latest_score": 0,
                "average_score": 0,
                "score_trend": "데이터없음",
                "diagnosis_count": 0
            }
        
        # 종합 평가
        overall_assessment = "보통"
        if brush_rate >= 80 and care_rate >= 80 and diagnosis_analysis["latest_score"] >= 80:
            overall_assessment = "우수"
        elif brush_rate >= 70 and care_rate >= 70 and diagnosis_analysis["latest_score"] >= 70:
            overall_assessment = "양호"
        elif brush_rate < 50 or care_rate < 50 or diagnosis_analysis["latest_score"] < 50:
            overall_assessment = "개선필요"
        
        return {
            "brush_performance": {
                "rate": brush_rate,
                "grade": brush_grade,
                "completed": monthly_stats.get('brush_stats', {}).get('completed_count', 0),
                "scheduled": monthly_stats.get('brush_stats', {}).get('total_scheduled', 0)
            },
            "care_performance": {
                "rate": care_rate,
                "grade": care_grade,
                "completed": monthly_stats.get('care_stats', {}).get('completed_count', 0),
                "scheduled": monthly_stats.get('care_stats', {}).get('total_scheduled', 0)
            },
            "diagnosis_analysis": diagnosis_analysis,
            "overall_assessment": overall_assessment
        }

class MonthlyFeedbackAgent:
    """월간 피드백 생성 Agent (각각 3문장 제한)"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    def generate_brushing_feedback(self, pet_info: Dict, performance: Dict, 
                                 target_period: str) -> str:
        """양치 실천률 기반 3문장 피드백 생성"""
        
        brush_perf = performance['brush_performance']
        diagnosis = performance['diagnosis_analysis']
        
        prompt = ChatPromptTemplate.from_template(
            """당신은 반려동물 치과 전문의로서 보호자에게 월간 양치 피드백을 제공합니다.

반려견 정보:
- 이름: {pet_name}
- 품종: {breed}
- 나이: {age}세
- 크기: {size_category}견
- 체중: {weight}kg

월간 양치 성과 ({target_period}):
- 실천률: {completion_rate}%
- 등급: {grade}
- 완료: {completed}/{scheduled}회
- 최신 구강 점수: {latest_score}/100점
- 점수 변화: {score_trend}

**중요**: 정확히 3문장으로 응답해주세요. 더 많거나 적으면 안됩니다.

피드백 구조:
1. 이번 달 양치 실천률과 성과에 대한 인정
2. 이 실천률이 구강 건강에 미치는 영향 설명 (진단 점수 연결)
3. 다음 달을 위한 구체적인 격려나 개선 제안

**좋은 예시들:**

예시 1 (우수한 경우 - 실천률 85%, 점수 상승):
"11월 양치 실천률이 85%로 매우 우수한 성과를 보여주셨네요. 이런 꾸준한 양치 습관 덕분에 구강 점수가 78점으로 상승했고, {pet_name}의 잇몸 건강이 눈에 띄게 개선되고 있습니다. 다음 달에는 90% 목표로 도전해서 더욱 건강한 치아를 만들어보세요."

예시 2 (보통인 경우 - 실천률 65%, 점수 유지):
"11월 양치 실천률이 65%로 평균 수준을 유지하고 계시네요. 현재 구강 점수 72점을 유지하고 있지만, 조금 더 규칙적인 양치를 통해 개선의 여지가 있어 보입니다. 주 5회 이상 목표로 양치 빈도를 늘려보시면 {pet_name}의 구강 건강이 한층 좋아질 거예요."

예시 3 (개선 필요한 경우 - 실천률 45%, 점수 하락):
"11월 양치 실천률이 45%로 목표에 다소 못 미쳤지만, 완전히 포기할 필요는 없어요. 실천률 저하로 구강 점수가 65점으로 하락했지만, 지금부터라도 꾸준히 관리하면 충분히 회복 가능합니다. 하루 한 번씩이라도 짧게나마 양치를 시도해보시고, 어려우면 치아 와이프로라도 시작해보세요."


작성 가이드라인:
- 자연스러운 한국어로 대화하듯 작성
- 성과에 관계없이 격려하는 톤 유지
- 양치 습관과 구강 건강 결과를 직접 연결
- 각 문장은 완전하고 의미있는 내용 포함
- 번호나 목록 형태 사용 금지

정확히 3문장으로 한국어 피드백을 작성해주세요."""
        )
        
        try:
            response = self.llm.invoke(
                prompt.format(
                    pet_name=pet_info.get('name', '반려견'),
                    breed=pet_info.get('breed', '미상'),
                    age=pet_info.get('age_years', '미상'),
                    size_category=pet_info.get('size_category', '중형'),
                    weight=pet_info.get('weight', '미상'),
                    target_period=target_period,
                    completion_rate=brush_perf['rate'],
                    grade=brush_perf['grade'],
                    completed=brush_perf['completed'],
                    scheduled=brush_perf['scheduled'],
                    latest_score=diagnosis['latest_score'],
                    score_trend=diagnosis['score_trend']
                )
            )
            
            return response.content
            
        except Exception as e:
            print(f"양치 피드백 생성 오류: {e}")
            return f"{target_period} 양치 실천률은 {brush_perf['rate']}%로 {brush_perf['grade']} 등급을 기록했습니다. 꾸준한 양치 습관이 {pet_info.get('name', '반려견')}의 구강 건강 개선에 큰 도움이 되고 있어요. 다음 달에도 현재 수준을 잘 유지하며 더 나은 결과를 만들어보세요."
    
    def generate_care_feedback(self, pet_info: Dict, performance: Dict,
                             target_period: str) -> str:
        """케어 실천률 기반 3문장 피드백 생성"""
        
        care_perf = performance['care_performance']
        diagnosis = performance['diagnosis_analysis']
        overall = performance['overall_assessment']
        
        prompt = ChatPromptTemplate.from_template(
            """당신은 반려동물 치과 전문의로서 보호자에게 월간 구강 케어 피드백을 제공합니다.

반려견 정보:
- 이름: {pet_name}
- 품종: {breed}
- 나이: {age}세
- 크기: {size_category}견
- 체중: {weight}kg

월간 케어 성과 ({target_period}):
- 실천률: {completion_rate}%
- 등급: {grade}
- 완료: {completed}/{scheduled}회
- 최신 구강 점수: {latest_score}/100점
- 종합 평가: {overall_assessment}

**중요**: 정확히 3문장으로 응답해주세요. 더 많거나 적으면 안됩니다.

피드백 구조:
1. 이번 달 구강 케어 실천률과 성과에 대한 인정
2. 종합적인 구강 케어(양치 외 활동)가 전체 구강 건강에 미치는 영향
3. 케어 루틴 유지/개선을 위한 구체적인 동기부여나 가이드

**좋은 예시들:**

예시 1 (우수한 경우 - 실천률 88%, 종합 평가 우수):
"11월 구강 케어 실천률이 88%로 정말 훌륭한 성과를 거두셨어요. 양치뿐만 아니라 치아 간식, 장난감 활용 등 다양한 케어 덕분에 {pet_name}의 전반적인 구강 건강이 크게 향상되었습니다. 이런 종합적인 관리 방식을 다음 달에도 꾸준히 유지해주시면 더욱 건강한 치아를 오래 보존할 수 있을 거예요."

예시 2 (양호한 경우 - 실천률 72%, 종합 평가 양호):
"11월 구강 케어 실천률이 72%로 양호한 수준을 보여주고 계시네요. 양치만으로는 해결하기 어려운 치석 제거나 잇몸 마사지 효과를 케어 활동을 통해 보완하고 있어 다행입니다. 덴탈 츄나 로프 장난감 같은 케어 아이템을 조금 더 자주 활용해보시면 80% 이상 달성도 충분히 가능할 것 같아요."

예시 3 (개선 필요한 경우 - 실천률 48%, 종합 평가 보통):
"11월 구강 케어 실천률이 48%로 아직 개선의 여지가 많이 남아있어요. 양치만으로는 한계가 있기 때문에 치아 간식이나 구강 청정제 같은 보조 케어가 {pet_name}의 구강 건강 전체를 좌우합니다. 무리하지 말고 주 2-3회부터 시작해서 {pet_name}가 좋아하는 케어 방법을 찾아보시는 것은 어떨까요."

작성 가이드라인:
- 자연스러운 한국어로 대화하듯 작성
- 양치 외 다양한 케어 활동의 중요성 강조
- 장기적인 구강 건강 유지 관점에서 조언
- 성과에 관계없이 격려하는 톤 유지
- 각 문장은 완전하고 의미있는 내용 포함
- 번호나 목록 형태 사용 금지

정확히 3문장으로 한국어 피드백을 작성해주세요."""
        )
        
        try:
            response = self.llm.invoke(
                prompt.format(
                    pet_name=pet_info.get('name', '반려견'),
                    breed=pet_info.get('breed', '미상'),
                    age=pet_info.get('age_years', '미상'),
                    size_category=pet_info.get('size_category', '중형'),
                    weight=pet_info.get('weight', '미상'),
                    target_period=target_period,
                    completion_rate=care_perf['rate'],
                    grade=care_perf['grade'],
                    completed=care_perf['completed'],
                    scheduled=care_perf['scheduled'],
                    latest_score=diagnosis['latest_score'],
                    overall_assessment=overall
                )
            )
            
            return response.content
            
        except Exception as e:
            print(f"케어 피드백 생성 오류: {e}")
            return f"{target_period} 구강 케어 실천률은 {care_perf['rate']}%로 {care_perf['grade']} 등급을 달성했습니다. 양치 외에도 다양한 케어 활동이 {pet_info.get('name', '반려견')}의 전반적인 구강 건강 유지에 매우 중요한 역할을 하고 있어요. 다음 달에는 케어 빈도를 조금 더 늘려서 더욱 건강한 치아를 만들어보시는 것은 어떨까요."

class MonthlyFeedbackSupervisor:
    """월 단위 피드백 생성 Supervisor"""
    
    def __init__(self):
        self.stats_agent = MonthlyStatsAgent()
        self.analysis_agent = PerformanceAnalysisAgent()
        self.feedback_agent = MonthlyFeedbackAgent()
        
        # LangGraph 설정
        self.workflow = StateGraph(MonthlyFeedbackState)
        self._setup_workflow()
    
    def _setup_workflow(self):
        """월간 피드백 워크플로우 설정"""
        
        # 노드 추가
        self.workflow.add_node("get_pet_info", self._get_pet_info_node)
        self.workflow.add_node("get_monthly_stats", self._get_monthly_stats_node)
        self.workflow.add_node("get_diagnosis_scores", self._get_diagnosis_scores_node)
        self.workflow.add_node("analyze_performance", self._analyze_performance_node)
        self.workflow.add_node("generate_feedback", self._generate_feedback_node)
        
        # 시작점 설정
        self.workflow.set_entry_point("get_pet_info")
        
        # 엣지 연결
        self.workflow.add_edge("get_pet_info", "get_monthly_stats")
        self.workflow.add_edge("get_monthly_stats", "get_diagnosis_scores")
        self.workflow.add_edge("get_diagnosis_scores", "analyze_performance")
        self.workflow.add_edge("analyze_performance", "generate_feedback")
        self.workflow.add_edge("generate_feedback", END)
        
        # 그래프 컴파일
        self.app = self.workflow.compile()
    
    def _get_pet_info_node(self, state: MonthlyFeedbackState) -> MonthlyFeedbackState:
        """펫 정보 조회 노드"""
        print(f"펫 정보 조회 중... (Guardian: {state['guardian_id']}, Pet: {state['pet_id']})")
        
        pet_info = self.stats_agent.get_pet_basic_info(
            state['guardian_id'], state['pet_id']
        )
        state['pet_info'] = pet_info
        
        print(f"펫 정보: {pet_info.get('name', '미상')} ({pet_info.get('breed', '미상')})")
        return state
    
    def _get_monthly_stats_node(self, state: MonthlyFeedbackState) -> MonthlyFeedbackState:
        """월간 실천률 통계 조회 노드"""
        print(f"월간 통계 조회 중... ({state['target_year']}년 {state['target_month']}월)")
        
        monthly_stats = self.stats_agent.get_monthly_completion_rates(
            state['guardian_id'], state['pet_id'],
            state['target_year'], state['target_month']
        )
        state['monthly_stats'] = monthly_stats
        
        brush_rate = monthly_stats.get('brush_stats', {}).get('completion_rate', 0)
        care_rate = monthly_stats.get('care_stats', {}).get('completion_rate', 0)
        print(f"실천률: 양치 {brush_rate}%, 케어 {care_rate}%")
        return state
    
    def _get_diagnosis_scores_node(self, state: MonthlyFeedbackState) -> MonthlyFeedbackState:
        """월간 진단 점수 조회 노드"""
        print("월간 진단 점수 조회 중...")
        
        diagnosis_scores = self.stats_agent.get_monthly_diagnosis_scores(
            state['guardian_id'], state['pet_id'],
            state['target_year'], state['target_month']
        )
        state['diagnosis_scores'] = diagnosis_scores
        
        print(f"진단 기록: {len(diagnosis_scores)}건")
        return state
    
    def _analyze_performance_node(self, state: MonthlyFeedbackState) -> MonthlyFeedbackState:
        """성과 분석 노드"""
        print("월간 성과 분석 중...")
        
        performance_analysis = self.analysis_agent.analyze_monthly_performance(
            state['monthly_stats'], state['diagnosis_scores']
        )
        state['performance_analysis'] = performance_analysis
        
        overall = performance_analysis['overall_assessment']
        print(f"종합 평가: {overall}")
        return state
    
    def _generate_feedback_node(self, state: MonthlyFeedbackState) -> MonthlyFeedbackState:
        """월간 피드백 생성 노드 (각각 3문장)"""
        print("월간 피드백 생성 중... (각각 3문장)")
        
        pet_info = state['pet_info']
        performance = state['performance_analysis']
        target_period = state['monthly_stats'].get('target_period', f"{state['target_year']}년 {state['target_month']}월")
        
        # 양치 피드백 (3문장)
        print("  양치 피드백 생성 중...")
        brush_feedback = self.feedback_agent.generate_brushing_feedback(
            pet_info, performance, target_period
        )
        
        # 케어 피드백 (3문장)
        print("  케어 피드백 생성 중...")
        care_feedback = self.feedback_agent.generate_care_feedback(
            pet_info, performance, target_period
        )
        
        state['monthly_feedback'] = {
            '양치': brush_feedback,
            '케어': care_feedback
        }
        
        print("월간 피드백 생성 완료")
        return state
    
    async def generate_monthly_feedback(self, guardian_id: int, pet_id: int,
                                      target_year: int, target_month: int) -> Dict[str, Any]:
        """월 단위 피드백 생성"""
        
        initial_state = MonthlyFeedbackState(
            guardian_id=guardian_id,
            pet_id=pet_id,
            target_year=target_year,
            target_month=target_month,
            pet_info={},
            monthly_stats={},
            diagnosis_scores=[],
            performance_analysis={},
            monthly_feedback={},
            messages=[],
            next_agent=""
        )
        
        # 워크플로우 실행
        final_state = await self.app.ainvoke(initial_state)
        
        # 결과 정리
        result = {
            "pet_info": final_state['pet_info'],
            "target_period": f"{target_year}년 {target_month}월",
            "monthly_stats": final_state['monthly_stats'],
            "diagnosis_scores": final_state['diagnosis_scores'],
            "performance_analysis": final_state['performance_analysis'],
            "monthly_feedback": final_state['monthly_feedback']
        }
        
        return result

# 사용 예시
async def main():
    # Supervisor 초기화
    supervisor = MonthlyFeedbackSupervisor()
    
    # 현재 날짜 기준으로 이전 달 피드백 생성
    today = date.today()
    target_year = today.year
    target_month = today.month
    
    try:
        result = await supervisor.generate_monthly_feedback(
            guardian_id=1,
            pet_id=1,
            target_year=target_year,
            target_month=target_month
        )
        
        print("\n" + "="*80)
        print("월 단위 피드백 리포트 (LangGraph + Direct Generation)")
        print("="*80)
        
        # 반려견 정보
        pet_info = result['pet_info']
        print(f"\n 반려견 정보:")
        print(f"   이름: {pet_info.get('name', '미상')}")
        print(f"   품종: {pet_info.get('breed', '미상')}")
        print(f"   나이: {pet_info.get('age_years', '미상')}세")
        print(f"   크기: {pet_info.get('size_category', '미상')}견 ({pet_info.get('weight', '미상')}kg)")
        
        # 대상 기간
        print(f"\n 대상 기간: {result['target_period']}")
        
        # 월간 통계
        monthly_stats = result['monthly_stats']
        brush_stats = monthly_stats.get('brush_stats', {})
        care_stats = monthly_stats.get('care_stats', {})
        
        print(f"\n 월간 실천 통계:")
        print(f"   양치: {brush_stats.get('completed_count', 0)}/{brush_stats.get('total_scheduled', 0)}회 "
              f"({brush_stats.get('completion_rate', 0)}%)")
        print(f"   케어: {care_stats.get('completed_count', 0)}/{care_stats.get('total_scheduled', 0)}회 "
              f"({care_stats.get('completion_rate', 0)}%)")
        
        # 진단 점수
        diagnosis_scores = result['diagnosis_scores']
        if diagnosis_scores:
            print(f"\n 월간 진단 기록 ({len(diagnosis_scores)}건):")
            for i, score in enumerate(diagnosis_scores[:3]):  # 최근 3건만 표시
                print(f"   {i+1}. {score['diagnosis_date']}: {score['total_score']}점")
        else:
            print(f"\n 월간 진단 기록: 없음")
        
        # 성과 분석
        performance = result['performance_analysis']
        print(f"\n 성과 분석:")
        print(f"   양치 등급: {performance['brush_performance']['grade']}")
        print(f"   케어 등급: {performance['care_performance']['grade']}")
        print(f"   종합 평가: {performance['overall_assessment']}")
        
        if diagnosis_scores:
            diagnosis_analysis = performance['diagnosis_analysis']
            print(f"   최신 진단 점수: {diagnosis_analysis['latest_score']}점")
            print(f"   점수 변화: {diagnosis_analysis['score_trend']}")
        
        # 월간 피드백 (각각 정확히 3문장)
        monthly_feedback = result['monthly_feedback']
        print(f"\n 월간 피드백 (각각 3문장):")
        
        print(f"\n   [양치 피드백]")
        brush_sentences = monthly_feedback['양치'].split('.')
        for i, sentence in enumerate(brush_sentences):
            if sentence.strip():
                print(f"   {i+1}. {sentence.strip()}.")
        
        print(f"\n   [케어 피드백]")
        care_sentences = monthly_feedback['케어'].split('.')
        for i, sentence in enumerate(care_sentences):
            if sentence.strip():
                print(f"   {i+1}. {sentence.strip()}.")
        
    except Exception as e:
        print(f"월간 피드백 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())