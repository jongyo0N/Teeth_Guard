"""
Flask Blueprint for Dental Chatbot API
LangGraph 기반 치아 관리 챗봇 시스템 API
"""

from flask import Blueprint, request, jsonify, session
from flask_cors import CORS
import asyncio
import traceback
import logging
from typing import Dict, Any, Optional
import re

# chatbotGraph 모듈에서 챗봇 클래스 import
from chatbotGraph import DentalChatbotSupervisor

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Blueprint 생성
chatbot_bp = Blueprint('chatbot', __name__, url_prefix='/api')

# CORS 설정
CORS(chatbot_bp)

# 전역 챗봇 인스턴스 (싱글톤)
chatbot_instance: Optional[DentalChatbotSupervisor] = None

def get_chatbot_instance() -> DentalChatbotSupervisor:
    """챗봇 인스턴스 싱글톤 패턴으로 관리"""
    global chatbot_instance
    if chatbot_instance is None:
        chatbot_instance = DentalChatbotSupervisor()
        logger.info("🤖 챗봇 인스턴스 초기화 완료")
    return chatbot_instance

def extract_category_from_message(message: str) -> tuple[str, str]:
    """메시지에서 [분류] 태그를 추출하고 제거"""
    # [분류] 패턴 찾기
    category_pattern = r'^\[([^\]]+)\]\s*'
    match = re.match(category_pattern, message)
    
    if match:
        category = match.group(1)
        clean_message = re.sub(category_pattern, '', message).strip()
        return clean_message, category
    
    return message, None

def validate_request_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """요청 데이터 유효성 검증"""
    required_fields = ['message']
    
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f"필수 필드 '{field}'가 누락되었습니다."
    
    # 메시지 길이 검증
    if len(data['message']) > 500:
        return False, "메시지가 너무 깁니다. (최대 500자)"
    
    # 카테고리 유효성 검증
    valid_categories = ['질병', '식습관', '양치법', '이상행동']
    if data.get('category') and data['category'] not in valid_categories:
        return False, f"유효하지 않은 카테고리입니다. 가능한 값: {valid_categories}"
    
    return True, ""

@chatbot_bp.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    챗봇 대화 API 엔드포인트
    
    Request Body:
    {
        "message": "사용자 질문",
        "category": "질병|식습관|양치법|이상행동" (optional),
        "guardian_id": 1 (optional, 기본값 세션에서 가져오거나 1),
        "pet_id": 1 (optional, 기본값 세션에서 가져오거나 1),
        "thread_id": "thread_123" (optional, 자동 생성)
    }
    
    Response:
    {
        "success": true,
        "response": "챗봇 응답 텍스트",
        "category": "분류",
        "is_valid_category": true,
        "confidence": 0.95,
        "contexts_found": 3,
        "thread_id": "user_1_pet_1",
        "pet_info": {...},
        "message": "성공적으로 처리되었습니다."
    }
    """
    try:
        # 요청 데이터 가져오기
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "JSON 데이터가 필요합니다.",
                "message": "요청 본문에 JSON 데이터를 포함해주세요."
            }), 400
        
        logger.info(f"📨 챗봇 요청 받음: {data}")
        
        # 데이터 유효성 검증
        is_valid, error_message = validate_request_data(data)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": error_message,
                "message": "요청 데이터를 확인해주세요."
            }), 400
        
        # 메시지에서 카테고리 추출 (프론트엔드에서 [카테고리] 형태로 보낸 경우)
        message = data['message']
        extracted_category = None
        
        if message.startswith('[') and ']' in message:
            message, extracted_category = extract_category_from_message(message)
        
        # 카테고리 결정 (우선순위: 추출된 카테고리 > 명시적 카테고리 > None)
        category = extracted_category or data.get('category')
        
        if not category:
            return jsonify({
                "success": False,
                "error": "카테고리가 지정되지 않았습니다.",
                "message": "질문의 카테고리를 선택해주세요. (질병, 식습관, 양치법, 이상행동)",
                "valid_categories": ['질병', '식습관', '양치법', '이상행동']
            }), 400
        
        # 사용자 정보 가져오기 (세션 또는 기본값)
        guardian_id = data.get('guardian_id') or session.get('guardian_id', 1)
        pet_id = data.get('pet_id') or session.get('pet_id', 1)
        thread_id = data.get('thread_id') or session.get('thread_id')
        
        logger.info(f"🔍 처리 정보 - Guardian: {guardian_id}, Pet: {pet_id}, Category: {category}")
        logger.info(f"💬 질문: {message}")
        
        # 챗봇 인스턴스 가져오기
        chatbot = get_chatbot_instance()
        
        # 비동기 함수를 동기적으로 실행
        def run_chat():
            return asyncio.run(chatbot.chat(
                guardian_id=guardian_id,
                pet_id=pet_id,
                question=message,
                category=category,
                thread_id=thread_id
            ))
        
        # 챗봇 실행
        result = run_chat()
        
        # thread_id를 세션에 저장 (향후 대화를 위해)
        session['thread_id'] = result['thread_id']
        session['guardian_id'] = guardian_id
        session['pet_id'] = pet_id
        
        logger.info(f"✅ 챗봇 응답 완료 - Thread: {result['thread_id']}")
        logger.info(f"🤖 응답: {result['response'][:100]}...")
        
        # 응답 구성
        response_data = {
            "success": True,
            "response": result['response'],
            "category": result['category'],
            "is_valid_category": result['is_valid_category'],
            "confidence": round(result['confidence'], 3),
            "contexts_found": result['contexts_found'],
            "thread_id": result['thread_id'],
            "pet_info": result.get('pet_info', {}),
            "message": "성공적으로 처리되었습니다."
        }
        
        # 분석 정보도 포함 (필요한 경우)
        if result.get('recent_analysis'):
            response_data['recent_analysis'] = result['recent_analysis']
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ 챗봇 처리 오류: {str(e)}")
        logger.error(f"🔍 상세 오류: {traceback.format_exc()}")
        
        return jsonify({
            "success": False,
            "error": "서버 내부 오류가 발생했습니다.",
            "message": "죄송합니다. 잠시 후 다시 시도해주세요.",
            "details": str(e) if chatbot_bp.debug else None
        }), 500

@chatbot_bp.route('/chat/history', methods=['GET'])
def get_chat_history():
    """
    대화 히스토리 조회 API
    
    Query Parameters:
    - guardian_id (optional)
    - pet_id (optional) 
    - thread_id (optional)
    
    Response:
    {
        "success": true,
        "history": [...],
        "count": 5,
        "thread_id": "user_1_pet_1"
    }
    """
    try:
        # 파라미터 가져오기
        guardian_id = request.args.get('guardian_id', type=int) or session.get('guardian_id', 1)
        pet_id = request.args.get('pet_id', type=int) or session.get('pet_id', 1)
        thread_id = request.args.get('thread_id') or session.get('thread_id')
        
        logger.info(f"📋 히스토리 요청 - Guardian: {guardian_id}, Pet: {pet_id}, Thread: {thread_id}")
        
        # 챗봇 인스턴스 가져오기
        chatbot = get_chatbot_instance()
        
        # 비동기 함수를 동기적으로 실행
        def run_get_history():
            return asyncio.run(chatbot.get_conversation_history(
                guardian_id=guardian_id,
                pet_id=pet_id,
                thread_id=thread_id
            ))
        
        history = run_get_history()
        
        logger.info(f"📊 히스토리 조회 완료: {len(history)}개 대화")
        
        return jsonify({
            "success": True,
            "history": history,
            "count": len(history),
            "thread_id": thread_id or f"user_{guardian_id}_pet_{pet_id}",
            "message": f"{len(history)}개의 대화 기록을 조회했습니다."
        }), 200
        
    except Exception as e:
        logger.error(f"❌ 히스토리 조회 오류: {str(e)}")
        
        return jsonify({
            "success": False,
            "error": "히스토리 조회 중 오류가 발생했습니다.",
            "message": "대화 기록을 불러올 수 없습니다.",
            "details": str(e) if chatbot_bp.debug else None
        }), 500

@chatbot_bp.route('/chat/clear', methods=['POST'])
def clear_chat_history():
    """
    대화 히스토리 초기화 API
    
    Request Body:
    {
        "guardian_id": 1 (optional),
        "pet_id": 1 (optional),
        "thread_id": "thread_123" (optional)
    }
    
    Response:
    {
        "success": true,
        "new_thread_id": "user_1_pet_1_20250607_143022",
        "message": "대화 기록이 초기화되었습니다."
    }
    """
    try:
        data = request.get_json() or {}
        
        # 파라미터 가져오기
        guardian_id = data.get('guardian_id') or session.get('guardian_id', 1)
        pet_id = data.get('pet_id') or session.get('pet_id', 1)
        thread_id = data.get('thread_id') or session.get('thread_id')
        
        logger.info(f"🔄 히스토리 초기화 요청 - Guardian: {guardian_id}, Pet: {pet_id}")
        
        # 챗봇 인스턴스 가져오기
        chatbot = get_chatbot_instance()
        
        # 비동기 함수를 동기적으로 실행
        def run_clear_history():
            return asyncio.run(chatbot.clear_conversation_history(
                guardian_id=guardian_id,
                pet_id=pet_id,
                thread_id=thread_id
            ))
        
        new_thread_id = run_clear_history()
        
        # 세션 업데이트
        session['thread_id'] = new_thread_id
        
        logger.info(f"✅ 히스토리 초기화 완료: {thread_id} → {new_thread_id}")
        
        return jsonify({
            "success": True,
            "new_thread_id": new_thread_id,
            "previous_thread_id": thread_id,
            "message": "대화 기록이 초기화되었습니다."
        }), 200
        
    except Exception as e:
        logger.error(f"❌ 히스토리 초기화 오류: {str(e)}")
        
        return jsonify({
            "success": False,
            "error": "히스토리 초기화 중 오류가 발생했습니다.",
            "message": "대화 기록 초기화에 실패했습니다.",
            "details": str(e) if chatbot_bp.debug else None
        }), 500

@chatbot_bp.route('/chat/status', methods=['GET'])
def get_chat_status():
    """
    챗봇 상태 확인 API
    
    Response:
    {
        "success": true,
        "status": "ready",
        "session_info": {...},
        "message": "챗봇이 준비되었습니다."
    }
    """
    try:
        # 챗봇 인스턴스 상태 확인
        chatbot = get_chatbot_instance()
        
        session_info = {
            "guardian_id": session.get('guardian_id'),
            "pet_id": session.get('pet_id'),
            "thread_id": session.get('thread_id'),
            "has_active_session": bool(session.get('thread_id'))
        }
        
        return jsonify({
            "success": True,
            "status": "ready",
            "session_info": session_info,
            "message": "챗봇이 준비되었습니다."
        }), 200
        
    except Exception as e:
        logger.error(f"❌ 상태 확인 오류: {str(e)}")
        
        return jsonify({
            "success": False,
            "status": "error",
            "error": "챗봇 상태 확인 중 오류가 발생했습니다.",
            "message": "챗봇 서비스에 문제가 발생했습니다.",
            "details": str(e) if chatbot_bp.debug else None
        }), 500

# 에러 핸들러
@chatbot_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "API 엔드포인트를 찾을 수 없습니다.",
        "message": "요청한 API 경로가 존재하지 않습니다."
    }), 404

@chatbot_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "허용되지 않은 HTTP 메서드입니다.",
        "message": "지원되는 HTTP 메서드를 확인해주세요."
    }), 405

@chatbot_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "서버 내부 오류가 발생했습니다.",
        "message": "잠시 후 다시 시도해주세요."
    }), 500

# Blueprint 등록을 위한 함수
def register_chatbot_blueprint(app):
    """Flask 앱에 챗봇 Blueprint 등록"""
    app.register_blueprint(chatbot_bp)
    logger.info("🔗 챗봇 Blueprint 등록 완료")
