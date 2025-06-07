"""
Signup Blueprint - 회원가입 관련 API
보호자 회원가입, 통합 회원가입, 중복 확인 등
"""
from flask import Blueprint, request, jsonify
from functionalModules.models import Guardian, Pet
from functionalModules.validators import ValidationError, Validator
from typing import Any, Dict, List
import hashlib

# Blueprint 생성
signup_bp = Blueprint('signup', __name__, url_prefix='/api/signup')

# 공통 응답 헬퍼
def success_response(message: str, data=None, status_code: int = 200):
    response = {'success': True, 'message': message}
    if data is not None:
        response['data'] = data
    return jsonify(response), status_code

def error_response(message: str, status_code: int = 400, errors=None):
    response = {'success': False, 'message': message}
    if errors:
        response['errors'] = errors
    return jsonify(response), status_code

def hash_password(password: str) -> str:
    """비밀번호 해시 생성"""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

# ===== 회원가입 관련 API =====

@signup_bp.route('/', methods=['POST'])
def guardian_signup():
    """
    보호자 단독 회원가입
    반려동물 정보 없이 보호자만 등록
    
    요청 예시:
    {
        "name": "홍길동",
        "nickname": "멍멍이주인",
        "email": "hong@example.com",
        "password": "password123",
        "phone": "010-1234-5678",
        "experience_level": "중급"
    }
    """
    try:
        data = request.get_json()
        print(f"📨 보호자 회원가입 요청: {data.get('name')} ({data.get('email')})")
        
        # 필수 필드 확인
        required_fields = ['name', 'email', 'password']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return error_response(f"필수 필드가 누락되었습니다: {', '.join(missing_fields)}", 400)
        
        # 비밀번호 최소 길이 확인
        if len(data.get('password', '')) < 6:
            return error_response("비밀번호는 6자 이상이어야 합니다.", 400)
        
        # 보호자 생성
        guardian_id = Guardian.create(data)
        
        # 생성된 보호자 정보 조회
        guardian_info = Guardian.get_by_id(guardian_id)
        
        print(f"✅ 보호자 회원가입 성공: {guardian_info.get('name')} (ID: {guardian_id})")
        
        return success_response(
            "보호자 회원가입이 성공적으로 완료되었습니다.",
            {
                "guardian_id": guardian_id,
                "name": guardian_info.get('name'),
                "nickname": guardian_info.get('nickname'),
                "email": guardian_info.get('email'),
                "experience_level": guardian_info.get('experience_level'),
                "registration_date": guardian_info.get('registration_date')
            },
            201
        )
        
    except ValidationError as e:
        print(f"❌ 유효성 검사 오류: {e}")
        return error_response(str(e), 400)
    except Exception as e:
        print(f"❌ 보호자 회원가입 오류: {e}")
        return error_response("회원가입 처리 중 오류가 발생했습니다.", 500)
