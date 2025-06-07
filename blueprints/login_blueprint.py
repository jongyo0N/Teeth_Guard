"""
Login Blueprint - 로그인 관련 API
이메일 또는 닉네임으로 로그인 가능
"""
from flask import Blueprint, request, jsonify, session
from functionalModules.models import Guardian, Pet
from functionalModules.validators import ValidationError, Validator
from typing import Any, Dict, Optional
import hashlib
import re

# Blueprint 생성
login_bp = Blueprint('login', __name__, url_prefix='/api/login')

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

def is_email(identifier: str) -> bool:
    """식별자가 이메일인지 닉네임인지 판별"""
    return '@' in identifier and Validator.validate_email(identifier)

# ===== 로그인 관련 API =====
@login_bp.route('/', methods=['POST'])
def login():
    """
    로그인 API
    이메일로 로그인 (프론트 요청에 맞춤, 세션 저장 없음)
    """
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        # remember_me는 더 이상 사용하지 않음

        print(f"📨 로그인 요청: email={email}")

        # 입력값 검증
        if not email or not password:
            return error_response("이메일과 비밀번호를 입력해주세요.", 400)

        # 이메일로 보호자 조회
        guardian = Guardian.get_by_email(email)
        if not guardian:
            return error_response("존재하지 않는 이메일입니다.", 401)

        # 비밀번호 확인 (평문 비교)
        if guardian.get('password_hash') != password:
            return error_response("비밀번호가 일치하지 않습니다.", 401)

        # 보호자의 반려동물 목록 조회
        pets = Pet.get_all_by_guardian(guardian['guardian_id'], include_photos=False)

        response_data = {
            "guardian_id": guardian['guardian_id'],
            "pets": pets,
        }

        print(f"✅ 로그인 성공: {guardian['name']} (email: {email})")

        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ 로그인 오류: {e}")
        return error_response("로그인 처리 중 오류가 발생했습니다.", 500)


