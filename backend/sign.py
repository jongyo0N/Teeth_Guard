from flask import Flask, request, jsonify
from flask_cors import CORS
from functionalModules.models import Guardian, Pet
from functionalModules.validators import ValidationError
from functionalModules.database import DatabaseManager
from typing import Any, List

app = Flask(__name__)
CORS(app)

# 성공 응답 헬퍼
def success_response(message: str, data: Any = None, status_code: int = 200):
    response = {'success': True, 'message': message}
    if data is not None:
        response['data'] = data
    return jsonify(response), status_code

# 오류 응답 헬퍼
def error_response(message: str, status_code: int = 400, errors: List[str] = None):
    response = {'success': False, 'message': message}
    if errors:
        response['errors'] = errors
    return jsonify(response), status_code

# ===== 보호자 관련 API =====

@app.route('/api/guardian/signup', methods=['POST'])
def guardian_signup():
    """보호자 회원가입"""
    try:
        data = request.get_json()
        guardian_id = Guardian.create(data)
        
        return success_response(
            "회원가입이 성공적으로 완료되었습니다.",
            {"guardian_id": guardian_id},
            201
        )
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        print(f"회원가입 오류: {e}")
        return error_response("회원가입 처리 중 오류가 발생했습니다.", 500)

@app.route('/api/guardian/<int:guardian_id>', methods=['GET'])
def get_guardian(guardian_id):
    """보호자 정보 조회"""
    try:
        guardian = Guardian.get_by_id(guardian_id)
        if not guardian:
            return error_response("존재하지 않는 보호자입니다.", 404)
        
        return success_response("조회 성공", guardian)
    except Exception as e:
        print(f"보호자 조회 오류: {e}")
        return error_response("조회 중 오류가 발생했습니다.", 500)

@app.route('/api/guardian/<int:guardian_id>', methods=['PUT'])
def update_guardian(guardian_id):
    """보호자 정보 수정"""
    try:
        data = request.get_json()
        success = Guardian.update(guardian_id, data)
        
        if success:
            return success_response("정보가 성공적으로 수정되었습니다.")
        else:
            return error_response("수정할 정보가 없습니다.", 400)
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        print(f"보호자 수정 오류: {e}")
        return error_response("수정 처리 중 오류가 발생했습니다.", 500)

@app.route('/api/guardian/<int:guardian_id>', methods=['DELETE'])
def delete_guardian(guardian_id):
    """보호자 계정 삭제"""
    try:
        success = Guardian.delete(guardian_id)
        
        if success:
            return success_response("계정이 성공적으로 삭제되었습니다.")
        else:
            return error_response("존재하지 않는 보호자입니다.", 404)
    except Exception as e:
        print(f"보호자 삭제 오류: {e}")
        return error_response("삭제 처리 중 오류가 발생했습니다.", 500)

@app.route('/api/check-email', methods=['POST'])
def check_email():
    """이메일 중복 확인"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        
        if not email:
            return error_response("이메일을 입력해주세요.", 400)
        
        available = Guardian.check_email_availability(email)
        
        return success_response(
            "사용 가능한 이메일입니다." if available else "이미 사용 중인 이메일입니다.",
            {"available": available}
        )
    except Exception as e:
        print(f"이메일 확인 오류: {e}")
        return error_response("이메일 확인 중 오류가 발생했습니다.", 500)

# ===== 반려동물 관련 API =====

@app.route('/api/guardian/<int:guardian_id>/pet', methods=['POST'])
def create_pet(guardian_id):
    """반려동물 등록"""
    try:
        data = request.get_json()
        pet_id = Pet.create(guardian_id, data)
        
        return success_response(
            "반려동물이 성공적으로 등록되었습니다.",
            {"guardian_id": guardian_id, "pet_id": pet_id},
            201
        )
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        print(f"반려동물 등록 오류: {e}")
        return error_response("등록 처리 중 오류가 발생했습니다.", 500)

@app.route('/api/guardian/<int:guardian_id>/pets', methods=['GET'])
def get_guardian_pets(guardian_id):
    """보호자의 모든 반려동물 조회"""
    try:
        include_photos = request.args.get('include_photos', 'false').lower() == 'true'
        
        guardian = Guardian.get_by_id(guardian_id)
        if not guardian:
            return error_response("존재하지 않는 보호자입니다.", 404)
        
        pets = Pet.get_all_by_guardian(guardian_id, include_photos)
        
        return success_response(
            "조회 성공",
            {
                "guardian": guardian,
                "pets": pets,
                "pet_count": len(pets)
            }
        )
    except Exception as e:
        print(f"반려동물 목록 조회 오류: {e}")
        return error_response("조회 중 오류가 발생했습니다.", 500)

@app.route('/api/guardian/<int:guardian_id>/pet/<int:pet_id>', methods=['GET'])
def get_pet(guardian_id, pet_id):
    """반려동물 상세 조회"""
    try:
        include_photo = request.args.get('include_photo', 'false').lower() == 'true'
        
        pet = Pet.get_by_id(guardian_id, pet_id, include_photo)
        if not pet:
            return error_response("존재하지 않는 반려동물입니다.", 404)
        
        return success_response("조회 성공", pet)
    except Exception as e:
        print(f"반려동물 조회 오류: {e}")
        return error_response("조회 중 오류가 발생했습니다.", 500)

@app.route('/api/guardian/<int:guardian_id>/pet/<int:pet_id>', methods=['PUT'])
def update_pet(guardian_id, pet_id):
    """반려동물 정보 수정"""
    try:
        data = request.get_json()
        success = Pet.update(guardian_id, pet_id, data)
        
        if success:
            return success_response("반려동물 정보가 성공적으로 수정되었습니다.")
        else:
            return error_response("수정할 정보가 없습니다.", 400)
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        print(f"반려동물 수정 오류: {e}")
        return error_response("수정 처리 중 오류가 발생했습니다.", 500)

@app.route('/api/guardian/<int:guardian_id>/pet/<int:pet_id>', methods=['DELETE'])
def delete_pet(guardian_id, pet_id):
    """반려동물 삭제"""
    try:
        success = Pet.delete(guardian_id, pet_id)
        
        if success:
            return success_response("반려동물 정보가 성공적으로 삭제되었습니다.")
        else:
            return error_response("존재하지 않는 반려동물입니다.", 404)
    except Exception as e:
        print(f"반려동물 삭제 오류: {e}")
        return error_response("삭제 처리 중 오류가 발생했습니다.", 500)

# ===== 통합 회원가입 API =====

@app.route('/api/signup', methods=['POST'])
def full_signup():
    """통합 회원가입 (보호자 + 반려동물)"""
    try:
        data = request.get_json()
        
        # 보호자 데이터 추출
        guardian_data = {
            'name': data.get('guardian_name', ''),
            'email': data.get('email', ''),
            'phone': data.get('phone', ''),
            'experience_level': data.get('experience_level', '초급')
        }
        
        # 반려동물 데이터 추출
        pet_data = {
            'name': data.get('pet_name', ''),
            'breed': data.get('breed', ''),
            'gender': data.get('pet_gender', ''),
            'birth_date': data.get('birth_date', ''),
            'weight': data.get('weight', ''),
            'photo_base64': data.get('pet_photo', ''),
            'photo_content_type': data.get('photo_content_type', 'image/jpeg')
        }
        
        # 보호자 생성
        guardian_id = Guardian.create(guardian_data)
        
        # 반려동물 생성
        pet_id = Pet.create(guardian_id, pet_data)
        
        return success_response(
            "회원가입이 성공적으로 완료되었습니다.",
            {
                "guardian_id": guardian_id,
                "pet_id": pet_id,
                "guardian_name": guardian_data['name'],
                "pet_name": pet_data['name'],
                "email": guardian_data['email']
            },
            201
        )
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        print(f"통합 회원가입 오류: {e}")
        return error_response("회원가입 처리 중 오류가 발생했습니다.", 500)

# ===== 헬스체크 및 오류 핸들러 =====
@app.errorhandler(404)
def not_found(error):
    return error_response("요청한 API 엔드포인트를 찾을 수 없습니다.", 404)

@app.errorhandler(405)
def method_not_allowed(error):
    return error_response("허용되지 않는 HTTP 메서드입니다.", 405)

@app.errorhandler(500)
def internal_error(error):
    return error_response("서버 내부 오류가 발생했습니다.", 500)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)