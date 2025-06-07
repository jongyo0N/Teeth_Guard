"""
치아 진단 API Blueprint
- 이미지 업로드 및 진단 처리
- diagnosis.py의 DentalDiagnosisSupervisor 활용
- 진단 결과 및 피드백 반환
"""
import os
import tempfile
import asyncio
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from typing import Dict, Any

# 진단 모듈 import
from diagnosis import DentalDiagnosisSupervisor
from functionalModules.database import DatabaseManager
from functionalModules.utils import ImageUtils

# Blueprint 생성
diagnosis_bp = Blueprint('diagnosis', __name__)

# CORS 설정
CORS(diagnosis_bp)

# 허용된 이미지 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 최대 파일 크기 (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def success_response(message: str, data: Any = None, status_code: int = 200):
    """성공 응답 헬퍼"""
    response = {'success': True, 'message': message}
    if data is not None:
        response['data'] = data
    return jsonify(response), status_code

def error_response(message: str, status_code: int = 400, errors: Any = None):
    """오류 응답 헬퍼"""
    response = {'success': False, 'message': message}
    if errors:
        response['errors'] = errors
    return jsonify(response), status_code

def validate_guardian_and_pet(guardian_id: int, pet_id: int) -> bool:
    """보호자와 반려동물 존재 여부 확인"""
    try:
        db_manager = DatabaseManager()
        with db_manager.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 보호자 및 반려동물 존재 확인
            query = """
            SELECT COUNT(*) FROM pets p
            JOIN guardians g ON p.guardian_id = g.guardian_id
            WHERE p.guardian_id = %s AND p.pet_id = %s
            """
            cursor.execute(query, (guardian_id, pet_id))
            count = cursor.fetchone()[0]
            
            return count > 0
    except Exception as e:
        print(f"데이터 검증 오류: {e}")
        return False

def save_diagnosis_to_db(guardian_id: int, pet_id: int, diagnosis_result: Dict[str, Any]) -> int:
    """진단 결과를 데이터베이스에 저장"""
    try:
        db_manager = DatabaseManager()
        with db_manager.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 진단 결과 데이터 추출
            diagnosis_data = diagnosis_result['diagnosis_results']
            feedback = diagnosis_result['feedback']
            pet_info = diagnosis_result['pet_info']
            
            # dental_analysis 테이블에 저장
            insert_query = """
            INSERT INTO dental_analysis (
                guardian_id, pet_id, analysis_date,
                caries_percentage, calculus_percentage, periodontal_level,
                total_score, recommend_guide
            ) VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s)
            """
            
            # 피드백 텍스트 결합
            combined_feedback = "\n".join([f"[{key}] {value}" for key, value in feedback.items()])
            
            values = (
                guardian_id,
                pet_id,
                diagnosis_data.get('caries_ratio', 0),
                diagnosis_data.get('calculus_ratio', 0),
                diagnosis_result.get('periodontitis_stage', 0),
                diagnosis_result.get('total_score', 0),
                combined_feedback
            )
            
            cursor.execute(insert_query, values)
            analysis_id = cursor.lastrowid
            conn.commit()
            
            return analysis_id
            
    except Exception as e:
        print(f"데이터베이스 저장 오류: {e}")
        return None

async def run_diagnosis_async(supervisor, guardian_id: int, pet_id: int, image_path: str):
    """비동기 진단 실행"""
    try:
        result = await supervisor.diagnose(guardian_id, pet_id, image_path)
        return result
    except Exception as e:
        print(f"진단 실행 오류: {e}")
        raise e

# 진단 결과 임시 저장용 (메모리/세션 등)
diagnosis_temp_store = {}

@diagnosis_bp.route('/api/diagnosis', methods=['POST'])
def upload_and_diagnose():
    """이미지 업로드 및 치아 진단 수행"""
    temp_file_path = None

    try:
        # 1. 요청 데이터 검증
        if 'image' not in request.files:
            return error_response("이미지 파일이 없습니다.", 400)

        file = request.files['image']
        if file.filename == '':
            return error_response("파일이 선택되지 않았습니다.", 400)

        guardian_id = request.form.get('guardian_id')
        pet_id = request.form.get('pet_id')

        if not guardian_id or not pet_id:
            return error_response("guardian_id와 pet_id가 필요합니다.", 400)

        try:
            guardian_id = int(guardian_id)
            pet_id = int(pet_id)
        except ValueError:
            return error_response("guardian_id와 pet_id는 숫자여야 합니다.", 400)

        # 2. 파일 유효성 검증
        if not allowed_file(file.filename):
            return error_response("지원하지 않는 파일 형식입니다. (PNG, JPG, JPEG, GIF만 허용)", 400)

        # 파일 크기 검증
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size > MAX_FILE_SIZE:
            return error_response("파일 크기는 10MB 이하여야 합니다.", 400)

        # 3. 보호자 및 반려동물 존재 확인
        if not validate_guardian_and_pet(guardian_id, pet_id):
            return error_response("존재하지 않는 보호자 또는 반려동물입니다.", 404)

        # 4. 임시 파일로 저장
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"diagnosis_{guardian_id}_{pet_id}_{timestamp}_{filename}"

        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, temp_filename)

        file.save(temp_file_path)
        print(f"임시 파일 저장: {temp_file_path}")

        # 5. 진단 수행
        supervisor = DentalDiagnosisSupervisor()

        # asyncio를 사용하여 비동기 진단 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            diagnosis_result = loop.run_until_complete(
                run_diagnosis_async(supervisor, guardian_id, pet_id, temp_file_path)
            )
        finally:
            loop.close()

        # 6. 결과 검증
        if not diagnosis_result or 'diagnosis_results' not in diagnosis_result:
            return error_response("진단 처리 중 오류가 발생했습니다.", 500)

        # 8. 응답 데이터 구성
        diagnosis_data = diagnosis_result['diagnosis_results']
        pet_info = diagnosis_result['pet_info']
        feedback = diagnosis_result['feedback']

        response_data = {
            "guardian_id": guardian_id,
            "pet_id": pet_id,
            "analysis_date": datetime.now().isoformat(),
            "pet_info": {
                "name": pet_info.get('name', ''),
                "breed": pet_info.get('breed', ''),
                "age_years": pet_info.get('age_years', 0),
                "weight": pet_info.get('weight', 0),
                "size_category": pet_info.get('size_category', ''),
                "gender": pet_info.get('gender', '')
            },
            "diagnosis_results": {
                "caries_percentage": round(diagnosis_data.get('caries_ratio', 0), 2),
                "calculus_percentage": round(diagnosis_data.get('calculus_ratio', 0), 2),
                "periodontitis_stage": diagnosis_result.get('periodontitis_stage', 0),
                "total_score": round(diagnosis_result.get('total_score', 0), 2),
                "total_tooth_area": diagnosis_data.get('total_tooth_area', 0),
                "caries_area": diagnosis_data.get('caries_area', 0),
                "calculus_area": diagnosis_data.get('calculus_area', 0)
            },
            "feedback": {
                "caries": feedback.get('충치', ''),
                "calculus": feedback.get('치석', ''),
                "periodontitis": feedback.get('치주염', '')
            },
        }

        # 진단 결과를 임시 저장 (guardian_id, pet_id 기준)
        key = f"{guardian_id}_{pet_id}"
        diagnosis_temp_store[key] = response_data

        return success_response(
            "치아 진단이 성공적으로 완료되었습니다.",
            None,  # 데이터는 반환하지 않음
            200
        )

    except Exception as e:
        print(f"진단 API 오류: {e}")
        import traceback
        traceback.print_exc()
        return error_response("진단 처리 중 예상치 못한 오류가 발생했습니다.", 500)

    finally:
        # 임시 파일 정리
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"임시 파일 삭제: {temp_file_path}")
            except Exception as e:
                print(f"임시 파일 삭제 실패: {e}")

@diagnosis_bp.route('/api/diagnosis/data', methods=['GET'])
def get_diagnosis_data():
    """
    진단 결과 데이터 반환 (guardian_id, pet_id 쿼리 파라미터 필요)
    예: /api/diagnosis/data?guardian_id=1&pet_id=2
    """
    guardian_id = request.args.get('guardian_id')
    pet_id = request.args.get('pet_id')
    if not guardian_id or not pet_id:
        return error_response("guardian_id와 pet_id가 필요합니다.", 400)
    key = f"{guardian_id}_{pet_id}"
    data = diagnosis_temp_store.get(key)
    if not data:
        return error_response("진단 결과가 존재하지 않습니다.", 404)
    return success_response("진단 결과 조회 성공", data, 200)

# 오류 핸들러
@diagnosis_bp.errorhandler(413)
def too_large(e):
    return error_response("파일 크기가 너무 큽니다.", 413)

@diagnosis_bp.errorhandler(500)
def internal_error(e):
    return error_response("서버 내부 오류가 발생했습니다.", 500)