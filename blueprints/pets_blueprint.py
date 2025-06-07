"""
Pets Blueprint - 반려동물 관리 API
프론트엔드 요구사항에 맞춘 데이터 변환 및 API 제공

이 파일이 실제로 사용되는 Blueprint입니다!
app.py에서 이 Blueprint를 import해서 사용합니다.
"""
from flask import Blueprint, request, jsonify
from functionalModules.models import Guardian, Pet
from functionalModules.validators import ValidationError
from functionalModules.database import DatabaseManager
from typing import Any, List, Dict, Optional
from datetime import datetime

# Blueprint 생성
pets_bp = Blueprint('pets', __name__, url_prefix='/api/pets')

# 공통 응답 헬퍼 함수
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

class PetDataTransformer:
    """프론트엔드와 백엔드 간의 pet 데이터 변환 클래스"""
    
    @staticmethod
    def frontend_to_backend(frontend_pet: Dict[str, Any], guardian_id: int) -> Dict[str, Any]:
        """프론트엔드 pet 데이터를 백엔드 형식으로 변환"""
        # 생년월일 조합
        birth_date = None
        if (frontend_pet.get('birthYear') and 
            frontend_pet.get('birthMonth') and 
            frontend_pet.get('birthDay')):
            try:
                year = frontend_pet['birthYear']
                month = frontend_pet['birthMonth'].zfill(2)
                day = frontend_pet['birthDay'].zfill(2)
                birth_date = f"{year}-{month}-{day}"
            except Exception as e:
                print(f"생년월일 변환 오류: {e}")
        
        # 성별 변환
        gender_map = {'male': '수컷', 'female': '암컷'}
        gender = gender_map.get(frontend_pet.get('gender'), '수컷')
        
        # 품종 변환
        breed = frontend_pet.get('breed', '')
        if breed == 'large':
            breed = '대형견'
        elif breed == 'small':
            breed = '소형견'
        
        return {
            'name': frontend_pet.get('name', ''),
            'breed': breed,
            'gender': gender,
            'birth_date': birth_date,
            'weight': None,
            'photo_base64': frontend_pet.get('photo'),
            'photo_content_type': 'image/jpeg' if frontend_pet.get('photo') else None
        }
    
    @staticmethod
    def backend_to_frontend(backend_pet: Dict[str, Any]) -> Dict[str, Any]:
        """백엔드 pet 데이터를 프론트엔드 형식으로 변환"""
        # 생년월일 분리
        birth_year = birth_month = birth_day = ''
        if backend_pet.get('birth_date'):
            try:
                if isinstance(backend_pet['birth_date'], str):
                    birth_date = datetime.fromisoformat(backend_pet['birth_date']).date()
                else:
                    birth_date = backend_pet['birth_date']
                
                birth_year = str(birth_date.year)
                birth_month = str(birth_date.month)
                birth_day = str(birth_date.day)
            except Exception as e:
                print(f"생년월일 분리 오류: {e}")
        
        # 성별 변환
        gender_map = {'수컷': 'male', '암컷': 'female'}
        gender = gender_map.get(backend_pet.get('gender'), 'male')
        
        # 품종 변환
        breed = 'small'  # 기본값
        breed_str = backend_pet.get('breed', '').lower()
        if '대형' in breed_str or 'large' in breed_str:
            breed = 'large'
        
        return {
            'id': str(backend_pet.get('pet_id', '')),
            'registrationNumber': str(backend_pet.get('pet_id', '')),
            'name': backend_pet.get('name', ''),
            'breed': breed,
            'gender': gender,
            'birthYear': birth_year,
            'birthMonth': birth_month,
            'birthDay': birth_day,
            'photo': backend_pet.get('photo_base64')
        }

class PetManager:
    """Pet 관리 확장 클래스"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def batch_create_pets(self, guardian_id: int, pets_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """여러 반려동물 일괄 등록"""
        if not pets_data:
            return {"created_pets": [], "failed_pets": [], "success_count": 0, "total_count": 0}
        
        # 보호자 존재 확인
        guardian = Guardian.get_by_id(guardian_id)
        if not guardian:
            raise ValidationError("존재하지 않는 보호자입니다.")
        
        created_pets = []
        failed_pets = []
        
        for i, pet_data in enumerate(pets_data):
            try:
                backend_data = PetDataTransformer.frontend_to_backend(pet_data, guardian_id)
                pet_id = Pet.create(guardian_id, backend_data)
                
                created_pet = Pet.get_by_id(guardian_id, pet_id, include_photo=True)
                if created_pet:
                    frontend_pet = PetDataTransformer.backend_to_frontend(created_pet)
                    created_pets.append(frontend_pet)
                    print(f"✅ 펫 생성 성공: {pet_data.get('name', 'Unknown')} (ID: {pet_id})")
                
            except ValidationError as e:
                error_msg = f"펫 생성 실패: {e}"
                print(f"❌ {error_msg}")
                failed_pets.append({
                    "index": i, 
                    "name": pet_data.get('name', 'Unknown'), 
                    "error": str(e)
                })
            except Exception as e:
                error_msg = f"펫 생성 중 오류: {e}"
                print(f"❌ {error_msg}")
                failed_pets.append({
                    "index": i, 
                    "name": pet_data.get('name', 'Unknown'), 
                    "error": str(e)
                })
        
        return {
            "created_pets": created_pets,
            "failed_pets": failed_pets,
            "success_count": len(created_pets),
            "total_count": len(pets_data)
        }
    
    def update_pet_by_pet_id(self, pet_id: int, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """pet_id만으로 반려동물 정보 수정"""
        with self.db_manager.get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            # pet_id로 guardian_id 조회
            cursor.execute("SELECT guardian_id FROM pets WHERE pet_id = %s", (pet_id,))
            result = cursor.fetchone()
            if not result:
                raise ValidationError("존재하지 않는 반려동물입니다.")
            
            guardian_id = result['guardian_id']
            
            # 데이터 변환 및 수정
            backend_data = PetDataTransformer.frontend_to_backend(update_data, guardian_id)
            success = Pet.update(guardian_id, pet_id, backend_data)
            
            if success:
                updated_pet = Pet.get_by_id(guardian_id, pet_id, include_photo=True)
                if updated_pet:
                    return PetDataTransformer.backend_to_frontend(updated_pet)
            
            return None
    
    def get_pets_by_guardian(self, guardian_id: int, include_photos: bool = False) -> List[Dict[str, Any]]:
        """보호자의 모든 반려동물 조회 (프론트엔드 형식으로 변환)"""
        backend_pets = Pet.get_all_by_guardian(guardian_id, include_photos)
        
        frontend_pets = []
        for backend_pet in backend_pets:
            frontend_pet = PetDataTransformer.backend_to_frontend(backend_pet)
            frontend_pets.append(frontend_pet)
        
        return frontend_pets

# 펫 매니저 인스턴스
pet_manager = PetManager()

# ===== API 엔드포인트 =====

@pets_bp.route('/<int:pet_id>', methods=['PUT'])
def update_pet_by_id(pet_id):
    """반려동물 정보 수정 (doginfo 페이지용)"""
    try:
        data = request.get_json()
        print(f"📨 펫 수정 요청: pet_id={pet_id}, data={data}")
        
        if not data:
            return error_response("요청 데이터가 없습니다.", 400)
        
        updated_pet = pet_manager.update_pet_by_pet_id(pet_id, data)
        
        if updated_pet:
            return success_response("반려동물 정보가 성공적으로 수정되었습니다.", updated_pet)
        else:
            return error_response("수정할 정보가 없습니다.", 400)
            
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        print(f"❌ 반려동물 수정 오류: {e}")
        return error_response("수정 처리 중 오류가 발생했습니다.", 500)

@pets_bp.route('/guardian/<int:guardian_id>/batch', methods=['POST'])
def batch_create_pets(guardian_id):
    """여러 반려동물 일괄 등록 (addpet 페이지용)"""
    try:
        data = request.get_json()
        pets_data = data.get('pets', [])
        pet_count = data.get('petCount', len(pets_data))
        
        print(f"📨 일괄 등록 요청: guardian_id={guardian_id}, 펫 수={pet_count}")
        
        if not pets_data:
            return error_response("등록할 반려동물 정보가 없습니다.", 400)
        
        result = pet_manager.batch_create_pets(guardian_id, pets_data)
        created_pets = result["created_pets"]
        failed_pets = result["failed_pets"]
        
        if len(failed_pets) > 0:
            return success_response(
                f"{len(created_pets)}마리 등록 성공, {len(failed_pets)}마리 실패",
                {
                    "pets": created_pets,
                    "created_count": len(created_pets),
                    "failed_count": len(failed_pets),
                    "failed_pets": failed_pets,
                    "requested_count": pet_count
                },
                201
            )
        else:
            return success_response(
                f"{len(created_pets)}마리의 반려동물이 성공적으로 등록되었습니다.",
                {
                    "pets": created_pets,
                    "created_count": len(created_pets),
                    "requested_count": pet_count
                },
                201
            )
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        print(f"❌ 일괄 등록 오류: {e}")
        return error_response("일괄 등록 처리 중 오류가 발생했습니다.", 500)

@pets_bp.route('/guardian/<int:guardian_id>', methods=['GET'])
def get_guardian_pets_frontend(guardian_id):
    """보호자의 모든 반려동물 조회 (doginfo 페이지용)"""
    try:
        include_photos = request.args.get('include_photos', 'false').lower() == 'true'
        
        print(f"📨 펫 목록 조회: guardian_id={guardian_id}, include_photos={include_photos}")
        
        # 보호자 존재 확인
        guardian = Guardian.get_by_id(guardian_id)
        if not guardian:
            return error_response("존재하지 않는 보호자입니다.", 404)
        
        pets = pet_manager.get_pets_by_guardian(guardian_id, include_photos)
        
        print(f"✅ 펫 목록 조회 성공: {len(pets)}마리")
        
        return success_response("조회 성공", pets)
        
    except Exception as e:
        print(f"❌ 반려동물 목록 조회 오류: {e}")
        return error_response("조회 중 오류가 발생했습니다.", 500)

@pets_bp.route('/<int:pet_id>', methods=['GET'])
def get_pet_by_id(pet_id):
    """반려동물 상세 조회"""
    try:
        include_photo = request.args.get('include_photo', 'false').lower() == 'true'
        
        # pet_id로 guardian_id 조회
        db_manager = DatabaseManager()
        with db_manager.get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT guardian_id FROM pets WHERE pet_id = %s", (pet_id,))
            result = cursor.fetchone()
            if not result:
                return error_response("존재하지 않는 반려동물입니다.", 404)
            guardian_id = result['guardian_id']
        
        # Pet 모델로 조회
        backend_pet = Pet.get_by_id(guardian_id, pet_id, include_photo)
        if not backend_pet:
            return error_response("존재하지 않는 반려동물입니다.", 404)
        
        # 프론트엔드 형식으로 변환
        frontend_pet = PetDataTransformer.backend_to_frontend(backend_pet)
        
        return success_response("조회 성공", frontend_pet)
        
    except Exception as e:
        print(f"❌ 반려동물 조회 오류: {e}")
        return error_response("조회 중 오류가 발생했습니다.", 500)

@pets_bp.route('/<int:pet_id>', methods=['DELETE'])
def delete_pet_by_id(pet_id):
    """반려동물 삭제"""
    try:
        # pet_id로 guardian_id 조회
        db_manager = DatabaseManager()
        with db_manager.get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT guardian_id FROM pets WHERE pet_id = %s", (pet_id,))
            result = cursor.fetchone()
            if not result:
                return error_response("존재하지 않는 반려동물입니다.", 404)
            guardian_id = result['guardian_id']
        
        # Pet 모델로 삭제
        success = Pet.delete(guardian_id, pet_id)
        
        if success:
            return success_response("반려동물 정보가 성공적으로 삭제되었습니다.")
        else:
            return error_response("존재하지 않는 반려동물입니다.", 404)
            
    except Exception as e:
        print(f"❌ 반려동물 삭제 오류: {e}")
        return error_response("삭제 처리 중 오류가 발생했습니다.", 500)
