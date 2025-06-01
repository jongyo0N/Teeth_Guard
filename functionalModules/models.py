# ===== models.py =====
from functionalModules.database import DatabaseManager
from functionalModules.validators import Validator, ValidationError
from functionalModules.utils import ImageUtils
from typing import Dict, List, Any, Optional
from mysql.connector import Error
import datetime
import hashlib

class Guardian:
    """보호자 모델 클래스"""
    
    @staticmethod
    def create(guardian_data: Dict[str, Any]) -> int:
        """새 보호자 생성"""
        # 데이터 유효성 검증
        validated_data = Validator.validate_guardian_data(guardian_data)
        
        # 비밀번호 해시 생성 (필수)
        if 'password' not in validated_data or not validated_data['password']:
            raise ValidationError("비밀번호가 필요합니다.")
        password_hash = hashlib.sha256(validated_data['password'].encode('utf-8')).hexdigest()
        
        with DatabaseManager.get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            # 이메일 중복 확인
            cursor.execute("SELECT guardian_id FROM guardians WHERE email = %s", 
                         (validated_data['email'],))
            if cursor.fetchone():
                raise ValidationError("이미 등록된 이메일입니다.")
            
            # 보호자 생성
            insert_query = """
                INSERT INTO guardians (name, phone, email, password_hash, experience_level) 
                VALUES (%s, %s, %s, %s, %s)
            """
            values = (
                validated_data['name'],
                validated_data['phone'],
                validated_data['email'],
                password_hash,
                validated_data['experience_level']
            )
            
            cursor.execute(insert_query, values)
            guardian_id = cursor.lastrowid
            conn.commit()
            
            return guardian_id

    @staticmethod
    def get_by_id(guardian_id: int) -> Optional[Dict[str, Any]]:
        """ID로 보호자 조회"""
        with DatabaseManager.get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            query = """
                SELECT guardian_id, name, phone, email, experience_level, 
                       registration_date, updated_at
                FROM guardians 
                WHERE guardian_id = %s
            """
            
            cursor.execute(query, (guardian_id,))
            result = cursor.fetchone()
            
            if result:
                # datetime 객체를 문자열로 변환
                if result['registration_date']:
                    result['registration_date'] = result['registration_date'].isoformat()
                if result['updated_at']:
                    result['updated_at'] = result['updated_at'].isoformat()
            
            return result

    @staticmethod
    def get_by_email(email: str) -> Optional[Dict[str, Any]]:
        """이메일로 보호자 조회"""
        with DatabaseManager.get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            query = "SELECT * FROM guardians WHERE email = %s"
            cursor.execute(query, (email.lower(),))
            return cursor.fetchone()

    @staticmethod
    def update(guardian_id: int, update_data: Dict[str, Any]) -> bool:
        """보호자 정보 수정"""
        # 업데이트할 데이터 유효성 검증
        validated_data = Validator.validate_guardian_data(update_data)
        
        with DatabaseManager.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 보호자 존재 확인
            cursor.execute("SELECT guardian_id FROM guardians WHERE guardian_id = %s", 
                         (guardian_id,))
            if not cursor.fetchone():
                raise ValidationError("존재하지 않는 보호자입니다.")
            
            # 이메일 중복 확인 (자신 제외)
            cursor.execute(
                "SELECT guardian_id FROM guardians WHERE email = %s AND guardian_id != %s", 
                (validated_data['email'], guardian_id)
            )
            if cursor.fetchone():
                raise ValidationError("이미 사용 중인 이메일입니다.")
            
            # 업데이트 실행
            update_query = """
                UPDATE guardians 
                SET name = %s, phone = %s, email = %s, experience_level = %s, updated_at = NOW()
                WHERE guardian_id = %s
            """
            values = (
                validated_data['name'],
                validated_data['phone'],
                validated_data['email'],
                validated_data['experience_level'],
                guardian_id
            )
            
            cursor.execute(update_query, values)
            conn.commit()
            
            return cursor.rowcount > 0

    @staticmethod
    def delete(guardian_id: int) -> bool:
        """보호자 삭제 (CASCADE로 반려동물도 함께 삭제됨)"""
        with DatabaseManager.get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM guardians WHERE guardian_id = %s", (guardian_id,))
            conn.commit()
            
            return cursor.rowcount > 0

    @staticmethod
    def check_email_availability(email: str) -> bool:
        """이메일 사용 가능 여부 확인"""
        if not Validator.validate_email(email):
            return False
        
        with DatabaseManager.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM guardians WHERE email = %s", (email.lower(),))
            count = cursor.fetchone()[0]
            return count == 0


class Pet:
    """반려동물 모델 클래스"""
    
    @staticmethod
    def get_next_pet_id(guardian_id: int) -> int:
        """해당 보호자의 다음 pet_id 생성"""
        with DatabaseManager.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COALESCE(MAX(pet_id), 0) + 1 as next_id FROM pets WHERE guardian_id = %s", 
                (guardian_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else 1

    @staticmethod
    def create(guardian_id: int, pet_data: Dict[str, Any]) -> int:
        """새 반려동물 생성"""
        # 데이터 유효성 검증
        validated_data = Validator.validate_pet_data(pet_data)
        
        with DatabaseManager.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 보호자 존재 확인
            cursor.execute("SELECT guardian_id FROM guardians WHERE guardian_id = %s", 
                         (guardian_id,))
            if not cursor.fetchone():
                raise ValidationError("존재하지 않는 보호자입니다.")
            
            # 새 pet_id 생성
            pet_id = Pet.get_next_pet_id(guardian_id)
            
            # 이미지 처리
            photo_blob = None
            if validated_data['photo_base64']:
                photo_blob = ImageUtils.convert_base64_to_blob(validated_data['photo_base64'])
                if photo_blob is None:
                    raise ValidationError("이미지 파일 처리 중 오류가 발생했습니다.")
            
            # 반려동물 생성
            insert_query = """
                INSERT INTO pets (guardian_id, pet_id, name, breed, gender, birth_date, weight, photo, photo_content_type) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                guardian_id,
                pet_id,
                validated_data['name'],
                validated_data['breed'],
                validated_data['gender'],
                validated_data['birth_date'],
                validated_data['weight'],
                photo_blob,
                validated_data['photo_content_type'] if photo_blob else None
            )
            
            cursor.execute(insert_query, values)
            conn.commit()
            
            return pet_id

    @staticmethod
    def get_by_id(guardian_id: int, pet_id: int, include_photo: bool = False) -> Optional[Dict[str, Any]]:
        """ID로 반려동물 조회"""
        with DatabaseManager.get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            if include_photo:
                query = """
                    SELECT guardian_id, pet_id, name, breed, gender, birth_date, weight, 
                           photo, photo_content_type, registration_date, updated_at
                    FROM pets 
                    WHERE guardian_id = %s AND pet_id = %s
                """
            else:
                query = """
                    SELECT guardian_id, pet_id, name, breed, gender, birth_date, weight, 
                           photo_content_type, registration_date, updated_at
                    FROM pets 
                    WHERE guardian_id = %s AND pet_id = %s
                """
            
            cursor.execute(query, (guardian_id, pet_id))
            result = cursor.fetchone()
            
            if result:
                # datetime 객체를 문자열로 변환
                if result['birth_date']:
                    result['birth_date'] = result['birth_date'].isoformat()
                if result['registration_date']:
                    result['registration_date'] = result['registration_date'].isoformat()
                if result['updated_at']:
                    result['updated_at'] = result['updated_at'].isoformat()
                
                # 나이 계산
                if result['birth_date']:
                    from datetime import date
                    birth_date = datetime.fromisoformat(result['birth_date']).date()
                    today = date.today()
                    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                    result['age_years'] = age
                
                # 이미지 처리
                if include_photo and result.get('photo') and result.get('photo_content_type'):
                    result['photo_base64'] = ImageUtils.blob_to_base64(
                        result['photo'], result['photo_content_type']
                    )
                
                result['has_photo'] = bool(result.get('photo_content_type'))
                if 'photo' in result:
                    del result['photo']  # BLOB 데이터는 응답에서 제거
            
            return result

    @staticmethod
    def get_all_by_guardian(guardian_id: int, include_photos: bool = False) -> List[Dict[str, Any]]:
        """보호자의 모든 반려동물 조회"""
        with DatabaseManager.get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            if include_photos:
                query = """
                    SELECT guardian_id, pet_id, name, breed, gender, birth_date, weight,
                           photo, photo_content_type, registration_date, updated_at
                    FROM pets 
                    WHERE guardian_id = %s
                    ORDER BY pet_id
                """
            else:
                query = """
                    SELECT guardian_id, pet_id, name, breed, gender, birth_date, weight,
                           photo_content_type, registration_date, updated_at
                    FROM pets 
                    WHERE guardian_id = %s
                    ORDER BY pet_id
                """
            
            cursor.execute(query, (guardian_id,))
            results = cursor.fetchall()
            
            pets = []
            for result in results:
                # datetime 객체를 문자열로 변환
                if result['birth_date']:
                    result['birth_date'] = result['birth_date'].isoformat()
                if result['registration_date']:
                    result['registration_date'] = result['registration_date'].isoformat()
                if result['updated_at']:
                    result['updated_at'] = result['updated_at'].isoformat()
                
                # 나이 계산
                if result['birth_date']:
                    from datetime import date
                    birth_date = datetime.fromisoformat(result['birth_date']).date()
                    today = date.today()
                    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                    result['age_years'] = age
                
                # 이미지 처리
                if include_photos and result.get('photo') and result.get('photo_content_type'):
                    result['photo_base64'] = ImageUtils.blob_to_base64(
                        result['photo'], result['photo_content_type']
                    )
                
                result['has_photo'] = bool(result.get('photo_content_type'))
                if 'photo' in result:
                    del result['photo']  # BLOB 데이터는 응답에서 제거
                
                pets.append(result)
            
            return pets

    @staticmethod
    def update(guardian_id: int, pet_id: int, update_data: Dict[str, Any]) -> bool:
        """반려동물 정보 수정"""
        # 업데이트할 데이터 유효성 검증
        validated_data = Validator.validate_pet_data(update_data)
        
        with DatabaseManager.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 반려동물 존재 확인
            cursor.execute(
                "SELECT pet_id FROM pets WHERE guardian_id = %s AND pet_id = %s", 
                (guardian_id, pet_id)
            )
            if not cursor.fetchone():
                raise ValidationError("존재하지 않는 반려동물입니다.")
            
            # 이미지 처리
            photo_blob = None
            photo_content_type = None
            
            if validated_data['photo_base64']:
                photo_blob = ImageUtils.convert_base64_to_blob(validated_data['photo_base64'])
                if photo_blob is None:
                    raise ValidationError("이미지 파일 처리 중 오류가 발생했습니다.")
                photo_content_type = validated_data['photo_content_type']
            
            # 업데이트 실행
            if photo_blob:
                # 이미지도 함께 업데이트
                update_query = """
                    UPDATE pets 
                    SET name = %s, breed = %s, gender = %s, birth_date = %s, weight = %s, 
                        photo = %s, photo_content_type = %s, updated_at = NOW()
                    WHERE guardian_id = %s AND pet_id = %s
                """
                values = (
                    validated_data['name'],
                    validated_data['breed'],
                    validated_data['gender'],
                    validated_data['birth_date'],
                    validated_data['weight'],
                    photo_blob,
                    photo_content_type,
                    guardian_id,
                    pet_id
                )
            else:
                # 이미지 제외하고 업데이트
                update_query = """
                    UPDATE pets 
                    SET name = %s, breed = %s, gender = %s, birth_date = %s, weight = %s, updated_at = NOW()
                    WHERE guardian_id = %s AND pet_id = %s
                """
                values = (
                    validated_data['name'],
                    validated_data['breed'],
                    validated_data['gender'],
                    validated_data['birth_date'],
                    validated_data['weight'],
                    guardian_id,
                    pet_id
                )
            
            cursor.execute(update_query, values)
            conn.commit()
            
            return cursor.rowcount > 0

    @staticmethod
    def delete(guardian_id: int, pet_id: int) -> bool:
        """반려동물 삭제"""
        with DatabaseManager.get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "DELETE FROM pets WHERE guardian_id = %s AND pet_id = %s", 
                (guardian_id, pet_id)
            )
            conn.commit()
            
            return cursor.rowcount > 0
