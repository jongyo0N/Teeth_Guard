# ===== validators.py =====
import re
from datetime import datetime, date
from typing import List, Dict, Any

class ValidationError(Exception):
    """유효성 검사 오류 클래스"""
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(self.message)

class Validator:
    """유효성 검사 클래스"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """이메일 유효성 검증"""
        if not email:
            return False
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email.strip()) is not None

    @staticmethod
    def validate_phone(phone: str) -> bool:
        """전화번호 유효성 검증 (한국 형식)"""
        if not phone:
            return True  # 선택사항이므로 빈 값 허용
        # 하이픈 제거 후 검증
        phone_clean = phone.replace('-', '').replace(' ', '')
        pattern = r'^01[0-9]{8,9}$'
        return re.match(pattern, phone_clean) is not None

    @staticmethod
    def validate_image_format(content_type: str) -> bool:
        """이미지 파일 형식 검증"""
        if not content_type:
            return True  # 선택사항
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif']
        return content_type.lower() in allowed_types

    @staticmethod
    def validate_weight(weight) -> float:
        """체중 유효성 검증"""
        if weight is None or weight == '':
            return None
        try:
            weight_float = float(weight)
            if weight_float <= 0 or weight_float > 100:
                raise ValidationError("체중은 0보다 크고 100kg 이하여야 합니다.", "weight")
            return weight_float
        except (ValueError, TypeError):
            raise ValidationError("체중은 유효한 숫자여야 합니다.", "weight")

    @staticmethod
    def validate_birth_date(birth_date_str: str) -> date:
        """생년월일 유효성 검증"""
        if not birth_date_str:
            return None
        try:
            birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d').date()
            if birth_date > date.today():
                raise ValidationError("생년월일은 오늘 날짜보다 이전이어야 합니다.", "birth_date")
            return birth_date
        except ValueError:
            raise ValidationError("생년월일은 YYYY-MM-DD 형식이어야 합니다.", "birth_date")

    @staticmethod
    def validate_guardian_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """보호자 데이터 종합 유효성 검증"""
        errors = []
        
        # 필수 필드 검증
        required_fields = ['name', 'email']
        for field in required_fields:
            if not data.get(field, '').strip():
                errors.append(f"{field}는 필수 입력 항목입니다.")
        
        # 이메일 검증
        if data.get('email') and not Validator.validate_email(data['email']):
            errors.append("올바른 이메일 형식이 아닙니다.")
        
        # 전화번호 검증
        if data.get('phone') and not Validator.validate_phone(data['phone']):
            errors.append("올바른 전화번호 형식이 아닙니다.")
        
        # 경험도 검증
        if data.get('experience_level') and data['experience_level'] not in ['초급', '중급', '고급']:
            errors.append("경험도는 초급, 중급, 고급 중 하나여야 합니다.")
        
        if errors:
            raise ValidationError("입력 데이터에 오류가 있습니다: " + ", ".join(errors))
        
        return {
            'name': data['name'].strip(),
            'email': data['email'].strip().lower(),
            'phone': data.get('phone', '').strip() or None,
            'experience_level': data.get('experience_level', '초급')
        }

    @staticmethod
    def validate_pet_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """반려동물 데이터 종합 유효성 검증"""
        errors = []
        
        # 필수 필드 검증
        required_fields = ['name', 'gender']
        for field in required_fields:
            if not data.get(field, '').strip():
                errors.append(f"반려동물 {field}는 필수 입력 항목입니다.")
        
        # 성별 검증
        if data.get('gender') and data['gender'] not in ['수컷', '암컷']:
            errors.append("반려동물 성별은 수컷 또는 암컷이어야 합니다.")
        
        if errors:
            raise ValidationError("입력 데이터에 오류가 있습니다: " + ", ".join(errors))
        
        # 체중 검증
        weight = None
        if data.get('weight'):
            weight = Validator.validate_weight(data['weight'])
        
        # 생년월일 검증
        birth_date = None
        if data.get('birth_date'):
            birth_date = Validator.validate_birth_date(data['birth_date'])
        
        # 이미지 검증
        if data.get('photo_content_type') and not Validator.validate_image_format(data['photo_content_type']):
            raise ValidationError("지원하지 않는 이미지 형식입니다. (JPEG, PNG, GIF만 지원)")
        
        return {
            'name': data['name'].strip(),
            'breed': data.get('breed', '').strip() or None,
            'gender': data['gender'],
            'birth_date': birth_date,
            'weight': weight,
            'photo_base64': data.get('photo_base64'),
            'photo_content_type': data.get('photo_content_type')
        }
