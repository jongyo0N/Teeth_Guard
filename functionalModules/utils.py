# ===== utils.py =====
import base64
from typing import Optional

class ImageUtils:
    """이미지 처리 유틸리티"""
    
    @staticmethod
    def convert_base64_to_blob(base64_string: str) -> Optional[bytes]:
        """Base64 문자열을 BLOB으로 변환"""
        if not base64_string:
            return None
        
        try:
            # data:image/jpeg;base64, 부분 제거
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Base64 디코딩
            image_data = base64.b64decode(base64_string)
            return image_data
        except Exception as e:
            print(f"Base64 변환 오류: {e}")
            return None

    @staticmethod
    def blob_to_base64(blob_data: bytes, content_type: str = 'image/jpeg') -> str:
        """BLOB을 Base64 문자열로 변환"""
        if not blob_data:
            return None
        
        try:
            base64_string = base64.b64encode(blob_data).decode('utf-8')
            return f"data:{content_type};base64,{base64_string}"
        except Exception as e:
            print(f"BLOB to Base64 변환 오류: {e}")
            return None
