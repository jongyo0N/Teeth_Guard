from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
from functionalModules.database import DatabaseManager
from typing import Any, List

# Blueprint 생성
user_bp = Blueprint('user', __name__, url_prefix='/api/user')

# 성공 응답 헬퍼
def success_response(message: str, data: Any = None, status_code: int = 200):
    response = {'success': True, 'message': message}
    if data is not None:
        response['data'] = data
    return jsonify(response), status_code

# 오류 응답 헬퍼
def error_response(message: str, status_code: int = 400):
    response = {'success': False, 'message': message}
    return jsonify(response), status_code

@user_bp.route('/<int:guardian_id>', methods=['GET'])
def get_user_info(guardian_id):
    """사용자 정보 조회"""
    try:
        db_manager = DatabaseManager()
        with db_manager.get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            query = """
                SELECT name, nickname, email, phone
                FROM guardians 
                WHERE guardian_id = %s
            """
            
            cursor.execute(query, (guardian_id,))
            result = cursor.fetchone()
            
            if not result:
                return error_response("존재하지 않는 사용자입니다.", 404)
            
            user_info = {
                'name': result['name'],
                'nickname': result['nickname'] if result['nickname'] else result['name'],
                'email': result['email'],
                'phone': result['phone']
            }
            
            return success_response("사용자 정보 조회 성공", user_info)
            
    except Exception as e:
        print(f"사용자 정보 조회 오류: {e}")
        return error_response("조회 중 오류가 발생했습니다.", 500)

@user_bp.route('/<int:guardian_id>', methods=['PUT'])
def update_user_info(guardian_id):
    """사용자 정보 수정"""
    try:
        data = request.get_json()
        
        db_manager = DatabaseManager()
        with db_manager.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 사용자 존재 확인
            cursor.execute("SELECT guardian_id FROM guardians WHERE guardian_id = %s", (guardian_id,))
            if not cursor.fetchone():
                return error_response("존재하지 않는 사용자입니다.", 404)
            
            # 업데이트할 필드들
            update_fields = []
            values = []
            
            if 'name' in data:
                update_fields.append("name = %s")
                values.append(data['name'])
            
            if 'nickname' in data:
                update_fields.append("nickname = %s")
                values.append(data['nickname'])
            
            if 'email' in data:
                update_fields.append("email = %s")
                values.append(data['email'])
            
            if 'phone' in data:
                update_fields.append("phone = %s")
                values.append(data['phone'])
            
            if not update_fields:
                return error_response("수정할 정보가 없습니다.", 400)
            
            # 업데이트 실행
            update_fields.append("updated_at = NOW()")
            values.append(guardian_id)
            
            update_query = f"""
                UPDATE guardians 
                SET {', '.join(update_fields)}
                WHERE guardian_id = %s
            """
            
            cursor.execute(update_query, values)
            conn.commit()
            
            # 업데이트된 정보 조회
            cursor.execute("""
                SELECT name, nickname, email, phone 
                FROM guardians 
                WHERE guardian_id = %s
            """, (guardian_id,))
            
            updated_info = cursor.fetchone()
            
            user_info = {
                'name': updated_info[0],
                'nickname': updated_info[1] if updated_info[1] else updated_info[0],
                'email': updated_info[2],
                'phone': updated_info[3]
            }
            
            return success_response("사용자 정보가 성공적으로 수정되었습니다.", user_info)
            
    except Exception as e:
        print(f"사용자 정보 수정 오류: {e}")
        return error_response("수정 처리 중 오류가 발생했습니다.", 500)