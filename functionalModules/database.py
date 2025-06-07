import mysql.connector
from mysql.connector import Error
import os
from contextlib import contextmanager

class DatabaseManager:
    """데이터베이스 연결 및 관리 클래스"""
    
    def __init__(self):
        self.MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
        self.MYSQL_USER = os.environ.get("MYSQL_USER", "root")
        self.MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
        self.MYSQL_DATABASE = "pet_dental_care"

    def get_connection(self):
        """데이터베이스 연결 생성"""
        try:
            connection = mysql.connector.connect(
                host=self.MYSQL_HOST,
                user=self.MYSQL_USER,
                password=self.MYSQL_PASSWORD,
                database=self.MYSQL_DATABASE,
                charset='utf8mb4',
                collation='utf8mb4_unicode_ci',
                autocommit=False
            )
            return connection
        except Error as e:
            print(f"데이터베이스 연결 오류: {e}")
            return None

    @contextmanager
    def get_db_connection(self):
        """컨텍스트 매니저를 사용한 안전한 데이터베이스 연결"""
        connection = None
        try:
            connection = self.get_connection()
            if connection:
                yield connection
            else:
                raise Exception("데이터베이스 연결 실패")
        except Exception as e:
            if connection:
                connection.rollback()
            raise e
        finally:
            if connection and connection.is_connected():
                connection.close()

    def test_connection(self):
        """데이터베이스 연결 테스트"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                return True
        except:
            return False
