"""
스케줄 관리 모듈 (Schedule Management)
- 캘린더에서 날짜 선택하여 스케줄 날짜 추가/삭제/완료 처리
- MySQL 데이터베이스와 연동
- 양치/케어 스케줄 통합 관리
"""
from functionalModules.database import DatabaseManager
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date, timedelta
import calendar
from dataclasses import dataclass
from enum import Enum

class ScheduleType(Enum):
    """스케줄 타입 열거형"""
    BRUSH = "양치"
    CARE = "케어"

class CompletionStatus(Enum):
    """완료 상태 열거형"""
    PENDING = 0
    COMPLETED = 1

@dataclass
class ScheduleDateInfo:
    """스케줄 날짜 정보 데이터 클래스"""
    schedule_date_id: int
    guardian_id: int
    pet_id: int
    schedule_id: int
    schedule_type: str
    scheduled_date: date
    is_completed: bool
    schedule_name: Optional[str] = None

@dataclass
class ScheduleInfo:
    """스케줄 기본 정보 데이터 클래스"""
    schedule_id: int
    guardian_id: int
    pet_id: int
    schedule_name: str
    care_type: str
    start_date: date
    is_active: bool

class ScheduleManagementError(Exception):
    """스케줄 관리 관련 예외"""
    pass

class ScheduleManager:
    """스케줄 관리 클래스"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def get_schedule_info(self, schedule_id: int) -> Optional[ScheduleInfo]:
        """스케줄 기본 정보 조회"""
        try:
            with self.db_manager.get_db_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                query = """
                SELECT schedule_id, guardian_id, pet_id, schedule_name, 
                       care_type, start_date, is_active
                FROM care_schedule
                WHERE schedule_id = %s
                """
                
                cursor.execute(query, (schedule_id,))
                result = cursor.fetchone()
                
                if result:
                    return ScheduleInfo(**result)
                return None
                
        except Exception as e:
            raise ScheduleManagementError(f"스케줄 정보 조회 실패: {e}")
    
    def get_user_schedules(self, guardian_id: int, pet_id: int, 
                          schedule_type: Optional[ScheduleType] = None) -> List[ScheduleInfo]:
        """사용자의 모든 활성 스케줄 조회"""
        try:
            with self.db_manager.get_db_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                query = """
                SELECT schedule_id, guardian_id, pet_id, schedule_name, 
                       care_type, start_date, is_active
                FROM care_schedule
                WHERE guardian_id = %s AND pet_id = %s AND is_active = 1
                """
                params = [guardian_id, pet_id]
                
                if schedule_type:
                    query += " AND care_type = %s"
                    params.append(schedule_type.value)
                
                query += " ORDER BY schedule_name"
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                return [ScheduleInfo(**result) for result in results]
                
        except Exception as e:
            raise ScheduleManagementError(f"사용자 스케줄 조회 실패: {e}")
    
    def get_schedule_dates(self, schedule_id: int, 
                          start_date: Optional[date] = None,
                          end_date: Optional[date] = None) -> List[ScheduleDateInfo]:
        """특정 스케줄의 날짜들 조회"""
        try:
            with self.db_manager.get_db_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                query = """
                SELECT sd.schedule_date_id, sd.guardian_id, sd.pet_id, 
                       sd.schedule_id, sd.schedule_type, sd.scheduled_date, 
                       sd.is_completed, cs.schedule_name
                FROM schedule_dates sd
                JOIN care_schedule cs ON sd.schedule_id = cs.schedule_id
                WHERE sd.schedule_id = %s
                """
                params = [schedule_id]
                
                if start_date:
                    query += " AND sd.scheduled_date >= %s"
                    params.append(start_date)
                
                if end_date:
                    query += " AND sd.scheduled_date <= %s"
                    params.append(end_date)
                
                query += " ORDER BY sd.scheduled_date"
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                return [ScheduleDateInfo(**result) for result in results]
                
        except Exception as e:
            raise ScheduleManagementError(f"스케줄 날짜 조회 실패: {e}")
    
    def get_monthly_schedule_dates(self, guardian_id: int, pet_id: int, 
                                 year: int, month: int,
                                 schedule_type: Optional[ScheduleType] = None) -> List[ScheduleDateInfo]:
        """특정 월의 모든 스케줄 날짜 조회"""
        try:
            # 해당 월의 첫째 날과 마지막 날
            first_day = date(year, month, 1)
            last_day = date(year, month, calendar.monthrange(year, month)[1])
            
            with self.db_manager.get_db_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                query = """
                SELECT sd.schedule_date_id, sd.guardian_id, sd.pet_id, 
                       sd.schedule_id, sd.schedule_type, sd.scheduled_date, 
                       sd.is_completed, cs.schedule_name
                FROM schedule_dates sd
                JOIN care_schedule cs ON sd.schedule_id = cs.schedule_id
                WHERE sd.guardian_id = %s AND sd.pet_id = %s
                AND sd.scheduled_date BETWEEN %s AND %s
                """
                params = [guardian_id, pet_id, first_day, last_day]
                
                if schedule_type:
                    query += " AND sd.schedule_type = %s"
                    params.append(schedule_type.value)
                
                query += " ORDER BY sd.scheduled_date, cs.schedule_name"
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                return [ScheduleDateInfo(**result) for result in results]
                
        except Exception as e:
            raise ScheduleManagementError(f"월별 스케줄 조회 실패: {e}")
    
    def add_schedule_dates(self, schedule_id: int, dates: List[date]) -> Dict[str, Any]:
        """스케줄에 새로운 날짜들 추가"""
        if not dates:
            return {"success": True, "added_count": 0, "duplicates": []}
        
        try:
            # 스케줄 정보 확인
            schedule_info = self.get_schedule_info(schedule_id)
            if not schedule_info:
                raise ScheduleManagementError(f"스케줄을 찾을 수 없습니다: {schedule_id}")
            
            with self.db_manager.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 기존 날짜들 확인
                existing_query = """
                SELECT scheduled_date FROM schedule_dates 
                WHERE schedule_id = %s AND scheduled_date IN ({})
                """.format(','.join(['%s'] * len(dates)))
                
                cursor.execute(existing_query, [schedule_id] + dates)
                existing_dates = {row[0] for row in cursor.fetchall()}
                
                # 새로운 날짜들만 필터링
                new_dates = [d for d in dates if d not in existing_dates]
                duplicates = [d for d in dates if d in existing_dates]
                
                if new_dates:
                    # 새로운 날짜들 추가
                    insert_query = """
                    INSERT INTO schedule_dates 
                    (guardian_id, pet_id, schedule_type, schedule_id, scheduled_date, is_completed)
                    VALUES (%s, %s, %s, %s, %s, 0)
                    """
                    
                    values = [
                        (schedule_info.guardian_id, schedule_info.pet_id, 
                         schedule_info.care_type, schedule_id, new_date)
                        for new_date in new_dates
                    ]
                    
                    cursor.executemany(insert_query, values)
                    conn.commit()
                
                return {
                    "success": True,
                    "added_count": len(new_dates),
                    "duplicates": duplicates,
                    "added_dates": new_dates
                }
                
        except Exception as e:
            raise ScheduleManagementError(f"스케줄 날짜 추가 실패: {e}")
    
    def remove_schedule_dates(self, schedule_id: int, dates: List[date]) -> Dict[str, Any]:
        """스케줄에서 특정 날짜들 삭제"""
        if not dates:
            return {"success": True, "removed_count": 0}
        
        try:
            with self.db_manager.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 삭제할 날짜들 확인
                check_query = """
                SELECT scheduled_date FROM schedule_dates 
                WHERE schedule_id = %s AND scheduled_date IN ({})
                """.format(','.join(['%s'] * len(dates)))
                
                cursor.execute(check_query, [schedule_id] + dates)
                existing_dates = {row[0] for row in cursor.fetchall()}
                
                # 실제로 존재하는 날짜들만 삭제
                dates_to_remove = [d for d in dates if d in existing_dates]
                
                if dates_to_remove:
                    delete_query = """
                    DELETE FROM schedule_dates 
                    WHERE schedule_id = %s AND scheduled_date IN ({})
                    """.format(','.join(['%s'] * len(dates_to_remove)))
                    
                    cursor.execute(delete_query, [schedule_id] + dates_to_remove)
                    conn.commit()
                
                return {
                    "success": True,
                    "removed_count": len(dates_to_remove),
                    "not_found": [d for d in dates if d not in existing_dates]
                }
                
        except Exception as e:
            raise ScheduleManagementError(f"스케줄 날짜 삭제 실패: {e}")
    
    def toggle_completion_status(self, schedule_date_ids: List[int]) -> Dict[str, Any]:
        """스케줄 날짜들의 완료 상태 토글"""
        if not schedule_date_ids:
            return {"success": True, "updated_count": 0}
        
        try:
            with self.db_manager.get_db_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                # 현재 상태 조회
                check_query = """
                SELECT schedule_date_id, is_completed 
                FROM schedule_dates 
                WHERE schedule_date_id IN ({})
                """.format(','.join(['%s'] * len(schedule_date_ids)))
                
                cursor.execute(check_query, schedule_date_ids)
                current_statuses = {row['schedule_date_id']: row['is_completed'] 
                                  for row in cursor.fetchall()}
                
                if not current_statuses:
                    return {"success": True, "updated_count": 0, "not_found": schedule_date_ids}
                
                # 각 ID별로 상태 토글
                updates = []
                for schedule_date_id in schedule_date_ids:
                    if schedule_date_id in current_statuses:
                        new_status = 0 if current_statuses[schedule_date_id] else 1
                        updates.append((new_status, schedule_date_id))
                
                if updates:
                    update_query = """
                    UPDATE schedule_dates 
                    SET is_completed = %s 
                    WHERE schedule_date_id = %s
                    """
                    cursor.executemany(update_query, updates)
                    conn.commit()
                
                return {
                    "success": True,
                    "updated_count": len(updates),
                    "not_found": [id for id in schedule_date_ids if id not in current_statuses]
                }
                
        except Exception as e:
            raise ScheduleManagementError(f"완료 상태 업데이트 실패: {e}")
    
    def set_completion_status(self, schedule_date_ids: List[int], 
                            status: CompletionStatus) -> Dict[str, Any]:
        """스케줄 날짜들의 완료 상태를 특정 값으로 설정"""
        if not schedule_date_ids:
            return {"success": True, "updated_count": 0}
        
        try:
            with self.db_manager.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 존재하는 ID들 확인
                check_query = """
                SELECT schedule_date_id 
                FROM schedule_dates 
                WHERE schedule_date_id IN ({})
                """.format(','.join(['%s'] * len(schedule_date_ids)))
                
                cursor.execute(check_query, schedule_date_ids)
                existing_ids = {row[0] for row in cursor.fetchall()}
                
                # 존재하는 ID들만 업데이트
                valid_ids = [id for id in schedule_date_ids if id in existing_ids]
                
                if valid_ids:
                    update_query = """
                    UPDATE schedule_dates 
                    SET is_completed = %s 
                    WHERE schedule_date_id IN ({})
                    """.format(','.join(['%s'] * len(valid_ids)))
                    
                    cursor.execute(update_query, [status.value] + valid_ids)
                    conn.commit()
                
                return {
                    "success": True,
                    "updated_count": len(valid_ids),
                    "not_found": [id for id in schedule_date_ids if id not in existing_ids]
                }
                
        except Exception as e:
            raise ScheduleManagementError(f"완료 상태 설정 실패: {e}")
    
    def get_schedule_statistics(self, schedule_id: int, 
                              start_date: Optional[date] = None,
                              end_date: Optional[date] = None) -> Dict[str, Any]:
        """스케줄 통계 조회 (완료율, 총 개수 등)"""
        try:
            schedule_dates = self.get_schedule_dates(schedule_id, start_date, end_date)
            
            if not schedule_dates:
                return {
                    "total_count": 0,
                    "completed_count": 0,
                    "pending_count": 0,
                    "completion_rate": 0.0
                }
            
            total_count = len(schedule_dates)
            completed_count = sum(1 for sd in schedule_dates if sd.is_completed)
            pending_count = total_count - completed_count
            completion_rate = (completed_count / total_count) * 100 if total_count > 0 else 0.0
            
            return {
                "total_count": total_count,
                "completed_count": completed_count,
                "pending_count": pending_count,
                "completion_rate": round(completion_rate, 2)
            }
            
        except Exception as e:
            raise ScheduleManagementError(f"통계 조회 실패: {e}")
    
    def bulk_add_dates_by_pattern(self, schedule_id: int, 
                                start_date: date, end_date: date,
                                pattern: str = "daily") -> Dict[str, Any]:
        """패턴에 따라 대량 날짜 추가 (daily, weekly, etc.)"""
        try:
            dates_to_add = []
            current_date = start_date
            
            if pattern == "daily":
                while current_date <= end_date:
                    dates_to_add.append(current_date)
                    current_date += timedelta(days=1)
            
            elif pattern == "weekly":
                while current_date <= end_date:
                    dates_to_add.append(current_date)
                    current_date += timedelta(days=7)
            
            elif pattern == "weekdays":  # 평일만
                while current_date <= end_date:
                    if current_date.weekday() < 5:  # 0-4: 월-금
                        dates_to_add.append(current_date)
                    current_date += timedelta(days=1)
            
            elif pattern == "weekends":  # 주말만
                while current_date <= end_date:
                    if current_date.weekday() >= 5:  # 5-6: 토-일
                        dates_to_add.append(current_date)
                    current_date += timedelta(days=1)
            
            else:
                raise ScheduleManagementError(f"지원하지 않는 패턴: {pattern}")
            
            return self.add_schedule_dates(schedule_id, dates_to_add)
            
        except Exception as e:
            raise ScheduleManagementError(f"패턴 기반 날짜 추가 실패: {e}")


if __name__ == "__main__":
    pass