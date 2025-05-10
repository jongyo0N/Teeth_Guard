import cv2
import numpy as np
import matplotlib.pyplot as plt

class teethDiagnosis:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError("이미지를 로드할 수 없습니다.")
        self.tooth_mask = None
        self.caries_mask = None
        self.calculus_mask = None

    def detect_total_teeth(self):
        """치아 전체 영역(흰색 치아, 충치, 치석 포함)을 감지하는 메서드"""
        # BGR에서 HSV 색상 공간으로 변환
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        # 1. 흰색/연한 노란색 치아 영역 (건강한 치아)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([30, 30, 255])
        
        # 2. 노란색/연한 갈색 영역 (치석)
        lower_yellow = np.array([20, 30, 100])
        upper_yellow = np.array([40, 255, 255])
        
        # 3. 갈색 영역 (치석의 다른 색상)
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([20, 255, 200])
        
        # 4. 어두운 갈색/검은색 영역 (충치)
        lower_dark_brown = np.array([0, 30, 0])
        upper_dark_brown = np.array([20, 255, 100])
        
        # 5. 검은색/회색 영역 (충치의 다른 색상)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        
        # 각 색상 범위에 대한 마스크 생성
        white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
        yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
        brown_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
        dark_brown_mask = cv2.inRange(hsv_img, lower_dark_brown, upper_dark_brown)
        black_mask = cv2.inRange(hsv_img, lower_black, upper_black)
        
        # 모든 마스크 결합 (흰색 치아 + 치석 + 충치)
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        combined_mask = cv2.bitwise_or(combined_mask, brown_mask)
        combined_mask = cv2.bitwise_or(combined_mask, dark_brown_mask)
        combined_mask = cv2.bitwise_or(combined_mask, black_mask)
        
        # 노이즈 제거를 위한 모폴로지 연산
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # 연결 요소 분석을 통해 작은 영역 제거
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, 8)
        min_size = 100  # 최소 영역 크기 (필요에 따라 조정)
        
        # 배경(인덱스 0)을 제외하고 일정 크기 이상의 영역만 유지
        refined_mask = np.zeros_like(combined_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                refined_mask[labels == i] = 255
        
        # 또는 사용자가 관심 영역을 지정했다면 그 영역으로 제한
        if hasattr(self, 'tooth_mask') and self.tooth_mask is not None:
            refined_mask = cv2.bitwise_and(refined_mask, self.tooth_mask)
        
        # 클래스 변수에 마스크 저장
        self.tooth_mask = refined_mask
        
        # 시각화를 위한 결과 이미지
        result_img = self.img.copy()
        # 마스크 영역 표시 (녹색으로)
        result_img[refined_mask == 255] = [0, 255, 0]  # 녹색 (BGR)
        
        cv2.imwrite(f"{self.image_path}_teeth_full_area.jpg", result_img)
        
        return refined_mask

    def detect_dental_caries(self):
    
        # BGR에서 HSV 색상 공간으로 변환
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # 충치의 HSV 색상 범위 설정 (어두운 갈색/검은색)
        # 어두운 갈색 범위 (Hue: 0-20, Saturation: 30-255, Value: 0-100)
        lower_dark_brown = np.array([0, 30, 0])
        upper_dark_brown = np.array([20, 255, 100])

        # 검은색/회색 범위 (낮은 명도)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])  # 명도(V)가 낮은 모든 색상

        # 각 색상 범위에 대한 마스크 생성
        dark_brown_mask = cv2.inRange(hsv_img, lower_dark_brown, upper_dark_brown)
        black_mask = cv2.inRange(hsv_img, lower_black, upper_black)

        # 충치 마스크 결합 (어두운 갈색 + 검은색)
        caries_mask = cv2.bitwise_or(dark_brown_mask, black_mask)

        # 노이즈 제거를 위한 모폴로지 연산
        kernel = np.ones((3, 3), np.uint8)
        caries_mask = cv2.morphologyEx(caries_mask, cv2.MORPH_OPEN, kernel)
        caries_mask = cv2.morphologyEx(caries_mask, cv2.MORPH_CLOSE, kernel)
        
        # 치아 영역이 감지되지 않은 경우 먼저 감지
        if self.tooth_mask is None:
            self.detect_teeth_area()
        
        # 치아 영역 내의 충치만 유지 (선택적)
        if self.tooth_mask is not None:
            caries_mask = cv2.bitwise_and(caries_mask, self.tooth_mask)
        
        # 클래스 변수에 충치 마스크 저장
        self.caries_mask = caries_mask

        # 원본 이미지에 충치 부분 표시 (파란색으로)
        result_img = self.img.copy()
        caries_indices = np.where(caries_mask == 255)
        result_img[caries_indices[0], caries_indices[1], :] = [255, 0, 0]  # 파란색 (BGR)

        cv2.imwrite(f"{self.image_path}_caries_detected.jpg", result_img)

        # 충치 마스크 반환으로 변경
        return caries_mask
        
    def detect_dental_calculus(self):
        # BGR에서 HSV 색상 공간으로 변환
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # 치석의 HSV 색상 범위 설정 (이 값은 실제 치석 이미지에 따라 조정 필요)
        # 일반적으로 치석은 노란색/갈색을 띄므로 해당 색상 범위를 타겟팅
        # 노란색/갈색 범위 (Hue: 20-40, Saturation: 50-255, Value: 50-255)
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([40, 255, 255])

        # 갈색 범위 (Hue: 10-20, Saturation: 50-255, Value: 50-255)
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([20, 255, 255])

        # 각 색상 범위에 대한 마스크 생성
        yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
        brown_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)

        # 두 마스크 결합
        combined_mask = cv2.bitwise_or(yellow_mask, brown_mask)

        # 노이즈 제거를 위한 모폴로지 연산
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # 치아 영역이 감지되지 않은 경우 먼저 감지
        if self.tooth_mask is None:
            self.detect_teeth_area()
        
        # 치아 영역 내의 치석만 유지 (선택적)
        if self.tooth_mask is not None:
            cleaned_mask = cv2.bitwise_and(cleaned_mask, self.tooth_mask)
        
        # 클래스 변수에 치석 마스크 저장
        self.calculus_mask = cleaned_mask

        # 원본 이미지에 치석 부분 표시 (빨간색으로 하이라이트)
        result_img = self.img.copy()
        # 마스크에서 흰색(255) 픽셀 위치 찾기
        indices = np.where(cleaned_mask == 255)
        # 해당 위치에 빨간색 적용 (BGR 형식)
        result_img[indices[0], indices[1], :] = [0, 0, 255]

        cv2.imwrite(f"{self.image_path}_calculus.jpg", result_img)

        # 치석 마스크 반환으로 변경
        return cleaned_mask
        
    def calculate_ratios(self): 
        # 필요한 마스크가 없으면 먼저 감지
        if self.tooth_mask is None:
            self.detect_teeth_area()
        if self.caries_mask is None:
            self.detect_dental_caries()
        if self.calculus_mask is None:
            self.detect_dental_calculus()
        
        # 픽셀 수 계산 (255값을 갖는 픽셀의 개수)
        total_tooth_pixels = np.sum(self.tooth_mask == 255)
        caries_pixels = np.sum(self.caries_mask == 255)
        calculus_pixels = np.sum(self.calculus_mask == 255)
        
        # 비율 계산 (0으로 나누기 방지)
        if total_tooth_pixels > 0:
            caries_ratio = (caries_pixels / total_tooth_pixels) * 100
            calculus_ratio = (calculus_pixels / total_tooth_pixels) * 100
        else:
            caries_ratio = 0
            calculus_ratio = 0
        
        # 결과 저장 및 출력
        results = {
            "total_tooth_area": total_tooth_pixels,
            "caries_area": caries_pixels,
            "calculus_area": calculus_pixels,
            "caries_ratio": caries_ratio,
            "calculus_ratio": calculus_ratio
        }
        
        print(f"전체 치아 픽셀 수: {total_tooth_pixels}")
        print(f"충치 픽셀 수: {caries_pixels}")
        print(f"치석 픽셀 수: {calculus_pixels}")
        print(f"충치 비율: {caries_ratio:.2f}%")
        print(f"치석 비율: {calculus_ratio:.2f}%")
        
        return results

if __name__ == "__main__":
    # 사용 예시
    image_path = "teeth3.jpg"
    
    # 치아 진단 객체 생성
    diagnosis = teethDiagnosis(image_path)
    
    # 치아 전체 영역 감지
    diagnosis.detect_total_teeth()
    
    # 충치 감지
    diagnosis.detect_dental_caries()
    
    # 치석 감지
    diagnosis.detect_dental_calculus()
    
    # 비율 계산 및 출력
    diagnosis.calculate_ratios()