import cv2
import numpy as np
import matplotlib.pyplot as plt

class teethDiagnosis:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError("이미지를 로드할 수 없습니다.")
    
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
    
        # 원본 이미지에 충치 부분 표시 (파란색으로)
        result_img = self.img.copy()
        caries_indices = np.where(caries_mask == 255)
        result_img[caries_indices[0], caries_indices[1], :] = [255, 0, 0]  # 파란색 (BGR)

        cv2.imwrite(f"{self.image_path}_caries_detected.jpg", result_img)

        return result_img
    
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

        # 원본 이미지에 치석 부분 표시 (빨간색으로 하이라이트)
        result_img = self.img.copy()
        # 마스크에서 흰색(255) 픽셀 위치 찾기
        indices = np.where(cleaned_mask == 255)
        # 해당 위치에 빨간색 적용 (BGR 형식)
        result_img[indices[0], indices[1], :] = [0, 0, 255]

        cv2.imwrite(f"{self.image_path}_calculus.jpg", result_img)

        return result_img
    
    