import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 이미지 파일 경로
image_path = "teeth2.jpg"

# 색상 기반 치아/잇몸 세그멘테이션 함수
def color_based_teeth_segmentation(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return None, None, None
    
    # 결과 저장을 위한 원본 이미지 복사
    original = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # HSV 색상 공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # HSV 채널 분리
    h, s, v = cv2.split(hsv)
    
    # 치아 영역 마스크 (밝은 흰색/노란색)
    # H: 0-30 (노란색 영역), S: 낮음 (채도 낮음), V: 높음 (밝기 높음)
    lower_teeth = np.array([0, 0, 180])
    upper_teeth = np.array([30, 80, 255])
    teeth_mask = cv2.inRange(hsv, lower_teeth, upper_teeth)
    
    # 잇몸 영역 마스크 (분홍색/붉은색)
    # H: 140-180 (붉은색 영역), S: 중간-높음, V: 중간-높음
    lower_gums = np.array([140, 50, 50])
    upper_gums = np.array([180, 255, 255])
    gums_mask = cv2.inRange(hsv, lower_gums, upper_gums)
    
    # 노이즈 제거 및 마스크 개선
    kernel = np.ones((3, 3), np.uint8)
    teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_OPEN, kernel)
    teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE, kernel)
    
    gums_mask = cv2.morphologyEx(gums_mask, cv2.MORPH_OPEN, kernel)
    gums_mask = cv2.morphologyEx(gums_mask, cv2.MORPH_CLOSE, kernel)
    
    # 결과 이미지 생성
    teeth_result = cv2.bitwise_and(image, image, mask=teeth_mask)
    gums_result = cv2.bitwise_and(image, image, mask=gums_mask)
    
    # 통합 마스크 생성
    enhanced_mask = np.zeros_like(teeth_mask)
    enhanced_mask[teeth_mask > 0] = 1
    enhanced_mask[gums_mask > 0] = 2
    
    # 컬러 오버레이 생성
    overlay = image.copy()
    
    # 치아 영역 파란색으로 표시
    teeth_overlay = np.zeros_like(image)
    teeth_overlay[teeth_mask > 0] = [255, 255, 0]  # 노란색 (BGR)
    
    # 잇몸 영역 빨간색으로 표시
    gums_overlay = np.zeros_like(image)
    gums_overlay[gums_mask > 0] = [255, 0, 255]  # 분홍색 (BGR)
    
    # 오버레이 합치기
    overlay = cv2.addWeighted(overlay, 0.7, teeth_overlay, 0.3, 0)
    overlay = cv2.addWeighted(overlay, 0.7, gums_overlay, 0.3, 0)
    
    # 시각화
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 3, 1)
    plt.imshow(image_rgb)
    plt.title("원본 이미지")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(teeth_mask, cmap='gray')
    plt.title("치아 마스크")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(gums_mask, cmap='gray')
    plt.title("잇몸 마스크")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(teeth_result, cv2.COLOR_BGR2RGB))
    plt.title("치아 영역")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(gums_result, cv2.COLOR_BGR2RGB))
    plt.title("잇몸 영역")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("세그멘테이션 결과")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_result.png')
    plt.show()
    
    # 마스크 반환
    return enhanced_mask, overlay, teeth_mask

# 개별 치아 세그멘테이션 함수 (워터쉐드 알고리즘 사용)
def segment_individual_teeth(image_path, teeth_mask):
    # 원본 이미지 로드
    image = cv2.imread(image_path)
    
    # 치아 마스크만 추출
    binary_teeth_mask = teeth_mask.copy()
    
    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    teeth_mask_clean = cv2.morphologyEx(binary_teeth_mask, cv2.MORPH_OPEN, kernel)
    teeth_mask_clean = cv2.morphologyEx(teeth_mask_clean, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    
    # 거리 변환 적용
    dist_transform = cv2.distanceTransform(teeth_mask_clean, cv2.DIST_L2, 5)
    
    # 거리 변환 정규화 (시각화용)
    dist_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # 전경 마커 생성 (임계값 기반)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    # 배경 마커 생성 (팽창 사용)
    sure_bg = cv2.dilate(teeth_mask_clean, kernel, iterations=3)
    
    # 알 수 없는 영역 찾기
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 마커 생성
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # 마커에 1 추가 (배경은 0이어야 함)
    markers = markers + 1
    
    # 알 수 없는 영역을 0으로 표시
    markers[unknown == 255] = 0
    
    # 워터쉐드 알고리즘 적용
    markers = cv2.watershed(image, markers)
    
    # 경계 표시
    image[markers == -1] = [0, 0, 255]  # 빨간색으로 경계 표시
    
    # 결과 시각화
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    plt.imshow(teeth_mask, cmap='gray')
    plt.title("치아 마스크")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(dist_norm, cmap='jet')
    plt.title("거리 변환")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(markers, cmap='jet')
    plt.title("개별 치아 분할")
    plt.colorbar(label='치아 ID')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("치아 경계 표시")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('teeth_separation.png')
    plt.show()
    
    # 개별 치아 마스크 생성
    unique_teeth = np.unique(markers)[2:]  # 배경(0)과 경계(-1) 제외
    teeth_masks = {}
    
    # 개별 치아 시각화
    if len(unique_teeth) > 0:
        cols = min(4, len(unique_teeth))
        rows = (len(unique_teeth) + cols - 1) // cols
        plt.figure(figsize=(16, 4 * rows))
        
        for i, tooth_id in enumerate(unique_teeth):
            # 개별 치아 마스크
            tooth_mask = (markers == tooth_id).astype(np.uint8) * 255
            teeth_masks[int(tooth_id)] = tooth_mask
            
            # 원본 이미지에 마스크 적용
            image = cv2.imread(image_path)
            tooth_segment = cv2.bitwise_and(image, image, mask=tooth_mask)
            
            # 치아 면적
            pixel_count = np.sum(tooth_mask > 0)
            
            plt.subplot(rows, cols, i + 1)
            plt.imshow(cv2.cvtColor(tooth_segment, cv2.COLOR_BGR2RGB))
            plt.title(f"치아 {i+1} (면적: {pixel_count}픽셀)")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('individual_teeth.png')
        plt.show()
    else:
        print("개별 치아를 감지할 수 없습니다.")
    
    return markers, teeth_masks

# 메인 실행
if __name__ == "__main__":
    print("색상 기반 치아 및 잇몸 세그멘테이션 시작...")
    
    # 이미지 파일 존재 확인
    if not os.path.exists(image_path):
        print(f"오류: 파일을 찾을 수 없습니다: {image_path}")
        print(f"현재 작업 디렉토리: {os.getcwd()}")
        exit(1)
    else:
        print(f"파일 확인됨: {image_path}")
    
    # 1단계: 치아와 잇몸 세그멘테이션
    enhanced_mask, overlay, teeth_mask = color_based_teeth_segmentation(image_path)
    
    if enhanced_mask is not None:
        # 결과 저장
        cv2.imwrite('segmentation_overlay.png', overlay)
        
        # 2단계: 개별 치아 세그멘테이션
        print("\n개별 치아 세그멘테이션 시작...")
        markers, individual_teeth = segment_individual_teeth(image_path, teeth_mask)
        
        # 개별 치아 마스크 저장
        os.makedirs('tooth_masks', exist_ok=True)
        for tooth_id, mask in individual_teeth.items():
            cv2.imwrite(f'tooth_masks/tooth_{tooth_id}.png', mask)
        
        print("\n세그멘테이션 완료!")
        print("결과 이미지가 저장되었습니다.")