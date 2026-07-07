# -*- coding: utf-8 -*-
"""
NEU-DET 데이터셋의 이미지 및 어노테이션 데이터를 분석하여
모델 학습 전 사전 데이터 분석(EDA)을 수행하는 스크립트입니다.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# 한글 폰트 설정 (Mac OS 전용 AppleGothic 설정)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정 (상대경로 사용)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "NEU-DET")
IMAGES_DIR = os.path.join(DATA_DIR, "IMAGES")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "ANNOTATIONS")
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, "images")

# 결과 저장 폴더 생성
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

def analyze_dataset():
    # 1. 파일 목록 확인
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.lower().endswith('.xml')]
    
    print(f"전체 이미지 수: {len(image_files)}")
    print(f"전체 XML 어노테이션 수: {len(xml_files)}")
    
    # 데이터 프레임을 만들기 위한 리스트들
    image_data = []
    bbox_data = []
    
    # 2. 각 이미지와 XML 파싱
    for img_name in image_files:
        base_name = os.path.splitext(img_name)[0]
        xml_name = base_name + ".xml"
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_name)
        img_path = os.path.join(IMAGES_DIR, img_name)
        
        # 이미지 정보 획득 (PIL 이용)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                channels = len(img.getbands())
                # 이미지 픽셀 강도 통계
                img_np = np.array(img)
                pixel_mean = np.mean(img_np)
                pixel_std = np.std(img_np)
                pixel_min = np.min(img_np)
                pixel_max = np.max(img_np)
        except Exception as e:
            print(f"이미지 읽기 실패: {img_name}, 에러: {e}")
            continue
            
        image_data.append({
            "image_name": img_name,
            "width": width,
            "height": height,
            "channels": channels,
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
            "pixel_min": pixel_min,
            "pixel_max": pixel_max,
            "has_xml": os.path.exists(xml_path)
        })
        
        # XML 파일 파싱 (객체 탐지 정보)
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # XML 파일 내 객체 수
                num_objects = len(root.findall('object'))
                
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    # bbox 속성 계산
                    bbox_width = xmax - xmin
                    bbox_height = ymax - ymin
                    bbox_area = bbox_width * bbox_height
                    bbox_ratio = bbox_width / (bbox_height + 1e-6)
                    center_x = xmin + bbox_width / 2.0
                    center_y = ymin + bbox_height / 2.0
                    
                    bbox_data.append({
                        "image_name": img_name,
                        "class_name": class_name,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "bbox_width": bbox_width,
                        "bbox_height": bbox_height,
                        "bbox_area": bbox_area,
                        "bbox_ratio": bbox_ratio,
                        "center_x": center_x,
                        "center_y": center_y,
                        "img_width": width,
                        "img_height": height
                    })
            except Exception as e:
                print(f"XML 파싱 실패: {xml_name}, 에러: {e}")
                
    # Pandas DataFrame 변환
    df_images = pd.DataFrame(image_data)
    df_bboxes = pd.DataFrame(bbox_data)
    
    # XML 매칭 통계 출력
    print(f"어노테이션이 존재하는 이미지 수: {df_images['has_xml'].sum()}")
    print(f"총 검출된 결함 객체(Bbox) 수: {len(df_bboxes)}")
    
    return df_images, df_bboxes

def generate_visualizations(df_images, df_bboxes):
    # --- 시각화 1: 클래스별 결함 객체 수 분포 ---
    plt.figure(figsize=(10, 6))
    class_counts = df_bboxes['class_name'].value_counts()
    sns.barplot(x=class_counts.values, y=class_counts.index, palette='viridis')
    plt.title('클래스별 결함 객체(Bounding Box) 수 분포', fontsize=14)
    plt.xlabel('객체 수 (개)', fontsize=12)
    plt.ylabel('결함 클래스명', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, '01_class_distribution.png'))
    plt.close()
    
    # --- 시각화 2: 이미지당 결함 객체 수 분포 ---
    plt.figure(figsize=(10, 6))
    bboxes_per_img = df_bboxes.groupby('image_name').size()
    # 결함이 없는 이미지는 0개로 추가
    all_imgs = df_images['image_name']
    bboxes_per_img = bboxes_per_img.reindex(all_imgs, fill_value=0)
    
    sns.histplot(bboxes_per_img, bins=np.arange(0, bboxes_per_img.max() + 2) - 0.5, kde=False, color='skyblue', edgecolor='black')
    plt.title('이미지당 결함 객체 수 분포', fontsize=14)
    plt.xlabel('결함 수 (개)', fontsize=12)
    plt.ylabel('이미지 수 (장)', fontsize=12)
    plt.xticks(range(int(bboxes_per_img.max()) + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, '02_defect_count_per_image.png'))
    plt.close()

    # --- 시각화 3: 결함 객체 크기(Bbox Area) 분포 ---
    plt.figure(figsize=(10, 6))
    # 이미지 면적 대비 비율 계산
    df_bboxes['area_ratio'] = df_bboxes['bbox_area'] / (df_bboxes['img_width'] * df_bboxes['img_height']) * 100
    sns.histplot(data=df_bboxes, x='area_ratio', hue='class_name', multiple='stack', bins=30, palette='Set2')
    plt.title('이미지 면적 대비 결함 영역 크기 비율 분포 (%)', fontsize=14)
    plt.xlabel('결함 영역 비율 (%)', fontsize=12)
    plt.ylabel('객체 수 (개)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, '03_bbox_area_ratio.png'))
    plt.close()

    # --- 시각화 4: 결함 객체의 가로/세로 크기 산점도 ---
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_bboxes, x='bbox_width', y='bbox_height', hue='class_name', alpha=0.7, palette='tab10')
    plt.title('결함 객체의 가로 및 세로 크기 산점도 (픽셀)', fontsize=14)
    plt.xlabel('결함 가로 크기 (Width, px)', fontsize=12)
    plt.ylabel('결함 세로 크기 (Height, px)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, '04_bbox_dimensions_scatter.png'))
    plt.close()

    # --- 시각화 5: 클래스별 결함 종횡비(Aspect Ratio) Boxplot ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_bboxes, x='class_name', y='bbox_ratio', palette='pastel')
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='정방형 (Ratio=1.0)')
    plt.title('클래스별 결함 종횡비(가로/세로) 분포', fontsize=14)
    plt.xlabel('결함 클래스명', fontsize=12)
    plt.ylabel('종횡비 (Width / Height)', fontsize=12)
    plt.yscale('log')  # 종횡비 편차가 클 수 있으므로 로그 스케일 적용
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, '05_bbox_aspect_ratio.png'))
    plt.close()

    # --- 시각화 6: 이미지 픽셀 평균 밝기(Pixel Intensity Mean) 분포 ---
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_images, x='pixel_mean', kde=True, color='purple', bins=30)
    plt.title('이미지별 평균 픽셀 밝기(Intensity Mean) 분포', fontsize=14)
    plt.xlabel('평균 픽셀 값 (0-255)', fontsize=12)
    plt.ylabel('이미지 수 (장)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, '06_image_pixel_mean.png'))
    plt.close()

    # --- 시각화 7: 이미지 픽셀 대비(표준편차, Standard Deviation) 분포 ---
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_images, x='pixel_std', kde=True, color='teal', bins=30)
    plt.title('이미지별 픽셀 표준편차(Intensity Std) 분포', fontsize=14)
    plt.xlabel('픽셀 표준편차 (명암 대비)', fontsize=12)
    plt.ylabel('이미지 수 (장)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, '07_image_pixel_std.png'))
    plt.close()

    # --- 시각화 8: 결함 중심점(Center X, Y) 위치 2D Density Plot (Heatmap) ---
    plt.figure(figsize=(8, 8))
    # 전체 이미지 영역 크기 설정 (대부분 200x200 픽셀)
    sns.kdeplot(data=df_bboxes, x='center_x', y='center_y', cmap='Reds', fill=True, thresh=0.05, bw_adjust=0.5)
    plt.xlim(0, 200)
    plt.ylim(200, 0) # 이미지 좌표계 특성상 Y축을 반전시킴
    plt.title('결함 객체 중심 좌표의 2차원 밀도 분포 (Heatmap)', fontsize=14)
    plt.xlabel('중심 X 좌표 (px)', fontsize=12)
    plt.ylabel('중심 Y 좌표 (px)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, '08_bbox_center_heatmap.png'))
    plt.close()

    # --- 시각화 9: 결함 클래스별 영역 면적 차이 Boxplot ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_bboxes, x='class_name', y='bbox_area', palette='Set3')
    plt.title('클래스별 결함 객체 실제 면적(픽셀 수) 분포', fontsize=14)
    plt.xlabel('결함 클래스명', fontsize=12)
    plt.ylabel('결함 면적 (px^2)', fontsize=12)
    plt.yscale('log') # 큰 결함과 작은 결함 격차 완화를 위해 로그 스케일
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, '09_class_bbox_area.png'))
    plt.close()

    # --- 시각화 10: 이미지별 픽셀 평균 vs 표준편차 상관관계 산점도 ---
    plt.figure(figsize=(10, 8))
    # 파일명 앞부분을 기준으로 결함 클래스 분류 추출
    df_images['class_prefix'] = df_images['image_name'].apply(lambda x: x.split('_')[0])
    sns.scatterplot(data=df_images, x='pixel_mean', y='pixel_std', hue='class_prefix', alpha=0.8, palette='Dark2')
    plt.title('이미지 평균 밝기 대 명암 대비(표준편차) 상관관계 산점도', fontsize=14)
    plt.xlabel('평균 픽셀 밝기 (Mean)', fontsize=12)
    plt.ylabel('픽셀 표준편차 (Std)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, '10_pixel_mean_vs_std.png'))
    plt.close()

def generate_text_statistics(df_images, df_bboxes):
    stats = {}
    
    # 1. 이미지 크기 통계
    stats['img_width_mean'] = df_images['width'].mean()
    stats['img_height_mean'] = df_images['height'].mean()
    
    # 2. 이미지당 결함 수 통계
    bboxes_per_img = df_bboxes.groupby('image_name').size().reindex(df_images['image_name'], fill_value=0)
    stats['defect_per_img_mean'] = bboxes_per_img.mean()
    stats['defect_per_img_max'] = bboxes_per_img.max()
    stats['defect_per_img_min'] = bboxes_per_img.min()
    stats['defect_per_img_std'] = bboxes_per_img.std()
    
    # 3. 클래스별 객체 수 및 비율
    class_counts = df_bboxes['class_name'].value_counts()
    class_ratios = df_bboxes['class_name'].value_counts(normalize=True)
    stats['class_counts'] = class_counts.to_dict()
    stats['class_ratios'] = class_ratios.to_dict()
    
    # 4. 결함 크기(bbox_area) 통계
    stats['bbox_area_mean'] = df_bboxes['bbox_area'].mean()
    stats['bbox_area_median'] = df_bboxes['bbox_area'].median()
    stats['bbox_area_min'] = df_bboxes['bbox_area'].min()
    stats['bbox_area_max'] = df_bboxes['bbox_area'].max()
    
    # 5. 결함 종횡비 통계
    stats['bbox_ratio_mean'] = df_bboxes['bbox_ratio'].mean()
    stats['bbox_ratio_median'] = df_bboxes['bbox_ratio'].median()
    
    # 6. 이미지 픽셀값 통계
    stats['pixel_mean_avg'] = df_images['pixel_mean'].mean()
    stats['pixel_std_avg'] = df_images['pixel_std'].mean()
    
    # 결과 출력
    print("\n--- 기술 통계 요약 ---")
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v:.4f}" if isinstance(sub_v, float) else f"  {sub_k}: {sub_v}")
        else:
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            
    return stats

if __name__ == "__main__":
    print("데이터셋 로딩 및 분석 시작...")
    df_images, df_bboxes = analyze_dataset()
    
    print("시각화 생성 중...")
    generate_visualizations(df_images, df_bboxes)
    
    print("기술통계 요약 계산 중...")
    stats = generate_text_statistics(df_images, df_bboxes)
    
    print("모든 작업이 성공적으로 완료되었습니다!")
