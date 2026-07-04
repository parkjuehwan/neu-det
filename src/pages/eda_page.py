# -*- coding: utf-8 -*-
"""
철강 표면 결함 데이터셋의 종합 기술 통계, 이미지 브라우징, 바운딩 박스 정밀 분석 및 7종 전처리 과정을 시각화하여 보고하는 대시보드 페이지 모듈입니다.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFilter
import platform

# Matplotlib 한글 폰트 설정 (다이어그램 한글 깨짐 방지)
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

def show_eda_page(df_images, df_bboxes, classes, IMAGES_DIR, CLASS_COLORS):
    """
    EDA 및 이미지 브라우징, 7종 전처리 과정을 대시보드상에 렌더링하는 함수입니다.

    인자:
        df_images (DataFrame): 이미지별 통계 메타데이터 데이터프레임
        df_bboxes (DataFrame): 바운딩 박스 결함 정보 데이터프레임
        classes (list): 결함 고유 클래스명 리스트
        IMAGES_DIR (str): 이미지 파일 로컬 디렉토리 경로
        CLASS_COLORS (dict): 클래스별 시각화 색상 매핑 딕셔너리
    """
    # 탭 메뉴 정의
    tab_eda, tab_browser, tab_cnn = st.tabs([
        "📊 데이터셋 통합 EDA", 
        "🔍 결함 이미지 검색 및 정밀 검사", 
        "🧠 CNN 전처리 및 이미지 심층 분석 (7종)"
    ])

    # 1.1 통합 EDA
    with tab_eda:
        st.header("📊 데이터셋 종합 기술통계 분석")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("전체 이미지 수", f"{len(df_images)} 장")
        with col2:
            st.metric("총 결함 객체(Bbox) 수", f"{len(df_bboxes)} 개")
        with col3:
            st.metric("이미지당 평균 결함 수", f"{df_bboxes.groupby('image_name').size().mean():.2f} 개")
        with col4:
            st.metric("평균 이미지 밝기 (0-255)", f"{df_images['pixel_mean'].mean():.1f}")
            
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("1. 결함 유형(클래스)별 객체 수 분포")
            fig, ax = plt.subplots(figsize=(8, 5))
            class_counts = df_bboxes['class_name'].value_counts()
            sns.barplot(x=class_counts.values, y=class_counts.index, ax=ax, palette="viridis")
            ax.set_title("결함 종류별 총 바운딩 박스 개수")
            ax.set_xlabel("검출 수 (개)")
            ax.set_ylabel("결함 클래스")
            st.pyplot(fig)
            plt.close(fig)
            
            st.subheader("2. 이미지 면적 대비 결함 크기 비율 (%)")
            fig, ax = plt.subplots(figsize=(8, 5))
            df_bboxes['area_ratio'] = df_bboxes['bbox_area'] / (200 * 200) * 100
            sns.histplot(data=df_bboxes, x='area_ratio', hue='class_name', multiple='stack', bins=30, ax=ax)
            ax.set_title("이미지 전체 면적 대비 Bounding Box 크기 비율 분포")
            ax.set_xlabel("결함 영역 비율 (%)")
            ax.set_ylabel("바운딩 박스 수 (개)")
            st.pyplot(fig)
            plt.close(fig)

        with col_chart2:
            st.subheader("3. 이미지별 밝기(Mean) 및 대비(Std) 분포")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df_images, x='pixel_mean', y='pixel_std', hue='class_prefix', ax=ax, alpha=0.7)
            ax.set_title("이미지 평균 밝기 대 대비(표준편차) 상관관계 산점도")
            ax.set_xlabel("평균 픽셀 밝기")
            ax.set_ylabel("픽셀 표준편차 (Contrast)")
            st.pyplot(fig)
            plt.close(fig)
            
            st.subheader("4. 결함 객체의 종횡비(Aspect Ratio) 분포")
            fig, ax = plt.subplots(figsize=(8, 5))
            df_bboxes['aspect_ratio'] = df_bboxes['bbox_width'] / (df_bboxes['bbox_height'] + 1e-6)
            sns.boxplot(data=df_bboxes, x='class_name', y='aspect_ratio', ax=ax, palette="Set2")
            ax.set_yscale('log')
            ax.axhline(1.0, color='red', linestyle='--')
            ax.set_title("클래스별 결함 종횡비 분포 (로그 스케일)")
            ax.set_xlabel("결함 유형")
            ax.set_ylabel("종횡비 (Width / Height)")
            st.pyplot(fig)
            plt.close(fig)

    # 1.2 결함 이미지 검색 및 정밀 검사
    with tab_browser:
        st.header("🔍 이미지 브라우징 및 클래스별 기술통계")
        
        selected_class = st.selectbox("조회할 결함 유형을 선택하세요:", ["전체"] + classes, key="browser_class_select")
        
        if selected_class == "전체":
            filtered_df = df_images
            filtered_bboxes = df_bboxes
        else:
            matching_images = df_bboxes[df_bboxes['class_name'] == selected_class]['image_name'].unique()
            filtered_df = df_images[df_images['image_name'].isin(matching_images)]
            filtered_bboxes = df_bboxes[df_bboxes['class_name'] == selected_class]
            
        st.markdown(f"#### 📊 [{selected_class}] 클래스 이미지 기술 통계 정보")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("해당 이미지 수", f"{len(filtered_df)} 장")
        with stat_col2:
            st.metric("해당 결함 인스턴스 수", f"{len(filtered_bboxes)} 개")
        with stat_col3:
            avg_area = filtered_bboxes['bbox_area'].mean() if not filtered_bboxes.empty else 0
            st.metric("결함 평균 면적", f"{avg_area:.1f} px²")
        with stat_col4:
            avg_ratio = (filtered_bboxes['bbox_width'] / filtered_bboxes['bbox_height']).mean() if not filtered_bboxes.empty else 0
            st.metric("결함 평균 종횡비", f"{avg_ratio:.2f}")

        if not filtered_bboxes.empty:
            with st.expander("📝 클래스별 상세 수치 기술 통계 보기"):
                desc_df = filtered_bboxes[['bbox_width', 'bbox_height', 'bbox_area']].describe()
                desc_df.columns = ["결함 너비 (px)", "결함 높이 (px)", "결함 면적 (px²)"]
                st.dataframe(desc_df, use_container_width=True)
                
                st.info(f"선택된 **{selected_class}** 결함의 너비는 평균 {filtered_bboxes['bbox_width'].mean():.1f}px 이며, 높이는 평균 {filtered_bboxes['bbox_height'].mean():.1f}px 입니다. "
                        f"가장 작은 결함은 {filtered_bboxes['bbox_area'].min()}px² 이며, 가장 큰 파손 범위는 {filtered_bboxes['bbox_area'].max()}px² 에 달해 넓은 형태학적 스케일 편차를 보여주고 있습니다.")

        st.markdown(f"선택된 결함 유형에 해당하는 이미지 목록: **{len(filtered_df)}장**")
        
        # 페이징 시스템 구축 (페이지당 30개씩 출력)
        items_per_page = 30
        total_items = len(filtered_df)
        total_pages = max((total_items - 1) // items_per_page + 1, 1)
        
        page_col1, page_col2 = st.columns([1, 4])
        with page_col1:
            current_page = st.number_input("페이지 선택", min_value=1, max_value=total_pages, value=1, step=1, key="page_select")
        
        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        page_images = filtered_df.iloc[start_idx:end_idx]
        
        st.markdown("### 2. 이미지 목록 (클릭하여 상세 정보 조회)")
        cols_per_row = 6
        rows = (len(page_images) - 1) // cols_per_row + 1
        
        for r in range(rows):
            row_cols = st.columns(cols_per_row)
            for c in range(cols_per_row):
                idx = r * cols_per_row + c
                if idx < len(page_images):
                    img_row = page_images.iloc[idx]
                    img_name = img_row['image_name']
                    img_path = os.path.join(IMAGES_DIR, img_name)
                    
                    try:
                        with Image.open(img_path) as thumb:
                            row_cols[c].image(thumb, use_container_width=True)
                    except Exception:
                        row_cols[c].write("이미지 로드 실패")
                        
                    if row_cols[c].button("🔍 상세검사", key=f"btn_{img_name}"):
                        st.session_state.selected_image = img_name
                        
        st.markdown("---")
        
        # 정밀 검사 상세 영역
        st.header(f"🔎 정밀 검사 정보: `{st.session_state.selected_image}`")
        target_img_name = st.session_state.selected_image
        img_row = df_images[df_images['image_name'] == target_img_name].iloc[0]
        img_path = os.path.join(IMAGES_DIR, target_img_name)
        
        try:
            orig_img = Image.open(img_path).convert("RGB")
            draw_img = orig_img.copy()
            draw = ImageDraw.Draw(draw_img)
            
            target_bboxes = df_bboxes[df_bboxes['image_name'] == target_img_name]
            for _, bbox in target_bboxes.iterrows():
                c_name = bbox['class_name']
                color = CLASS_COLORS.get(c_name, "#FFFFFF")
                draw.rectangle([bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']], outline=color, width=2)
                draw.text((bbox['xmin'] + 2, bbox['ymin'] + 2), c_name, fill=color)
        except Exception as e:
            st.error(f"상세 이미지를 읽어오는 중 에러가 발생했습니다: {e}")
            st.stop()
            
        detail_col1, detail_col2 = st.columns([1, 1])
        with detail_col1:
            st.subheader("📸 결함 경계 상자(Bounding Box) 오버레이")
            st.image(draw_img, caption="결함 검출 영역 오버레이 이미지", use_container_width=True)
            
        with detail_col2:
            st.subheader("📋 이미지 및 결함 객체 메타데이터 정보")
            meta_df = pd.DataFrame({
                "메타 데이터 항목": ["파일명", "가로 해상도", "세로 해상도", "채널 수", "평균 밝기", "픽셀 표준편차 (대비)"],
                "수치 및 정보": [
                    img_row['image_name'],
                    f"{img_row['width']} px",
                    f"{img_row['height']} px",
                    str(img_row['channels']),
                    f"{img_row['pixel_mean']:.2f}",
                    f"{img_row['pixel_std']:.2f}"
                ]
            })
            st.table(meta_df.set_index("메타 데이터 항목"))
            
            st.markdown(f"**현재 검출된 총 결함 인스턴스:** `{len(target_bboxes)}`개")
            if not target_bboxes.empty:
                display_bboxes = target_bboxes[["class_name", "xmin", "ymin", "xmax", "ymax", "bbox_width", "bbox_height", "bbox_area"]].copy()
                display_bboxes.columns = ["결함 종류", "X_Min", "Y_Min", "X_Max", "Y_Max", "너비(W)", "높이(H)", "면적(px^2)"]
                st.dataframe(display_bboxes, use_container_width=True)
            else:
                st.info("이 이미지에는 라벨링된 결함 정보가 없습니다.")
                
        st.markdown("---")
        
        # 픽셀 분석 영역
        st.subheader("🔢 이미지 픽셀 행렬 데이터 (Pixel Intensity Matrix) 분석")
        img_gray = np.array(Image.open(img_path).convert("L"))
        
        matrix_tab1, matrix_tab2 = st.tabs(["📊 2차원 픽셀 히트맵", "🔢 서브 행렬(Matrix) 값 조회"])
        
        with matrix_tab1:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(img_gray, cmap="gray", vmin=0, vmax=255)
            fig.colorbar(im, ax=ax, label="픽셀 강도 (0-255)")
            ax.set_title("200 x 200 픽셀 강도(Intensity) 히트맵")
            ax.set_xlabel("가로 픽셀 좌표 (X)")
            ax.set_ylabel("세로 픽셀 좌표 (Y)")
            st.pyplot(fig)
            plt.close(fig)
            
        with matrix_tab2:
            col_x = st.slider("가로(X) 조사 영역 범위 (20픽셀 단위)", min_value=0, max_value=180, value=90, step=10, key="slider_x")
            col_y = st.slider("세로(Y) 조사 영역 범위 (20픽셀 단위)", min_value=0, max_value=180, value=90, step=10, key="slider_y")
            
            sub_matrix = img_gray[col_y:col_y+20, col_x:col_x+20]
            cols_names = [f"X_{i}" for i in range(col_x, col_x + 20)]
            rows_names = [f"Y_{i}" for i in range(col_y, col_y + 20)]
            df_sub_matrix = pd.DataFrame(sub_matrix, columns=cols_names, index=rows_names)
            
            st.markdown(f"📍 **선택된 국소 행렬 좌표 영역**: X: `[{col_x} ~ {col_x + 19}]`, Y: `[{col_y} ~ {col_y + 19}]`")
            st.dataframe(df_sub_matrix, height=450, use_container_width=True)
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("해당 국소 영역 픽셀 평균", f"{np.mean(sub_matrix):.2f}")
            with col_m2:
                st.metric("해당 국소 영역 픽셀 최솟값", f"{np.min(sub_matrix)}")
            with col_m3:
                st.metric("해당 국소 영역 픽셀 최댓값", f"{np.max(sub_matrix)}")

    # 1.3 CNN 전처리 7종 분석
    with tab_cnn:
        st.header("🧠 CNN 모델 예측을 위한 이미지 심층 분석 및 전처리 파이프라인 (7종)")
        selected_img_cnn = st.session_state.selected_image
        img_path_cnn = os.path.join(IMAGES_DIR, selected_img_cnn)
        img_gray_cnn = Image.open(img_path_cnn).convert("L")
        img_np_cnn = np.array(img_gray_cnn)
        
        cnn_class_prefix = selected_img_cnn.rsplit('_', 1)[0]
        class_images = df_images[df_images['class_prefix'] == cnn_class_prefix]
        
        st.info(f"현재 심층 분석 타겟 이미지: `{selected_img_cnn}` (클래스: **{cnn_class_prefix}**)")
        st.markdown("---")

        # 1) 평균 이미지
        st.subheader("1️⃣ 클래스 평균 이미지 (Mean Image) 시각화")
        cnn_col1, cnn_col2 = st.columns([1, 2])
        with cnn_col1:
            mean_img_list = []
            sample_files = class_images['image_name'].head(30)
            for fname in sample_files:
                fpath = os.path.join(IMAGES_DIR, fname)
                try:
                    mean_img_list.append(np.array(Image.open(fpath).convert("L")))
                except Exception:
                    pass
            mean_array = np.mean(mean_img_list, axis=0) if mean_img_list else img_np_cnn
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(mean_array, cmap="gray")
            ax.set_title(f"{cnn_class_prefix} 클래스의 평균 합성 이미지 (30장)")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
        with cnn_col2:
            st.markdown("**💡 CNN 모델링 관점의 분석 및 제언**")
            st.write("클래스 평균 이미지는 해당 결함 클래스가 가지는 배경의 조도 특성과 결함 발생 위치의 중심 경향을 보여줍니다. "
                     "CNN이 위치 편향에 고착되지 않고 본질적 형태를 특징맵으로 포착하도록 돕는 배경 정규화 혹은 평행 이동(Translation) 증강 처리가 효과적입니다.")

        st.markdown("---")

        # 2) 표준편차 이미지
        st.subheader("2️⃣ 클래스 표준편차 이미지 (Standard Deviation Image) 시각화")
        cnn_col1, cnn_col2 = st.columns([1, 2])
        with cnn_col1:
            std_array = np.std(mean_img_list, axis=0) if mean_img_list else img_np_cnn
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(std_array, cmap="hot")
            fig.colorbar(im, ax=ax)
            ax.set_title(f"{cnn_class_prefix} 클래스의 픽셀 변동성(표준편차)")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
        with cnn_col2:
            st.markdown("**💡 CNN 모델링 관점의 분석 및 제언**")
            st.write("표준편차 핫스팟 이미지는 결함이 활성화될 확률이 높은 변동 영역을 추적합니다. "
                     "합성곱 신경망 가중치가 특정 외곽선 영역에 편향되어 오버핏되지 않도록 하기 위한 랜덤 회전(Rotation) 및 크롭(Crop) 등의 공간적 아규멘테이션의 정당성을 보여줍니다.")

        st.markdown("---")

        # 3) 픽셀 밝기 히스토그램
        st.subheader("3️⃣ 픽셀 밝기 분포(Pixel Intensity Histogram) 대조")
        cnn_col1, cnn_col2 = st.columns([1, 2])
        with cnn_col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(img_np_cnn.flatten(), bins=50, kde=True, color="purple", ax=ax)
            ax.set_title("현재 조사 대상 이미지의 픽셀 강도 분포")
            ax.set_xlabel("픽셀 값")
            ax.set_ylabel("빈도 수")
            st.pyplot(fig)
            plt.close(fig)
        with cnn_col2:
            st.markdown("**💡 CNN 모델링 관점의 분석 및 제언**")
            st.write("픽셀 강도 히스토그램은 조도 불균형 정도를 알려줍니다. CNN 학습 시 기울기 소실 및 폭발을 효과적으로 방지하기 위해 "
                     "데이터를 0~1 픽셀 범위로 정규화하거나 전체 데이터셋의 평균/편차를 적용하는 표준화(Normalization) 전처리가 필수적으로 권장됩니다.")

        st.markdown("---")

        # 4) 소벨 엣지 검출
        st.subheader("4️⃣ 소벨 필터(Sobel Filter)를 통한 수평/수직 경계선(Edge) 추출")
        cnn_col1, cnn_col2 = st.columns([1, 2])
        with cnn_col1:
            edge_img = img_gray_cnn.filter(ImageFilter.FIND_EDGES)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(edge_img, cmap="gray")
            ax.set_title("Edge Detection 필터 적용 결과")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
        with cnn_col2:
            st.markdown("**💡 CNN 모델링 관점의 분석 및 제언**")
            st.write("엣지 성분은 CNN의 하위 레이어가 학습하는 첫 핵심 피처입니다. 스크래치 등 선형 결함의 엣지가 두드러지게 검출되며, "
                     "엣지 신호가 밋밋한 결함들은 CNN 필터의 로컬 활성도를 높이기 위해 엣지 인핸스먼트 기법을 사전 오프라인 전처리로 결합해 줄 수 있습니다.")

        st.markdown("---")

        # 5) 라플라시안 노이즈 진단
        st.subheader("5️⃣ 라플라시안 필터(Laplacian Filter) 기반 고주파 질감 및 노이즈 진단")
        cnn_col1, cnn_col2 = st.columns([1, 2])
        with cnn_col1:
            enhanced_img = img_gray_cnn.filter(ImageFilter.EDGE_ENHANCE_MORE)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(enhanced_img, cmap="gray")
            ax.set_title("Edge Enhancement 필터 적용 결과")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
        with cnn_col2:
            st.markdown("**💡 CNN 모델링 관점의 분석 및 제언**")
            st.write("고주파 노이즈와 미세 실선 정보를 모사한 텍스처 분석입니다. 미세 질감이 강조되는 과정에서 철판 표면의 잡음도 노출됩니다. "
                     "CNN이 오진하지 않고 안정적인 형태 학습에 집중하도록 가우시안 블러링을 통해 잔진동 잡음을 억제하는 전처리 설계가 필요합니다.")

        st.markdown("---")

        # 6) 임계값 이진화
        st.subheader("6️⃣ 임계값 이진화(Adaptive Thresholding)를 통한 결함 마스크(Mask) 추출")
        cnn_col1, cnn_col2 = st.columns([1, 2])
        with cnn_col1:
            mean_threshold = np.mean(img_np_cnn)
            binary_img = (img_np_cnn > mean_threshold).astype(np.uint8) * 255
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(binary_img, cmap="binary")
            ax.set_title(f"평균치({mean_threshold:.1f}) 기준 이진화 이미지")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
        with cnn_col2:
            st.markdown("**💡 CNN 모델링 관점의 분석 및 제언**")
            st.write("이진화를 통한 배경 분리도 분석입니다. 배경에서 분리된 이진 마스크를 CNN의 보조 입력으로 사용하거나, "
                     "배경 노이즈의 강도를 임계처리하여 세그멘테이션 및 위치 회귀 손실을 가속화시키는 아키텍처 확장이 유용합니다.")

        st.markdown("---")

        # 7) 픽셀 프로파일링
        st.subheader("7️⃣ 픽셀 프로파일링(Pixel Profiling)을 통한 신호 대 잡음 격차(Contrast Gap) 분석")
        cnn_col1, cnn_col2 = st.columns([1, 2])
        with cnn_col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            row_profile = img_np_cnn[100, :]
            ax.plot(row_profile, color="red", linewidth=1.5)
            ax.axhline(np.mean(img_np_cnn), color="blue", linestyle="--", label="전체 평균")
            ax.set_title("가로 중앙 단면(Y=100)의 픽셀 강도 변화 라인차트")
            ax.set_xlabel("가로 픽셀 좌표 (X)")
            ax.set_ylabel("픽셀 밝기 값")
            ax.set_ylim(0, 255)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        with cnn_col2:
            st.markdown("**💡 CNN 모델링 관점의 분석 및 제언**")
            st.write("정상 배경 대비 결함 영역의 픽셀 값 급강하(Valley)를 보여주는 단면 라인차트입니다. "
                     "낙차가 깊을수록 CNN 합성곱 피처 추출 속도가 가속화됩니다. 만약 명암 격차가 좁고 희미한 결함인 경우에는 "
                     "대비 스트레칭(Contrast Stretching)을 통해 이 골짜기 폭을 늘려 가중치 전파를 활성화해야 합니다.")
