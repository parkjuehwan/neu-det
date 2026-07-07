# -*- coding: utf-8 -*-
"""
물리 디렉토리에 저장된 CNN 체크포인트를 복원하고, 추가 학습 단계 없이 테스트셋 결과 조회 및 
신규 이미지의 6종 결함 분류 추론과 XAI(Grad-CAM, Saliency, Feature Map) 3대 기법 시각화 결과를 제공하는 대시보드 페이지 모듈입니다.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
from model import SimpleCNN, load_checkpoint, GradCAM, generate_saliency, get_feature_maps
import platform

# Matplotlib 한글 폰트 설정 (다이어그램 한글 깨짐 방지)
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

def show_predict_page(df_images, df_bboxes, classes, IMAGES_DIR, CHECKPOINT_DIR, CLASS_COLORS):
    """
    학습 완료된 CNN 모델을 불러와 성능 지표를 검증하고, 신규 업로드된 이미지에 대해 
    실시간으로 결함 추론 및 XAI 3대 해석을 렌더링하는 함수입니다.

    인자:
        df_images (DataFrame): 이미지별 통계 메타데이터 데이터프레임
        df_bboxes (DataFrame): 결함 바운딩박스 영역 데이터프레임
        classes (list): 결함 고유 클래스 리스트
        IMAGES_DIR (str): 이미지 파일 로컬 디렉토리 경로
        CHECKPOINT_DIR (str): 모델 가중치 보관 디렉토리 경로
        CLASS_COLORS (dict): 클래스별 컬러 딕셔너리
    """
    st.header("🚀 학습 완료 모델 기반 예측 및 평가")
    st.markdown("디스크에 물리적으로 저장된 체크포인트 가중치를 로드하여 실시간 추론을 진단하고, 3대 XAI 기법으로 AI의 판단 근거를 시각화합니다. 이 화면에서는 추가 학습 단계를 거치지 않습니다.")

    # 1. 체크포인트 자동 복원 시도
    if not st.session_state.is_trained:
        temp_model = SimpleCNN(num_classes=len(classes))
        ckpt = load_checkpoint(temp_model, CHECKPOINT_DIR)
        if ckpt is not None:
            st.session_state.trained_model = temp_model
            st.session_state.is_trained = True
            st.session_state.eval_metrics = ckpt["eval_metrics"]
            st.session_state.test_predictions = ckpt["test_predictions"]
            st.session_state.is_ckpt_loaded = True

    # 2. 로드 실패 시 에러 출력 및 중단
    if not st.session_state.is_trained:
        st.warning("⚠️ **학습된 모델을 찾을 수 없습니다.**")
        st.info("이 페이지를 이용하시려면 먼저 '🧠 CNN 모델 학습 및 평가' 페이지에서 모델 학습을 완료하거나, 'src/checkpoints/' 디렉토리에 학습 가중치와 메타데이터 파일이 존재하는지 확인해 주세요.")
        st.stop()

    # 3. 모델 레이어 요약 표출
    st.subheader("📋 ImprovedCNN 모델 레이어 요약 정보")
    
    # 모델 파라미터 통계 계산
    t_model = st.session_state.trained_model
    total_params = sum(p.numel() for p in t_model.parameters())
    trainable_params = sum(p.numel() for p in t_model.parameters() if p.requires_grad)
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric("총 파라미터 수 (Total Parameters)", f"{total_params:,} 개")
    with stat_col2:
        st.metric("학습 가능한 파라미터 수", f"{trainable_params:,} 개")
    with stat_col3:
        st.metric("최종 출력 클래스 수 (Classes)", f"{len(classes)} 개 (6개 고정)")

    # 레이어 디테일 설명 테이블
    st.markdown("""
    | 레이어 단계 | 레이어 명칭 | 구성 요소 및 상세 설정 | 입력 형태 (Shape) | 출력 형태 (Shape) | 특징 및 역할 |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **입력** | Input Layer | Grayscale Image | (1, 64, 64) | (1, 64, 64) | 철판 표면 픽셀 정규화 데이터 |
    | **Block 1** | Conv Block 1 | Conv2d (32ch, 3x3) <br>+ BatchNorm2d <br>+ MaxPool2d (2x2) | (1, 64, 64) | (32, 32, 32) | 초기 저수준 형태학적 엣지 특징 추출 |
    | **Block 2** | Conv Block 2 | Conv2d (64ch, 3x3) <br>+ BatchNorm2d <br>+ MaxPool2d (2x2) | (32, 32, 32) | (64, 16, 16) | 텍스처 및 국소 결함 정보 집합화 |
    | **Block 3** | Conv Block 3 | Conv2d (128ch, 3x3) <br>+ BatchNorm2d <br>+ MaxPool2d (2x2) | (64, 16, 16) | (128, 8, 8) | 거시적 영역 결함 분포 특징 추출 (XAI 타겟) |
    | **Flatten** | Reshape | Tensor Flattening | (128, 8, 8) | (8192,) | 1차원 연산을 위한 전벡터화 단계 |
    | **Dense 1** | Fully-Connected | Linear (8192 -> 256) <br>+ Dropout (p=0.3) | (8192,) | (256,) | 고차원 추상 특징 학습 및 오버핏 방지 |
    | **Dense 2** | Output Layer | Linear (256 -> 6) | (256,) | (6,) | 6종 결함 클래스별 최종 Logits 분류 출력 |
    """)

    st.markdown("---")

    # 3. 탭 구성 (테스트 데이터셋 성능 vs 신규 이미지 업로드 분석)
    tab_eval, tab_upload = st.tabs(["📊 테스트 데이터셋 성능 및 XAI 진단", "📤 신규 이미지 업로드 실시간 결함 분석"])

    with tab_eval:
        # 4. 학습 모델 성능 요약 (5대 지표 & 혼동 행렬)
        st.subheader("📊 CNN 모델 테스트 평가 결과 (5대 지표)")
        
        metrics = st.session_state.eval_metrics
        
        # 5대 평가지표 카드 출력
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("1. 정확도 (Accuracy)", f"{metrics['accuracy'] * 100:.2f} %")
        with m_col2:
            st.metric("2. 가중 정밀도 (Precision)", f"{metrics['precision']:.4f}")
        with m_col3:
            st.metric("3. 가중 재현율 (Recall)", f"{metrics['recall']:.4f}")
        with m_col4:
            st.metric("4. 가중 F1-Score", f"{metrics['f1_score']:.4f}")
        with m_col5:
            st.metric("5. 테스트 손실 (Test Loss)", f"{metrics['test_loss']:.4f}")
            
        st.markdown("---")
        
        # 혼동 행렬 시각화
        st.subheader("📈 혼동 행렬 (Confusion Matrix)")
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(metrics['conf_matrix'], annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes, ax=ax_cm)
        ax_cm.set_title("예측 클래스 대 실제 클래스 교차 혼동행렬")
        ax_cm.set_xlabel("모델 예측 클래스")
        ax_cm.set_ylabel("실제 정답 클래스")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        st.markdown("---")

        # 5. 테스트 데이터 이미지별 결과 조회
        st.subheader("🔎 테스트 데이터 이미지별 예측 평가 및 오분류 검증")
        
        test_preds_df = st.session_state.test_predictions
        
        # 필터 추가 (성공만 보기 / 실패만 보기 / 전체 보기)
        st.markdown("### 1. 필터 설정")
        inf_filter_mode = st.radio("조회 모드:", ["전체", "예측 성공 (정답 일치)", "예측 실패 (오분류)"], key="inf_filter_mode")
        
        if inf_filter_mode == "예측 성공 (정답 일치)":
            filtered_test = test_preds_df[test_preds_df['is_correct'] == True]
        elif inf_filter_mode == "예측 실패 (오분류)":
            filtered_test = test_preds_df[test_preds_df['is_correct'] == False]
        else:
            filtered_test = test_preds_df
            
        st.markdown(f"조회된 테스트 이미지 수: **{len(filtered_test)}장** (전체 테스트 셋 크기: {len(test_preds_df)}장)")
        
        # 테스트 이미지 브라우저 페이징 처리 (30개 단위)
        t_items_per_page = 30
        t_total_items = len(filtered_test)
        t_total_pages = max((t_total_items - 1) // t_items_per_page + 1, 1)
        
        t_page_col1, t_page_col2 = st.columns([1, 4])
        with t_page_col1:
            inf_t_current_page = st.number_input("테스트 페이지 선택", min_value=1, max_value=t_total_pages, value=1, step=1, key="inf_t_page_select")
            
        t_start_idx = (inf_t_current_page - 1) * t_items_per_page
        t_end_idx = min(t_start_idx + t_items_per_page, t_total_items)
        
        t_page_images = filtered_test.iloc[t_start_idx:t_end_idx]
        
        # 30개 테스트 이미지 썸네일 그리드 표출 (6열 5행 구조)
        st.markdown("### 2. 테스트 이미지 목록 (배지를 확인하고 정밀 진단할 이미지를 클릭하세요)")
        
        t_cols_per_row = 6
        t_rows = (len(t_page_images) - 1) // t_cols_per_row + 1
        
        # 정밀 조회용 테스트 이미지 선택을 위한 세션 상태
        if 'inf_selected_test_image' not in st.session_state:
            st.session_state.inf_selected_test_image = test_preds_df['image_name'].iloc[0]
            
        for r in range(t_rows):
            t_row_cols = st.columns(t_cols_per_row)
            for c in range(t_cols_per_row):
                t_idx = r * t_cols_per_row + c
                if t_idx < len(t_page_images):
                    t_row = t_page_images.iloc[t_idx]
                    t_name = t_row['image_name']
                    t_path = os.path.join(IMAGES_DIR, t_name)
                    
                    try:
                        with Image.open(t_path) as thumb:
                            t_row_cols[c].image(thumb, use_container_width=True)
                    except Exception:
                        t_row_cols[c].write("이미지 로드 실패")
                        
                    # 성공 실패 여부 표시
                    if t_row['is_correct']:
                        t_row_cols[c].markdown("🟢 **정답 일치**")
                    else:
                        t_row_cols[c].markdown(f"🔴 **오분류 ({t_row['pred_label']})**")
                        
                    # 상세 보기 지정 버튼
                    if t_row_cols[c].button("🔍 결과 분석", key=f"inf_t_btn_{t_name}"):
                        st.session_state.inf_selected_test_image = t_name
                        
        st.markdown("---")
        
        # ==================== 테스트 개별 예측 검증 보드 ====================
        st.header(f"🔎 테스트 이미지 개별 예측 검증 보드: `{st.session_state.inf_selected_test_image}`")
        
        # 세션에 기록된 타겟 이미지가 현재 테스트 데이터프레임에 존재하는지 검증
        sel_test_name = st.session_state.inf_selected_test_image
        matching_rows = test_preds_df[test_preds_df['image_name'] == sel_test_name]
        
        # 재학습이나 필터 조건(성공/실패) 변경으로 인해 해당 이미지가 매칭되지 않는 경우 첫 번째 테스트 이미지로 초기화
        if matching_rows.empty:
            sel_test_name = test_preds_df['image_name'].iloc[0]
            st.session_state.inf_selected_test_image = sel_test_name
            matching_rows = test_preds_df[test_preds_df['image_name'] == sel_test_name]
            
        test_row_info = matching_rows.iloc[0]
        sel_img_path = os.path.join(IMAGES_DIR, sel_test_name)
        
        # 어노테이션 정보 오버레이 그리기
        try:
            test_orig_img = Image.open(sel_img_path).convert("RGB")
            test_draw_img = test_orig_img.copy()
            t_draw = ImageDraw.Draw(test_draw_img)
            
            sel_bboxes = df_bboxes[df_bboxes['image_name'] == sel_test_name]
            for _, bbox in sel_bboxes.iterrows():
                c_name = bbox['class_name']
                color = CLASS_COLORS.get(c_name, "#FFFFFF")
                t_draw.rectangle([bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']], outline=color, width=2)
                t_draw.text((bbox['xmin'] + 2, bbox['ymin'] + 2), c_name, fill=color)
        except Exception as e:
            st.error(f"테스트 이미지 로드 에러: {e}")
            st.stop()
            
        t_detail_col1, t_detail_col2 = st.columns([1, 1])
        
        with t_detail_col1:
            st.subheader("📸 실제 이미지 및 결함 박스 오버레이")
            st.image(test_draw_img, caption=f"실제 결함 종류: {test_row_info['target_label']}", use_container_width=True)
            
        with t_detail_col2:
            st.subheader("🤖 CNN 모델의 예측 결과 및 Softmax 확률 분포")
            
            # 예측 상태 리포팅 카드
            pred_status_col1, pred_status_col2 = st.columns(2)
            with pred_status_col1:
                st.markdown(f"**실제 정답**: `{test_row_info['target_label']}`")
                st.markdown(f"**모델 예측**: `{test_row_info['pred_label']}`")
            with pred_status_col2:
                if test_row_info['is_correct']:
                    st.success("🟢 최종 판정: 정답 일치")
                else:
                    st.error("🔴 최종 판정: 오분류 발생")
                    
            st.markdown("#### 클래스별 모델 예측 확률 분포 (%)")
            probs_pct = test_row_info['prob_dist'] * 100
            
            # 가로 막대 그래프로 Softmax 분포 그리기
            fig_prob, ax_prob = plt.subplots(figsize=(6, 4))
            y_pos = range(len(classes))
            bars = ax_prob.barh(y_pos, probs_pct, align='center', color='skyblue', edgecolor='black')
            ax_prob.set_yticks(y_pos)
            ax_prob.set_yticklabels(classes)
            ax_prob.invert_yaxis()  # 맨 위에 제일 먼저 나오게
            ax_prob.set_xlabel("예측 확률 (%)")
            ax_prob.set_xlim(0, 100)
            ax_prob.set_title("Softmax 예측 분포도")
            
            # 각 막대 끝에 수치 값 표출
            for bar in bars:
                width = bar.get_width()
                ax_prob.text(width + 2, bar.get_y() + bar.get_height()/2.0, f'{width:.1f}%', 
                             ha='left', va='center', fontsize=9, fontweight='bold')
                             
            st.pyplot(fig_prob)
            plt.close(fig_prob)
            
            # 정밀 픽셀 차트도 제공
            with st.expander("📝 해당 테스트 이미지의 국소 10x10 중앙 픽셀값 배열 보기"):
                test_gray = np.array(Image.open(sel_img_path).convert("L"))
                mid_y, mid_x = 100, 100
                test_sub = test_gray[mid_y-5:mid_y+5, mid_x-5:mid_x+5]
                test_sub_df = pd.DataFrame(test_sub, 
                                           columns=[f"X_{i}" for i in range(mid_x-5, mid_x+5)],
                                           index=[f"Y_{i}" for i in range(mid_y-5, mid_y+5)])
                st.dataframe(test_sub_df, use_container_width=True)

        # ==================== 설명 가능한 AI (XAI) 다중 기법 진단 보드 ====================
        st.markdown("---")
        st.subheader("💡 설명 가능한 AI (XAI) - 3대 기법 결합 진단")
        st.markdown("ImprovedCNN 모델의 판단 근거를 Grad-CAM, Saliency Map, Feature Map 3가지 기법으로 다각도 시각화합니다.")
        
        # 단일 이미지 텐서 준비
        img_gray_xai = Image.open(sel_img_path).convert("L").resize((64, 64))
        img_np_xai = np.array(img_gray_xai, dtype=np.float32) / 255.0
        img_tensor_xai = torch.tensor(img_np_xai).unsqueeze(0).unsqueeze(0) # (1, 1, 64, 64)
        
        # 3대 XAI 탭 생성
        inf_xai_tab1, inf_xai_tab2, inf_xai_tab3 = st.tabs([
            "1️⃣ Grad-CAM (최종 레이어 활성도)", 
            "2️⃣ Saliency Map (픽셀 민감도 진단)", 
            "3️⃣ Feature Map (하위 레이어 피처 추출 상태)"
        ])
        
        # 탭 1: Grad-CAM
        with inf_xai_tab1:
            try:
                # conv3 레이어를 타겟으로 훅 생성
                gradcam = GradCAM(t_model, t_model.conv3)
                
                # 예측한 인덱스를 가져와 맵 생성
                cam_map, target_class_idx = gradcam(img_tensor_xai, class_idx=test_row_info['pred_idx'])
                gradcam.release() # 훅 해제
                
                # cam_map 크기를 원본 이미지(200x200) 형태로 보간
                cam_img = Image.fromarray((cam_map * 255).astype(np.uint8)).resize((200, 200), Image.BILINEAR)
                cam_np_resized = np.array(cam_img) / 255.0
                
                x1_col1, x1_col2 = st.columns([1.2, 1.8])
                with x1_col1:
                    fig_xai, ax_xai = plt.subplots(figsize=(6, 6))
                    ax_xai.imshow(test_orig_img)
                    im_xai = ax_xai.imshow(cam_np_resized, cmap='jet', alpha=0.5, vmin=0.0, vmax=1.0)
                    ax_xai.axis('off')
                    ax_xai.set_title(f"Grad-CAM (클래스: {classes[target_class_idx]})")
                    st.pyplot(fig_xai)
                    plt.close(fig_xai)
                with x1_col2:
                    st.markdown(f"#### 🔎 Grad-CAM 활성화 영역 판독")
                    st.markdown(f"- **진단 클래스**: `{classes[target_class_idx]}`")
                    st.write("Grad-CAM은 딥러닝 모델의 최종합성곱 레이어가 결함의 대략적인 형태와 범주를 파악하기 위해 주목한 넓은 수용장(Receptive Field) 영역을 표시합니다. "
                             "빨간색/노란색 영역은 모델이 분류 판단에 가장 가중치를 많이 실은 부위이며, 파란색은 분류 과정에서 무시한 무영향 배경 영역입니다. 결함이 주로 발생하는 철판의 거시적 균열 및 판형 파손부와 히트맵 핫스팟이 잘 매칭되는지 입증해 줍니다.")
            except Exception as e_gc:
                st.error(f"Grad-CAM 렌더링 중 오류: {e_gc}")
                
        # 탭 2: Saliency Map
        with inf_xai_tab2:
            try:
                # Saliency 계산 (Requires grad)
                saliency_map = generate_saliency(t_model, img_tensor_xai.clone(), target_class_idx=test_row_info['pred_idx'])
                
                # 원본 이미지(200x200) 형태로 보간
                sal_img = Image.fromarray((saliency_map * 255).astype(np.uint8)).resize((200, 200), Image.BILINEAR)
                sal_np_resized = np.array(sal_img) / 255.0
                
                x2_col1, x2_col2 = st.columns([1.2, 1.8])
                with x2_col1:
                    fig_sal, ax_sal = plt.subplots(figsize=(6, 6))
                    ax_sal.imshow(sal_np_resized, cmap='hot')
                    ax_sal.axis('off')
                    ax_sal.set_title("Saliency Map (픽셀 민감도)")
                    st.pyplot(fig_sal)
                    plt.close(fig_sal)
                with x2_col2:
                    st.markdown("#### 🔎 Saliency Map 픽셀 기여 판독")
                    st.write("Saliency Map은 모델 예측 결과에 대해 **각 개별 픽셀이 미치는 민감도(Gradient의 절대값)**를 역추적하여 픽셀 해상도로 보여줍니다. "
                             "히트맵에서 하얗고 빨갛게 빛나는 얇은 선이나 점들이 실제 결함의 아주 미세한 외곽선 경계(Scratch 엣지) 혹은 미세 핏(Pit) 점자국 영역과 정확히 겹치는 것을 볼 수 있습니다. "
                             "이는 거시적 영역만 보던 Grad-CAM에 비해 픽셀 한 조각 단위로 모델이 경계 엣지에 극도로 예민하게 반응하여 결함을 잡아내고 있음을 과학적으로 설명해 줍니다.")
            except Exception as e_sal:
                st.error(f"Saliency Map 렌더링 중 오류: {e_sal}")
                
        # 탭 3: Feature Map Activations
        with inf_xai_tab3:
            try:
                # conv1의 32채널 피처맵 수집
                fmaps = get_feature_maps(t_model, img_tensor_xai.clone()) # (32, 64, 64)
                
                st.markdown("#### 🔎 첫 번째 합성곱층(conv1) 32채널 피처 맵 활성화 상태")
                st.write("하위 합성곱 필터들이 원본 철강 표면에서 어떠한 형태학적 특징들을 전처리 추출하여 상위 레이어로 전달하는지 보여줍니다. "
                         "필터에 따라 수평선 엣지를 감지하거나, 반대로 배경을 날리고 어두운 점자국만 부각하는 등 각기 다른 수학적 마스크로 이미지 특징을 수집하고 있음을 입증합니다.")
                
                # 4행 8열 형태로 32개 피처 맵 그리드 생성
                fig_fmap, axes = plt.subplots(4, 8, figsize=(12, 6))
                for i in range(32):
                    ax = axes[i // 8, i % 8]
                    ax.imshow(fmaps[i], cmap='gray')
                    ax.axis('off')
                    ax.set_title(f"Ch {i}", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_fmap)
                plt.close(fig_fmap)
            except Exception as e_fm:
                st.error(f"Feature Map 렌더링 중 오류: {e_fm}")

    with tab_upload:
        st.subheader("📤 신규 철강 표면 이미지 업로드 및 실시간 결함 분석")
        st.markdown("로컬 디렉토리에서 분석할 철강 표면 이미지 파일(.jpg, .jpeg, .png)을 업로드하세요. 즉시 ImprovedCNN 모델이 6종 결함을 분류하고 판단 근거(XAI 3대 기법)를 실시간으로 해석합니다.")
        
        # 파일 업로더 생성
        up_file = st.file_uploader("철강 표면 이미지 파일 선택", type=["jpg", "jpeg", "png"], key="inf_file_uploader")
        
        if up_file is not None:
            try:
                up_img_orig = Image.open(up_file).convert("RGB")
                
                # 예측을 위한 전처리 (그레이스케일 변환, 64x64 리사이즈, 정규화 텐서 변환)
                up_img_gray = up_img_orig.convert("L").resize((64, 64))
                up_img_np = np.array(up_img_gray, dtype=np.float32) / 255.0
                up_img_tensor = torch.tensor(up_img_np).unsqueeze(0).unsqueeze(0) # (1, 1, 64, 64)
                
                # 모델 추론 진행
                t_model.eval()
                # Saliency 계산용 gradients 추적을 위해 requires_grad 활성화 가능한 텐서 복제 생성
                up_img_tensor_grad = up_img_tensor.clone()
                
                with torch.no_grad():
                    up_outputs = t_model(up_img_tensor)
                    up_probs = F.softmax(up_outputs, dim=1).cpu().numpy()[0]
                    
                up_pred_idx = np.argmax(up_probs)
                up_pred_label = classes[up_pred_idx]
                up_pred_pct = up_probs[up_pred_idx] * 100
                
                # ----------------- 결함 Bounding Box XML 검사 및 파싱 시작 -----------------
                # 업로드된 파일명에서 확장자를 제거하고 xml 매핑 파일 경로 확인
                up_base_name = os.path.splitext(up_file.name)[0]
                up_xml_path = os.path.join(BASE_DIR, "data", "NEU-DET", "ANNOTATIONS", up_base_name + ".xml")
                
                up_bboxes = []
                up_draw_img = up_img_orig.copy()
                
                # XML 파일이 실제로 존재하면 파싱 수행
                if os.path.exists(up_xml_path):
                    import xml.etree.ElementTree as ET
                    try:
                        tree_up = ET.parse(up_xml_path)
                        root_up = tree_up.getroot()
                        
                        draw_ctx = ImageDraw.Draw(up_draw_img)
                        for obj in root_up.findall('object'):
                            c_name = obj.find('name').text
                            bndbox = obj.find('bndbox')
                            xmin = float(bndbox.find('xmin').text)
                            ymin = float(bndbox.find('ymin').text)
                            xmax = float(bndbox.find('xmax').text)
                            ymax = float(bndbox.find('ymax').text)
                            
                            w_val = xmax - xmin
                            h_val = ymax - ymin
                            area_val = w_val * h_val
                            
                            up_bboxes.append({
                                "결함 종류": c_name,
                                "시작 X (xmin)": xmin,
                                "시작 Y (ymin)": ymin,
                                "너비 (width)": w_val,
                                "높이 (height)": h_val,
                                "면적 (area, px²)": area_val
                            })
                            
                            # 클래스별 매핑된 컬러로 사각형 테두리와 이름 드로잉
                            color_hex = CLASS_COLORS.get(c_name, "#FFFFFF")
                            draw_ctx.rectangle([xmin, ymin, xmax, ymax], outline=color_hex, width=3)
                            draw_ctx.text((xmin + 2, ymin + 2), c_name, fill=color_hex)
                    except Exception as e_xml:
                        st.warning(f"업로드 이미지의 XML 어노테이션을 파싱하는 도중 에러가 발생했습니다: {e_xml}")
                # ----------------- 결함 Bounding Box XML 검사 및 파싱 끝 -----------------
                
                st.markdown("---")
                
                # 1. 추론 결과 리포트 영역 (2열 배치)
                up_col1, up_col2 = st.columns([1, 1])
                
                with up_col1:
                    if len(up_bboxes) > 0:
                        st.subheader("📸 Ground Truth 결함 박스 시각화")
                        st.image(up_draw_img, caption=f"결함 영역이 오버레이된 이미지 (파일명: {up_file.name})", use_container_width=True)
                    else:
                        st.subheader("📸 업로드된 원본 이미지")
                        st.image(up_img_orig, caption=f"매핑된 XML 결함 데이터가 없습니다 (파일명: {up_file.name})", use_container_width=True)
                    
                with up_col2:
                    st.subheader("🤖 실시간 결함 판정 결과")
                    st.info(f"💡 결함 판정 결과: **{up_pred_label}** (신뢰도: **{up_pred_pct:.2f}%**)")
                    
                    # Softmax 확률 분포 바 차트 그리기
                    fig_up_prob, ax_up_prob = plt.subplots(figsize=(6, 4))
                    y_pos_up = range(len(classes))
                    bars_up = ax_up_prob.barh(y_pos_up, up_probs * 100, align='center', color='skyblue', edgecolor='black')
                    ax_up_prob.set_yticks(y_pos_up)
                    ax_up_prob.set_yticklabels(classes)
                    ax_up_prob.invert_yaxis()
                    ax_up_prob.set_xlabel("예측 확률 (%)")
                    ax_up_prob.set_xlim(0, 100)
                    ax_up_prob.set_title("예측 확률 상세 분포도")
                    
                    # 막대 우측에 백분율 표기
                    for bar in bars_up:
                        width = bar.get_width()
                        ax_up_prob.text(width + 2, bar.get_y() + bar.get_height()/2.0, f'{width:.1f}%', 
                                     ha='left', va='center', fontsize=9, fontweight='bold')
                                     
                    st.pyplot(fig_up_prob)
                    plt.close(fig_up_prob)
                    
                # ----------------- 결함 상세 데이터 테이블 추가 시작 -----------------
                if len(up_bboxes) > 0:
                    st.subheader("📊 검출된 결함 객체 스펙 테이블 및 요약 리포트")
                    df_up_box = pd.DataFrame(up_bboxes)
                    st.dataframe(df_up_box, use_container_width=True)
                    
                    total_area_ratio = df_up_box["면적 (area, px²)"].sum() / (up_img_orig.width * up_img_orig.height) * 100
                    st.success(f"검출 요약: 업로드된 이미지에는 총 **{len(up_bboxes)}개**의 실제 결함 객체가 정의되어 있으며, "
                               f"이미지 내 전체 면적 대비 결함 총 크기 비율은 **{total_area_ratio:.2f}%**입니다.")
                # ----------------- 결함 상세 데이터 테이블 추가 끝 -----------------

                
                # 2. XAI 다각도 해석 패널
                st.markdown("---")
                st.subheader("💡 실시간 설명 가능한 AI (XAI) 진단 보드")
                st.markdown("업로드된 이미지의 특징 중 모델이 특정 결함으로 판단하게 만든 결정적 활성 영역을 3대 해석학 기법으로 보여줍니다.")
                
                up_xai_tab1, up_xai_tab2, up_xai_tab3 = st.tabs([
                    "1️⃣ Grad-CAM (최종 레이어 활성도)", 
                    "2️⃣ Saliency Map (픽셀 민감도 진단)", 
                    "3️⃣ Feature Map (하위 레이어 피처 추출 상태)"
                ])
                
                # XAI 탭 1: Grad-CAM
                with up_xai_tab1:
                    try:
                        up_gradcam = GradCAM(t_model, t_model.conv3)
                        up_cam_map, _ = up_gradcam(up_img_tensor_grad, class_idx=up_pred_idx)
                        up_gradcam.release() # 메모리 훅 제거 필수
                        
                        # 히트맵 원본 해상도 매핑 보간
                        up_cam_img = Image.fromarray((up_cam_map * 255).astype(np.uint8)).resize((up_img_orig.width, up_img_orig.height), Image.BILINEAR)
                        up_cam_np_resized = np.array(up_cam_img) / 255.0
                        
                        ux_col1, ux_col2 = st.columns([1.2, 1.8])
                        with ux_col1:
                            fig_up_cam, ax_up_cam = plt.subplots(figsize=(6, 6))
                            ax_up_cam.imshow(up_img_orig)
                            ax_up_cam.imshow(up_cam_np_resized, cmap='jet', alpha=0.5, vmin=0.0, vmax=1.0)
                            ax_up_cam.axis('off')
                            ax_up_cam.set_title(f"Grad-CAM (판단 타겟: {up_pred_label})")
                            st.pyplot(fig_up_cam)
                            plt.close(fig_up_cam)
                        with ux_col2:
                            st.markdown("#### 🔎 실시간 Grad-CAM 분석 결과")
                            st.markdown(f"- **판정된 결함 종류**: `{up_pred_label}`")
                            st.write("해당 업로드 이미지 상에서 **빨갛게 표시된 부위**는 ImprovedCNN 모델이 최종 합성곱 레이어(conv3)에서 결함 클래스 특성을 분류 결정하기 위해 가장 강력한 거시적 가중치를 부여해 관찰한 공간적 위치입니다. "
                                     "결함의 주된 질감 분포와 기하학적 수용 영역이 히트맵 분포도와 조화롭게 매칭됨을 입증해 줍니다.")
                    except Exception as e_up_gc:
                        st.error(f"실시간 Grad-CAM 연산 중 에러: {e_up_gc}")
                        
                # XAI 탭 2: Saliency Map
                with up_xai_tab2:
                    try:
                        up_saliency_map = generate_saliency(t_model, up_img_tensor_grad, target_class_idx=up_pred_idx)
                        
                        # 히트맵 원본 해상도 매핑 보간
                        up_sal_img = Image.fromarray((up_saliency_map * 255).astype(np.uint8)).resize((up_img_orig.width, up_img_orig.height), Image.BILINEAR)
                        up_sal_np_resized = np.array(up_sal_img) / 255.0
                        
                        us_col1, us_col2 = st.columns([1.2, 1.8])
                        with us_col1:
                            fig_up_sal, ax_up_sal = plt.subplots(figsize=(6, 6))
                            ax_up_sal.imshow(up_sal_np_resized, cmap='hot')
                            ax_up_sal.axis('off')
                            ax_up_sal.set_title("Saliency Map (픽셀 민감도)")
                            st.pyplot(fig_up_sal)
                            plt.close(fig_up_sal)
                        with us_col2:
                            st.markdown("#### 🔎 실시간 Saliency Map 분석 결과")
                            st.write("Saliency Map은 모델 출력에 대해 이미지 내 **각 개별 픽셀의 미세 변화에 따른 영향도(기울기 절대값)**를 표시합니다. "
                                     "빛이 나듯이 하얗고 붉은 점 및 실선 모양 핫스팟이 보이는 경계들은 모델이 미세한 스크래치 엣지나 점자국을 감지하는 핵심 포인트입니다. "
                                     "배경이 아닌 실제 표면 결함 요철의 가장자리 선조에 픽셀별 기울기 기여도가 높은 정합성을 띰을 과학교육적으로 입증합니다.")
                    except Exception as e_up_sal:
                        st.error(f"실시간 Saliency Map 연산 중 에러: {e_up_sal}")
                        
                # XAI 탭 3: Feature Map Activations
                with up_xai_tab3:
                    try:
                        up_fmaps = get_feature_maps(t_model, up_img_tensor.clone()) # (32, 64, 64)
                        
                        st.markdown("#### 🔎 첫 번째 합성곱층(conv1) 32채널 피처 맵 실시간 활성화 상태")
                        st.write("하위 층의 다양한 공간 마스크 필터(32ch)가 원본 업로드 이미지에서 어떠한 형상학적 모서리, 명도 대비 특징을 수집하여 고차원 학습 층으로 전달하는지 시각적 대조를 증명합니다.")
                        
                        fig_up_fmap, axes_up = plt.subplots(4, 8, figsize=(12, 6))
                        for i in range(32):
                            ax = axes_up[i // 8, i % 8]
                            ax.imshow(up_fmaps[i], cmap='gray')
                            ax.axis('off')
                            ax.set_title(f"Ch {i}", fontsize=8)
                        plt.tight_layout()
                        st.pyplot(fig_up_fmap)
                        plt.close(fig_up_fmap)
                    except Exception as e_up_fm:
                        st.error(f"실시간 Feature Map 추출 중 에러: {e_up_fm}")
            except Exception as e_up_all:
                st.error(f"업로드 이미지 로드 및 분석 중 알 수 없는 에러가 발생했습니다: {e_up_all}")
        else:
            st.info("💡 위 업로더 상자에 결함을 검사할 철판 표면 이미지를 올려주세요.")
