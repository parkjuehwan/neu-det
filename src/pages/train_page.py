# -*- coding: utf-8 -*-
"""
자체 구현된 CNN 모델(SimpleCNN)의 실시간 훈련 제어, 검증 곡선 렌더링, 5대 평가 지표 분석, 
혼동 행렬 시각화 및 3대 XAI(Grad-CAM, Saliency, Feature Map) 기법을 통한 오분류 정밀 분석 대시보드 페이지 모듈입니다.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from model import SimpleCNN, NEUDataset, evaluate_model, save_checkpoint, GradCAM, generate_saliency, get_feature_maps
import platform

# Matplotlib 한글 폰트 설정 (다이어그램 한글 깨짐 방지)
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

def show_train_page(df_images, df_bboxes, classes, IMAGES_DIR, CHECKPOINT_DIR, CLASS_COLORS):
    """
    CNN 모델의 실시간 학습, 평가 지표 분석 및 개별 테스트 이미지의 XAI 예측 분석 보드를 렌더링하는 함수입니다.

    인자:
        df_images (DataFrame): 이미지 통계 메타데이터
        df_bboxes (DataFrame): 바운딩 박스 정보 데이터프레임
        classes (list): 결함 고유 클래스 리스트
        IMAGES_DIR (str): 이미지 파일 디렉토리
        CHECKPOINT_DIR (str): 모델 가중치 보관 디렉토리
        CLASS_COLORS (dict): 클래스별 색상 딕셔너리
    """
    st.header("🧠 CNN 모델 실시간 학습 및 다각도 평가")
    st.markdown("기본 CNN 모델을 정의하여 실시간으로 훈련을 진행하고, 다각도 평가지표(5대 지표 및 3종 Mermaid 다이어그램)를 통해 성능을 검증합니다.")

    # 탭 구성
    tab_train, tab_test_panel = st.tabs(["🚀 모델 학습 및 평가지표 검증", "🔎 테스트 데이터 이미지별 결과 조회"])

    # 2.1 모델 학습 및 평가지표 검증
    with tab_train:
        # 이전에 디스크에서 저장된 가중치를 읽어 자동 복구에 성공한 경우 안내 카드 출력
        if 'is_ckpt_loaded' in st.session_state and st.session_state.is_ckpt_loaded:
            st.success("💾 **로컬 체크포인트 복원 완료**: 디스크(`src/checkpoints/`)에 저장되어 있던 이전 훈련 모델(ImprovedCNN) 및 성능 지표 기록을 자동으로 탐지하여 복원했습니다. 다시 새로 학습하지 않고 우측 탭이나 개별 이미지 조회 메뉴에서 바로 예측 결과를 확인할 수 있습니다.")

        # 모델 훈련 세부 파라미터 제어
        st.subheader("⚙️ 모델 훈련 설정 및 제어")
        
        config_col1, config_col2, config_col3, config_col4 = st.columns(4)
        with config_col1:
            epochs = st.slider("학습 Epoch 수", min_value=1, max_value=10, value=3, step=1)
        with config_col2:
            batch_size = st.selectbox("배치 크기 (Batch Size)", [8, 16, 32, 64], index=1)
        with config_col3:
            learning_rate = st.selectbox("학습률 (Learning Rate)", [0.01, 0.005, 0.001, 0.0001], index=2)
        with config_col4:
            sampling_ratio = st.slider("훈련 데이터 샘플링 비율 (%)", min_value=10, max_value=100, value=20, step=5)
            
        st.caption("🚨 리소스 보호를 위해 기본적으로 훈련 데이터의 20%만 샘플링하여 가벼운 CNN 모델 학습을 진행하도록 추천 설정되어 있습니다.")
        
        # 학습 시작 버튼
        btn_train = st.button("🚀 CNN 모델 학습 시작", use_container_width=True)
        
        # 실시간 진행 그래프 컨테이너
        graph_placeholder = st.empty()
        
        if btn_train:
            # 1. 데이터 샘플링 및 분할
            st.info("데이터를 준비하고 분할하는 중...")
            
            # 클래스별로 균등하게 20% 샘플링
            sampled_df_list = []
            for c in classes:
                c_df = df_images[df_images['class_prefix'] == c]
                if len(c_df) > 0:
                    sample_count = max(int(len(c_df) * (sampling_ratio / 100.0)), 5)
                    sample_count = min(sample_count, len(c_df))
                    sampled_df_list.append(c_df.sample(n=sample_count, random_state=42))
                
            df_sampled = pd.concat(sampled_df_list).sample(frac=1.0, random_state=42).reset_index(drop=True)
            
            # 훈련 / 검증 / 테스트 분할 (70% / 15% / 15%)
            n_total = len(df_sampled)
            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.15)
            
            train_df = df_sampled.iloc[:n_train].reset_index(drop=True)
            val_df = df_sampled.iloc[n_train:n_train+n_val].reset_index(drop=True)
            test_df = df_sampled.iloc[n_train+n_val:].reset_index(drop=True)
            
            # PyTorch 데이터셋 & 로더 생성 (Images 디렉토리 전달 필수)
            train_dataset = NEUDataset(train_df['image_name'].values, train_df['class_prefix'].values, classes, IMAGES_DIR)
            val_dataset = NEUDataset(val_df['image_name'].values, val_df['class_prefix'].values, classes, IMAGES_DIR)
            test_dataset = NEUDataset(test_df['image_name'].values, test_df['class_prefix'].values, classes, IMAGES_DIR)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            st.success(f"데이터셋 분할 완료! (훈련셋: {len(train_dataset)}장, 검증셋: {len(val_dataset)}장, 테스트셋: {len(test_dataset)}장)")
            
            # 모델 및 최적화 설정
            model = SimpleCNN(num_classes=len(classes))
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # 실시간 로그 저장을 위한 리스트
            train_losses, train_accs = [], []
            val_losses, val_accs = [], []
            
            progress_bar = st.progress(0)
            
            # 훈련 루프
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for images, labels, _ in train_loader:
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                epoch_loss = running_loss / len(train_dataset)
                epoch_acc = correct / total
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
                
                # 검증 루프
                model.eval()
                val_running_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for val_images, val_labels, _ in val_loader:
                        val_outputs = model(val_images)
                        val_loss = criterion(val_outputs, val_labels)
                        
                        val_running_loss += val_loss.item() * val_images.size(0)
                        _, val_predicted = torch.max(val_outputs.data, 1)
                        val_total += val_labels.size(0)
                        val_correct += (val_predicted == val_labels).sum().item()
                        
                val_epoch_loss = val_running_loss / len(val_dataset)
                val_epoch_acc = val_correct / val_total
                val_losses.append(val_epoch_loss)
                val_accs.append(val_epoch_acc)
                
                # 실시간 프로그레스 바 업데이트
                progress_bar.progress((epoch + 1) / epochs)
                
                # 실시간 차트 업데이트
                with graph_placeholder.container():
                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        # 손실 곡선 그리기
                        fig_l, ax_l = plt.subplots(figsize=(6, 4))
                        ax_l.plot(range(1, epoch + 2), train_losses, label="Train Loss", marker='o', color='blue')
                        ax_l.plot(range(1, epoch + 2), val_losses, label="Val Loss", marker='s', color='orange')
                        ax_l.set_xlabel("Epoch")
                        ax_l.set_ylabel("Loss")
                        ax_l.legend()
                        ax_l.set_title("Epoch별 Loss 곡선")
                        st.pyplot(fig_l)
                        plt.close(fig_l)
                    with chart_col2:
                        # 정확도 곡선 그리기
                        fig_a, ax_a = plt.subplots(figsize=(6, 4))
                        ax_a.plot(range(1, epoch + 2), train_accs, label="Train Accuracy", marker='o', color='green')
                        ax_a.plot(range(1, epoch + 2), val_accs, label="Val Accuracy", marker='s', color='red')
                        ax_a.set_xlabel("Epoch")
                        ax_a.set_ylabel("Accuracy")
                        ax_a.legend()
                        ax_a.set_title("Epoch별 Accuracy 곡선")
                        st.pyplot(fig_a)
                        plt.close(fig_a)
                        
            # 테스트 데이터 최종 평가 (model.py에서 분리된 평가 유틸리티 호출)
            st.info("테스트 데이터셋에 대해 최종 성능 평가를 수행하는 중...")
            eval_results = evaluate_model(model, test_loader, criterion, classes)
            
            # 세션 상태에 훈련된 모델 객체 저장 (XAI Grad-CAM 계산용)
            st.session_state.trained_model = model
            st.session_state.is_trained = True
            st.session_state.train_history = {
                "train_losses": train_losses,
                "train_accs": train_accs,
                "val_losses": val_losses,
                "val_accs": val_accs
            }
            st.session_state.eval_metrics = {
                "accuracy": eval_results["accuracy"],
                "precision": eval_results["precision"],
                "recall": eval_results["recall"],
                "f1_score": eval_results["f1_score"],
                "test_loss": eval_results["test_loss"],
                "conf_matrix": eval_results["conf_matrix"]
            }
            
            # 개별 이미지 예측 정밀 데이터 저장
            st.session_state.test_predictions = pd.DataFrame(eval_results["predictions_df_data"])
            
            # 디스크 체크포인트 자동 저장 연동 (매번 학습하지 않고 이어서 예측할 수 있게 처리)
            save_checkpoint(model, CHECKPOINT_DIR, st.session_state.eval_metrics, st.session_state.test_predictions)
            st.session_state.is_ckpt_loaded = False  # 새로 학습을 시킨 것이므로 복원 상태는 해제
            
            st.success("모델 학습 및 테스트셋 최종 평가가 모두 완료되었으며, 학습 결과 체크포인트가 src/checkpoints/에 안전하게 저장되었습니다!")
            
        # 학습 결과 보고서 출력
        if st.session_state.is_trained:
            st.markdown("---")
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
        else:
            st.warning("아직 학습된 CNN 모델이 없습니다. 위의 'CNN 모델 학습 시작' 버튼을 눌러 모델을 생성하고 검증을 진행해 주세요.")

    # 2.2 테스트 데이터 이미지별 결과 조회
    with tab_test_panel:
        st.subheader("🔎 테스트 데이터 이미지별 예측 평가 및 오분류 검증")
        
        if not st.session_state.is_trained:
            st.warning("먼저 '모델 학습 및 평가지표 검증' 탭에서 CNN 모델을 학습시켜야 테스트 셋에 대한 정밀 검증을 수행할 수 있습니다.")
        else:
            test_preds_df = st.session_state.test_predictions
            
            # 필터 추가 (성공만 보기 / 실패만 보기 / 전체 보기)
            st.markdown("### 1. 필터 설정")
            filter_mode = st.radio("조회 모드:", ["전체", "예측 성공 (정답 일치)", "예측 실패 (오분류)"])
            
            if filter_mode == "예측 성공 (정답 일치)":
                filtered_test = test_preds_df[test_preds_df['is_correct'] == True]
            elif filter_mode == "예측 실패 (오분류)":
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
                t_current_page = st.number_input("테스트 페이지 선택", min_value=1, max_value=t_total_pages, value=1, step=1, key="t_page_select")
                
            t_start_idx = (t_current_page - 1) * t_items_per_page
            t_end_idx = min(t_start_idx + t_items_per_page, t_total_items)
            
            t_page_images = filtered_test.iloc[t_start_idx:t_end_idx]
            
            # 30개 테스트 이미지 썸네일 그리드 표출 (6열 5행 구조)
            st.markdown("### 2. 테스트 이미지 목록 (배지를 확인하고 정밀 진단할 이미지를 클릭하세요)")
            
            t_cols_per_row = 6
            t_rows = (len(t_page_images) - 1) // t_cols_per_row + 1
            
            # 정밀 조회용 테스트 이미지 선택을 위한 세션 상태
            if 'selected_test_image' not in st.session_state:
                st.session_state.selected_test_image = test_preds_df['image_name'].iloc[0]
                
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
                        if t_row_cols[c].button("🔍 결과 분석", key=f"t_btn_{t_name}"):
                            st.session_state.selected_test_image = t_name
                            
            st.markdown("---")
            
            # ==================== 테스트 개별 예측 검증 보드 ====================
            st.header(f"🔎 테스트 이미지 개별 예측 검증 보드: `{st.session_state.selected_test_image}`")
            
            # 세션에 기록된 타겟 이미지가 현재 테스트 데이터프레임에 존재하는지 검증
            sel_test_name = st.session_state.selected_test_image
            matching_rows = test_preds_df[test_preds_df['image_name'] == sel_test_name]
            
            # 재학습이나 필터 조건(성공/실패) 변경으로 인해 해당 이미지가 매칭되지 않는 경우 첫 번째 테스트 이미지로 초기화
            if matching_rows.empty:
                sel_test_name = test_preds_df['image_name'].iloc[0]
                st.session_state.selected_test_image = sel_test_name
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
            
            # 훈련된 모델 객체가 세션에 있는지 확인
            if 'trained_model' in st.session_state:
                # 단일 이미지 텐서 준비
                img_gray_xai = Image.open(sel_img_path).convert("L").resize((64, 64))
                img_np_xai = np.array(img_gray_xai, dtype=np.float32) / 255.0
                img_tensor_xai = torch.tensor(img_np_xai).unsqueeze(0).unsqueeze(0) # (1, 1, 64, 64)
                
                # 3대 XAI 탭 생성
                xai_tab1, xai_tab2, xai_tab3 = st.tabs([
                    "1️⃣ Grad-CAM (최종 레이어 활성도)", 
                    "2️⃣ Saliency Map (픽셀 민감도 진단)", 
                    "3️⃣ Feature Map (하위 레이어 피처 추출 상태)"
                ])
                
                # 탭 1: Grad-CAM
                with xai_tab1:
                    try:
                        t_model = st.session_state.trained_model
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
                with xai_tab2:
                    try:
                        t_model = st.session_state.trained_model
                        # Saliency 계산 (Requires grad)
                        saliency_map = generate_saliency(t_model, img_tensor_xai.clone(), target_class_idx=test_row_info['pred_idx'])
                        
                        # 원본 이미지(200x200) 형태로 보간
                        sal_img = Image.fromarray((saliency_map * 255).astype(np.uint8)).resize((200, 200), Image.BILINEAR)
                        sal_np_resized = np.array(sal_img) / 255.0
                        
                        x2_col1, x2_col2 = st.columns([1.2, 1.8])
                        with x2_col1:
                            fig_sal, ax_sal = plt.subplots(figsize=(6, 6))
                            # Saliency Map은 픽셀 기여도를 hot 스케일로 출력
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
                with xai_tab3:
                    try:
                        t_model = st.session_state.trained_model
                        # conv1의 32채널 피처맵 수집
                        fmaps = get_feature_maps(t_model, img_tensor_xai.clone()) # (32, 64, 64)
                        
                        st.markdown("#### 🔎 첫 번째 합성곱층(conv1) 32채널 피처 맵 활성화 상태")
                        st.write("하위 합성곱 필터들이 원본 철강 표면에서 어떠한 형태학적 특징들을 전처리 추출하여 상위 레이어로 전달하는지 보여줍니다. "
                                 "필터에 따라 수평선 엣지를 감지하거나, 반대로 배경을 날리고 어두운 점자국만 부각하는 등 각기 다른 수학적 마스크로 이미지 특징을 수집하고 있음을 입증합니다.")
                        
                        # 4행 8열 형태로 32개 피처 맵 그리드 생성
                        fig_fmap, axes = plt.subplots(4, 8, figsize=(12, 6))
                        for i in range(32):
                            ax = axes[i // 8, i % 8]
                            # 각 피처 맵 시각화 (gray)
                            ax.imshow(fmaps[i], cmap='gray')
                            ax.axis('off')
                            ax.set_title(f"Ch {i}", fontsize=8)
                        plt.tight_layout()
                        st.pyplot(fig_fmap)
                        plt.close(fig_fmap)
                    except Exception as e_fm:
                        st.error(f"Feature Map 렌더링 중 오류: {e_fm}")
            else:
                st.info("💡 XAI 다각도 진단을 확인하려면 먼저 '모델 학습 및 평가지표 검증' 탭에서 학습을 정상 완료하거나 로컬 모델을 불러와야 합니다.")
