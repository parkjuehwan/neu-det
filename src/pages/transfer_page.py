# -*- coding: utf-8 -*-
"""
Hugging Face의 사전학습 경량 모델(MobileNetV2)을 로드 및 동결 전이학습시키고, 디스크에 가중치를 보관 및 복원하며, 
자체 CNN과의 평가지표 비교 분석 및 신규 이미지의 결함 탐지 예측 기능을 수행하는 대시보드 페이지 모듈입니다.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import platform

# Matplotlib 한글 폰트 설정 (다이어그램 한글 깨짐 방지)
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

from model import (TransferDataset, get_transfer_model, load_transfer_checkpoint, 
                   save_transfer_checkpoint, evaluate_transfer_model)

def render_transfer_metrics_report(tr_metrics, classes):
    """
    전이학습 모델의 5대 핵심 지표(Accuracy, Precision, Recall, F1-Score, Test Loss) 및
    클래스별 상세 성능 스펙 테이블과 한글 해석 리포트를 렌더링합니다.
    
    인자:
        tr_metrics (dict): 전이학습 성능 지표 사전
        classes (list): 결함 고유 클래스 리스트
    """
    st.markdown("##### 📱 MobileNetV2 전이학습 모델 5대 평가 지표")
    mc_col1, mc_col2, mc_col3, mc_col4, mc_col5 = st.columns(5)
    with mc_col1:
        st.metric("1. 정확도 (Accuracy)", f"{tr_metrics['accuracy'] * 100:.2f} %")
    with mc_col2:
        st.metric("2. 가중 정밀도 (Precision)", f"{tr_metrics['precision']:.4f}")
    with mc_col3:
        st.metric("3. 가중 재현율 (Recall)", f"{tr_metrics['recall']:.4f}")
    with mc_col4:
        st.metric("4. 가중 F1-Score", f"{tr_metrics['f1_score']:.4f}")
    with mc_col5:
        st.metric("5. 테스트 손실 (Test Loss)", f"{tr_metrics['test_loss']:.4f}")

    st.markdown("---")
    st.subheader("📊 전이학습 모델의 5대 평가지표 종합 진단 리포트")
    
    desc_metrics = f"""
    본 전이학습 평가는 검증 데이터셋에 대해 다음 **5개 이상의 핵심 평가지표**를 활용하여 종합적으로 판정되었습니다:
    
    1. **정확도 (Accuracy - {tr_metrics['accuracy'] * 100:.2f}%)**:
       - 전체 테스트 이미지 중 모델이 올바르게 결함 종류를 맞춘 샘플의 비율입니다. 전이학습 모델이 도메인 특징을 전반적으로 얼마나 잘 분류하는지 판가름하는 척도입니다.
    2. **가중 정밀도 (Precision - {tr_metrics['precision']:.4f})**:
       - 모델이 특정 결함이라고 판정했을 때, 실제로 그 결함이 맞았던 비율의 가중 평균입니다. 오탐지(False Positive)를 방어하는 강도를 평가합니다.
    3. **가중 재현율 (Recall - {tr_metrics['recall']:.4f})**:
       - 실제 철판에 존재하는 특정 결함들을 모델이 누락하지 않고 검출해 낸 비율의 가중 평균입니다. 결함을 놓치는 미탐지(False Negative)를 방지하는 실질적 검출 능력을 의미합니다.
    4. **가중 F1-Score ({tr_metrics['f1_score']:.4f})**:
       - 정밀도와 재현율의 조화평균값으로, 클래스 불균형 데이터셋에서 모델 성능의 균형감을 측정하는 가장 신뢰도 높은 통합 지표입니다.
    5. **최종 테스트 손실 (Test Loss - {tr_metrics['test_loss']:.4f})**:
       - 교차 엔트로피 손실 함수의 최종 수치로, 예측 확률 분포가 실제 정답 원-핫 레이블과 얼마나 좁게 밀착되어 수렴했는지를 에너지 함수 관점에서 평가합니다.
    """
    st.markdown(desc_metrics)
    
    # 클래스별 세부 지표 표출
    if "class_precisions" in tr_metrics:
        st.markdown("##### 📋 결함 유형(클래스)별 5대 세부 평가지표 분석 스펙")
        class_metrics_data = []
        for i, c_name in enumerate(classes):
            class_metrics_data.append({
                "결함 종류": c_name,
                "정밀도 (Precision)": f"{tr_metrics['class_precisions'][i]:.4f}",
                "재현율 (Recall)": f"{tr_metrics['class_recalls'][i]:.4f}",
                "F1-Score": f"{tr_metrics['class_f1s'][i]:.4f}",
                "지원 데이터 수 (Support)": f"{int(tr_metrics['class_supports'][i])} 개"
            })
        df_class_metrics = pd.DataFrame(class_metrics_data)
        st.table(df_class_metrics.set_index("결함 종류"))

def show_transfer_page(df_images, df_bboxes, classes, IMAGES_DIR, CHECKPOINT_DIR):
    """
    MobileNetV2 사전학습 모델의 동결 전이학습 수행, 가중치 저장/로드 및 
    자체 개선 모델(ImprovedCNN)과의 비교 분석과 실시간 업로드 추론을 렌더링하는 함수입니다.

    인자:
        df_images (DataFrame): 이미지별 데이터셋 메타데이터
        df_bboxes (DataFrame): 바운딩 박스 영역 정보
        classes (list): 고유 결함 클래스 정렬 리스트
        IMAGES_DIR (str): 이미지 파일 로컬 디렉토리 경로
        CHECKPOINT_DIR (str): 가중치 체크포인트 경로
    """
    # 전이학습 세션 상태 정의
    if 'is_trans_trained' not in st.session_state:
        st.session_state.is_trans_trained = False
    if 'trans_history' not in st.session_state:
        st.session_state.trans_history = None
    if 'trans_eval_metrics' not in st.session_state:
        st.session_state.trans_eval_metrics = None
    if 'trans_predictions' not in st.session_state:
        st.session_state.trans_predictions = None
    if 'is_trans_ckpt_loaded' not in st.session_state:
        st.session_state.is_trans_ckpt_loaded = False

    # 페이지 최초 진입 시 로컬 디바이스에 기존 학습 가중치가 있으면 자동 로드 시도
    if not st.session_state.is_trans_trained:
        try:
            temp_trans_model = get_transfer_model(num_classes=len(classes), freeze_backbone=True)
            trans_ckpt = load_transfer_checkpoint(temp_trans_model, CHECKPOINT_DIR)
            if trans_ckpt is not None:
                st.session_state.trans_model = temp_trans_model
                st.session_state.is_trans_trained = True
                st.session_state.trans_eval_metrics = trans_ckpt["eval_metrics"]
                st.session_state.trans_predictions = trans_ckpt["test_predictions"]
                st.session_state.is_trans_ckpt_loaded = True
        except Exception:
            pass

    st.header("🎯 경량 모델 전이학습 및 진단")
    st.markdown("PyTorch의 torchvision에서 제공하는 사전학습 경량 이미지 분류 모델(`models.mobilenet_v2`)을 전이학습하여, 저사양 노트북 환경에서도 CPU 수준으로 초고속 학습을 진행하고 자체 개선 모델과 성능을 정교하게 비교합니다.")

    # 1. 원본 모델 설명 및 핵심 아키텍처 가이드 (Rich Aesthetics 적용)
    with st.expander("ℹ️ **사전 학습 원본 모델 (MobileNetV2) 아키텍처 기술 명세 및 원리 가이드**", expanded=True):
        col_desc1, col_desc2 = st.columns([1.2, 1.8])
        with col_desc1:
            st.markdown("""
            ### 📱 MobileNetV2 개요
            * **개발 주체**: Google (2018년 발표)
            * **핵심 지표**: 총 파라미터 수 약 350만 개 (극히 경량화)
            * **설계 목적**: 스마트폰 등 모바일 디바이스 및 임베디드 기기, 저사양 CPU 환경에서 실시간 추론을 가능케 함
            * **사전학습 데이터셋**: ImageNet-1k (1,000개 클래스, 120만 장 이미지)
            """)
        with col_desc2:
            st.markdown("""
            ### 🏗️ 3대 핵심 아키텍처 메커니즘
            1. **깊이별 분리 합성곱 (Depthwise Separable Convolution)**
               * 3D 커널 연산을 각 채널의 평면만 스캔하는 **Depthwise**와 1x1 커널로 채널을 혼합하는 **Pointwise**로 분할하여, 전통적인 합성곱 연산량 대비 **약 8~9배 수준의 연산 소모량 감소**를 성취했습니다.
            2. **역 잔차 블록 (Inverted Residual Block)**
               * 채널을 좁혔다 넓히는 일반 Residual Block과 반대로, **'좁음 ➔ 팽창(Expansion) ➔ 축소(Projection)'**의 구조를 띱니다. 채널을 중간에 넓혀 풍부한 특징 공간을 제공함으로써 정보 전달 효율을 극대화합니다.
            3. **선형 병목 (Linear Bottleneck)**
               * 채널이 좁아지는 마지막 Projection 층의 직후에는 활성화 함수(ReLU)를 쓰지 않고 **Linear**하게 흘려보냅니다. 차원이 축소될 때 ReLU가 고주파 정보 및 특징 형상을 파괴하는 왜곡을 원천적으로 차단합니다.
            """)
        st.info("💡 **전이학습(Transfer Learning) 메커니즘의 정당성**: ImageNet의 방대한 실물 이미지 학습 과정에서 정교하게 다듬어진 저수준/중수준 시각 필터(모서리, 질감, 대비, 국소 요철 등)를 모델이 이미 내포하고 있습니다. 따라서 마지막 분류 층만 당사의 철강 결함 6종 클래스에 맞춰 교체하고 훈련하면, 저사양 CPU 노트북 환경에서도 에폭당 수 초 만에 높은 정확도를 도출할 수 있습니다.")

    # 2. 자동 복원 알림 및 요약 정보 제공
    if 'is_trans_ckpt_loaded' in st.session_state and st.session_state.is_trans_ckpt_loaded:
        st.success("💾 **전이학습 로컬 체크포인트 복원 완료**: 디스크(`src/checkpoints/transfer/`)에 보관되어 있던 이전 MobileNetV2 전이학습 모델 및 지표 데이터를 자동으로 복원했습니다. 다시 학습시키지 않고 하단의 성능 비교 및 신규 이미지 예측 기능을 즉시 활용할 수 있습니다.")

    # 2. 탭 구성: 전이학습 수행 vs 학습완료 모델 로딩 및 실시간 결함 진단
    tab_tr_train, tab_tr_diag = st.tabs([
        "⚙️ 전이학습 모델 학습 및 가중치 저장",
        "🔬 학습완료 모델 로드 및 실시간 진단"
    ])

    with tab_tr_train:
        st.subheader("📋 사전 학습 경량 모델 (MobileNetV2) 요약")
        
        # 모델 파라미터 정보 동적 계산
        if st.session_state.is_trans_trained and hasattr(st.session_state, 'trans_model'):
            tr_model = st.session_state.trans_model
            tr_total_params = sum(p.numel() for p in tr_model.parameters())
            tr_trainable_params = sum(p.numel() for p in tr_model.parameters() if p.requires_grad)
        else:
            tr_total_params = 3538272  # MobileNetV2 Base parameter count (num_classes=6)
            tr_trainable_params = 7686  # model.classifier (1280 -> 6 Linear layer)
            
        stat_tr1, stat_tr2, stat_tr3 = st.columns(3)
        with stat_tr1:
            st.metric("총 파라미터 수 (Total Parameters)", f"{tr_total_params:,} 개")
        with stat_tr2:
            st.metric("학습 파라미터 (동결 적용 시)", f"{tr_trainable_params:,} 개")
        with stat_tr3:
            st.metric("torchvision 원본 모델 명칭", "models.mobilenet_v2")

        st.markdown("""
        | 레이어 구분 | 구성 레이어 및 명세 | 파라미터 수 | 입력 형태 (Shape) | 출력 형태 (Shape) | 전이학습 동결 여부 |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        | **입력층** | RGB 이미지 (ImageNet 규격) | - | (3, 224, 224) | (3, 224, 224) | - |
        | **백본 (Backbone)** | MobileNetV2 Features (Inverted Residuals) | 약 3.5M | (3, 224, 224) | (1280, 7, 7) | **동결 (Frozen, 학습 안함)** |
        | **풀링층** | Global Average Pooling 2D | - | (1280, 7, 7) | (1280,) | - |
        | **분류층 (Classifier)** | nn.Linear(1280, 6) | 7,686 | (1280,) | (6,) | **활성화 (Trainable, 실시간 훈련)** |
        """)
        st.info("💡 **저사양 노트북 지원의 핵심 원리**: MobileNetV2 백본의 모든 레이어 가중치(약 350만 개)를 학습 과정에서 동결(Frozen)하여 기울기 전파(Backprop) 연산을 원천 배제합니다. 오직 마지막 7,686개의 분류기 파라미터만 훈련하므로 CPU 환경에서도 초당 수십 장의 고속 경량 훈련이 가능합니다.")

        st.markdown("---")
        st.subheader("⚙️ 전이학습(Fine-tuning) 설정 및 실시간 학습 진행")
        
        tr_col1, tr_col2, tr_col3, tr_col4 = st.columns(4)
        with tr_col1:
            tr_epochs = st.slider("전이학습 Epoch 수", min_value=1, max_value=5, value=2, step=1, key="tr_epochs_sl")
        with tr_col2:
            tr_batch_size = st.selectbox("전이학습 배치 크기", [8, 16, 32], index=1, key="tr_batch_sl")
        with tr_col3:
            tr_learning_rate = st.selectbox("전이학습 학습률", [0.01, 0.005, 0.001], index=2, key="tr_lr_sl")
        with tr_col4:
            tr_sampling_ratio = st.slider("전이학습 데이터셋 비율 (%)", min_value=10, max_value=100, value=20, step=10, key="tr_ratio_sl")

        btn_tr_train = st.button("🚀 MobileNetV2 전이학습 시작", use_container_width=True)
        tr_graph_placeholder = st.empty()

        if btn_tr_train:
            st.info("torchvision 사전 학습된 MobileNetV2 모델을 준비하고 데이터셋을 전이학습 규격으로 가공하는 중...")
            
            # 1. 전이학습 데이터셋 구축 및 분할
            sampled_tr_list = []
            for c in classes:
                c_df = df_images[df_images['class_prefix'] == c]
                if len(c_df) > 0:
                    sample_count = max(int(len(c_df) * (tr_sampling_ratio / 100.0)), 5)
                    sample_count = min(sample_count, len(c_df))
                    sampled_tr_list.append(c_df.sample(n=sample_count, random_state=42))
                
            df_tr_sampled = pd.concat(sampled_tr_list).sample(frac=1.0, random_state=42).reset_index(drop=True)
            
            n_tr_total = len(df_tr_sampled)
            n_tr_train = int(n_tr_total * 0.7)
            n_tr_val = int(n_tr_total * 0.15)
            
            train_tr_df = df_tr_sampled.iloc[:n_tr_train].reset_index(drop=True)
            val_tr_df = df_tr_sampled.iloc[n_tr_train:n_tr_train+n_tr_val].reset_index(drop=True)
            test_tr_df = df_tr_sampled.iloc[n_tr_train+n_tr_val:].reset_index(drop=True)
            
            # RGB, 224x224 리사이즈 규격의 TransferDataset 생성
            train_tr_dataset = TransferDataset(train_tr_df['image_name'].values, train_tr_df['class_prefix'].values, classes, IMAGES_DIR)
            val_tr_dataset = TransferDataset(val_tr_df['image_name'].values, val_tr_df['class_prefix'].values, classes, IMAGES_DIR)
            test_tr_dataset = TransferDataset(test_tr_df['image_name'].values, test_tr_df['class_prefix'].values, classes, IMAGES_DIR)
            
            train_tr_loader = DataLoader(train_tr_dataset, batch_size=tr_batch_size, shuffle=True)
            val_tr_loader = DataLoader(val_tr_dataset, batch_size=tr_batch_size, shuffle=False)
            test_tr_loader = DataLoader(test_tr_dataset, batch_size=tr_batch_size, shuffle=False)
            
            st.success(f"전이학습 데이터 준비 완료! (훈련셋: {len(train_tr_dataset)}장, 검증셋: {len(val_tr_dataset)}장, 테스트셋: {len(test_tr_dataset)}장)")
            
            # 2. 모델 로드 (Hugging Face 가중치 다운로드 및 백본 동결)
            tr_model = get_transfer_model(num_classes=len(classes), freeze_backbone=True)
            tr_criterion = nn.CrossEntropyLoss()
            tr_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, tr_model.parameters()), lr=tr_learning_rate)
            
            tr_train_losses, tr_train_accs = [], []
            tr_val_losses, tr_val_accs = [], []
            
            tr_progress = st.progress(0)
            
            # 3. 훈련 루프
            for epoch in range(tr_epochs):
                tr_model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for images, labels, _ in train_tr_loader:
                    tr_optimizer.zero_grad()
                    logits = tr_model(images)
                    loss = tr_criterion(logits, labels)
                    loss.backward()
                    tr_optimizer.step()
                    
                    running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                epoch_loss = running_loss / len(train_tr_dataset)
                epoch_acc = correct / total
                tr_train_losses.append(epoch_loss)
                tr_train_accs.append(epoch_acc)
                
                # 검증 단계
                tr_model.eval()
                val_running_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for val_images, val_labels, _ in val_tr_loader:
                        val_logits = tr_model(val_images)
                        val_loss = tr_criterion(val_logits, val_labels)
                        
                        val_running_loss += val_loss.item() * val_images.size(0)
                        _, val_predicted = torch.max(val_logits.data, 1)
                        val_total += val_labels.size(0)
                        val_correct += (val_predicted == val_labels).sum().item()
                        
                val_epoch_loss = val_running_loss / len(val_tr_dataset)
                val_epoch_acc = val_correct / val_total
                tr_val_losses.append(val_epoch_loss)
                tr_val_accs.append(val_epoch_acc)
                
                tr_progress.progress((epoch + 1) / tr_epochs)
                
                # 실시간 차트 업데이트
                with tr_graph_placeholder.container():
                    g_col1, g_col2 = st.columns(2)
                    with g_col1:
                        fig_trl, ax_trl = plt.subplots(figsize=(6, 4))
                        ax_trl.plot(range(1, epoch + 2), tr_train_losses, label="Train Loss", marker='o', color='blue')
                        ax_trl.plot(range(1, epoch + 2), tr_val_losses, label="Val Loss", marker='s', color='orange')
                        ax_trl.set_xlabel("Epoch")
                        ax_trl.set_ylabel("Loss")
                        ax_trl.legend()
                        ax_trl.set_title("MobileNetV2 전이학습 Loss 곡선")
                        st.pyplot(fig_trl)
                        plt.close(fig_trl)
                    with g_col2:
                        fig_tra, ax_tra = plt.subplots(figsize=(6, 4))
                        ax_tra.plot(range(1, epoch + 2), tr_train_accs, label="Train Acc", marker='o', color='green')
                        ax_tra.plot(range(1, epoch + 2), tr_val_accs, label="Val Acc", marker='s', color='red')
                        ax_tra.set_xlabel("Epoch")
                        ax_tra.set_ylabel("Accuracy")
                        ax_tra.legend()
                        ax_tra.set_title("MobileNetV2 전이학습 Accuracy 곡선")
                        st.pyplot(fig_tra)
                        plt.close(fig_tra)
                        
            # 최종 테스트 셋 평가
            st.info("전이학습 모델의 최종 테스트셋 성능 검증을 수행하는 중...")
            eval_tr_results = evaluate_transfer_model(tr_model, test_tr_loader, tr_criterion, classes)
            
            st.session_state.trans_model = tr_model
            st.session_state.is_trans_trained = True
            st.session_state.trans_history = {
                "train_losses": tr_train_losses,
                "train_accs": tr_train_accs,
                "val_losses": tr_val_losses,
                "val_accs": tr_val_accs
            }
            st.session_state.trans_eval_metrics = {
                "accuracy": eval_tr_results["accuracy"],
                "precision": eval_tr_results["precision"],
                "recall": eval_tr_results["recall"],
                "f1_score": eval_tr_results["f1_score"],
                "test_loss": eval_tr_results["test_loss"],
                "conf_matrix": eval_tr_results["conf_matrix"],
                "class_precisions": eval_tr_results.get("class_precisions", []),
                "class_recalls": eval_tr_results.get("class_recalls", []),
                "class_f1s": eval_tr_results.get("class_f1s", []),
                "class_supports": eval_tr_results.get("class_supports", [])
            }
            st.session_state.trans_predictions = pd.DataFrame(eval_tr_results["predictions_df_data"])
            
            # 체크포인트 디스크 저장
            save_transfer_checkpoint(tr_model, CHECKPOINT_DIR, st.session_state.trans_eval_metrics, st.session_state.trans_predictions)
            st.session_state.is_trans_ckpt_loaded = True
            
            st.success("MobileNetV2 전이학습 및 테스트 평가 완료! 학습 모델 가중치가 `src/checkpoints/transfer/` 경로에 안전하게 백업 저장되었습니다.")
            # 학습 탭 하단에도 5대 평가지표와 세부 스펙 분석 추가 노출
            render_transfer_metrics_report(st.session_state.trans_eval_metrics, classes)
            st.info("💡 **'학습완료 모델 로드 및 실시간 진단'** 탭으로 이동하시면 기존 ImprovedCNN 자체 모델과의 1:1 성능 비교 그래프도 함께 확인하실 수 있습니다.")

    with tab_tr_diag:
        st.subheader("🔬 저장된 전이학습 모델 로드 및 분석/예측 진단")
        st.markdown("이미 로컬 디스크에 저장된 MobileNetV2 전이학습 결과를 불러오거나, 학습 완료 상태인 모델을 사용하여 실시간 표면 결함 예측을 수행합니다. 매번 오랜 시간 학습을 반복할 필요가 없습니다.")
        
        # 명시적 가중치 로드 제어 버튼 제공
        btn_tr_load = st.button("💾 저장된 전이학습 가중치 불러오기 (Import Checkpoint)", use_container_width=True)
        
        if btn_tr_load:
            with st.spinner("로컬 디바이스 디스크(src/checkpoints/transfer/)에서 모델을 조회하는 중..."):
                temp_trans_model = get_transfer_model(num_classes=len(classes), freeze_backbone=True)
                trans_ckpt = load_transfer_checkpoint(temp_trans_model, CHECKPOINT_DIR)
                if trans_ckpt is not None:
                    st.session_state.trans_model = temp_trans_model
                    st.session_state.is_trans_trained = True
                    st.session_state.trans_eval_metrics = trans_ckpt["eval_metrics"]
                    st.session_state.trans_predictions = trans_ckpt["test_predictions"]
                    st.session_state.is_trans_ckpt_loaded = True
                    st.success("💾 **체크포인트 복원 성공**: 이전 학습된 가중치와 5대 평가지표 메타데이터를 완벽히 탑재했습니다.")
                else:
                    st.error("⚠️ **복원 실패**: 저장된 전이학습 모델을 찾을 수 없거나 파일 형식이 손상되었습니다. '전이학습 모델 학습 및 가중치 저장' 탭에서 학습을 새로 진행해 주세요.")

        # 모델이 학습 또는 복원된 상태라면 시각화 및 예측 진단 보드 출력
        if st.session_state.is_trans_trained and hasattr(st.session_state, 'trans_model'):
            tr_metrics = st.session_state.trans_eval_metrics
            
            # 헬퍼 함수를 통한 5대 지표 및 종합 평가지표 테이블/리포트 렌더링
            render_transfer_metrics_report(tr_metrics, classes)

            # ImprovedCNN vs MobileNetV2 비교
            if st.session_state.is_trained:
                st.markdown("##### 📊 모델 간 핵심 성능 지표(Metric) 1:1 대조 비교")
                cnn_metrics = st.session_state.eval_metrics
                
                compare_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                cnn_values = [cnn_metrics['accuracy'], cnn_metrics['precision'], cnn_metrics['recall'], cnn_metrics['f1_score']]
                mob_values = [tr_metrics['accuracy'], tr_metrics['precision'], tr_metrics['recall'], tr_metrics['f1_score']]
                
                x_axis = np.arange(len(compare_labels))
                width_val = 0.35
                
                fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
                rects1 = ax_comp.bar(x_axis - width_val/2, cnn_values, width_val, label='ImprovedCNN (자체 CNN 모델)', color='lightcoral', edgecolor='black')
                rects2 = ax_comp.bar(x_axis + width_val/2, mob_values, width_val, label='MobileNetV2 (HF 전이학습)', color='skyblue', edgecolor='black')
                
                ax_comp.set_ylabel('스코어 값 (Score)')
                ax_comp.set_title('두 분류 모델 간의 핵심 지표 대조 비교 분석')
                ax_comp.set_xticks(x_axis)
                ax_comp.set_xticklabels(compare_labels)
                ax_comp.set_ylim(0, 1.1)
                ax_comp.legend()
                
                def autolabel(rects):
                    for rect in rects:
                        h_val = rect.get_height()
                        ax_comp.annotate(f'{h_val:.3f}',
                                    xy=(rect.get_x() + rect.get_width() / 2, h_val),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8, fontweight='bold')
                                    
                autolabel(rects1)
                autolabel(rects2)
                
                st.pyplot(fig_comp)
                plt.close(fig_comp)
                st.write("📈 **비교 분석 의견**: 사전 학습된 MobileNetV2 모델은 ImageNet의 방대한 기하학적 형상화 데이터셋에 길들여져 있어, 철판 표면 결함과 같은 좁은 도메인에서도 백본을 동결한 채 소수의 Classifier 에폭 학습만으로 매우 신속하게 훈련 수렴도가 올라가는 장점이 있습니다. 반면, 자체 구조인 ImprovedCNN은 철판 결함 도메인에 전용화된 특징을 64x64부터 학습하여 특정 결함 경계 감지에 우위를 보일 수 있습니다.")
                
            st.markdown("---")

            # 5. 신규 데이터 업로드 실시간 예측 분석
            st.subheader("📤 신규 데이터를 통한 전이학습 모델의 예측 검증")
            st.markdown("새로운 철판 표면 이미지를 주입하여 불러온 MobileNetV2 전이학습 분류 모델이 결함을 정상적으로 예측하고 분류해내는지 실시간 검증합니다.")
            
            tr_up_file = st.file_uploader("검증용 이미지 업로드", type=["jpg", "jpeg", "png"], key="tr_up_uploader")
            
            if tr_up_file is not None:
                try:
                    tr_up_orig = Image.open(tr_up_file).convert("RGB")
                    
                    tr_up_resized = tr_up_orig.resize((224, 224))
                    tr_up_np = np.array(tr_up_resized, dtype=np.float32) / 255.0
                    tr_up_tensor = torch.tensor(tr_up_np).permute(2, 0, 1)
                    
                    tr_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    tr_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    tr_up_tensor = ((tr_up_tensor - tr_mean) / tr_std).unsqueeze(0)
                    
                    tr_model = st.session_state.trans_model
                    tr_model.eval()
                    with torch.no_grad():
                        tr_up_logits = tr_model(tr_up_tensor)
                        tr_up_probs = F.softmax(tr_up_logits, dim=1).cpu().numpy()[0]
                        
                    tr_up_pred_idx = np.argmax(tr_up_probs)
                    tr_up_pred_label = classes[tr_up_pred_idx]
                    tr_up_pred_pct = tr_up_probs[tr_up_pred_idx] * 100
                    
                    # ----------------- 결함 Bounding Box XML 검사 및 파싱 시작 -----------------
                    # 업로드된 파일명에서 확장자를 제거하고 xml 매핑 파일 경로 확인
                    tr_base_name = os.path.splitext(tr_up_file.name)[0]
                    # 상위 폴더 경로 구조 기준
                    tr_xml_path = os.path.join(os.path.dirname(IMAGES_DIR), "ANNOTATIONS", tr_base_name + ".xml")
                    
                    tr_bboxes = []
                    tr_draw_img = tr_up_orig.copy()
                    
                    # 결함 클래스별 색상 매핑
                    tr_colors = {
                        "crazing": "#FF0000",
                        "inclusion": "#00FF00",
                        "patches": "#0000FF",
                        "pitted_surface": "#FFFF00",
                        "rolled-in_scale": "#FF00FF",
                        "scratches": "#00FFFF"
                    }
                    
                    if os.path.exists(tr_xml_path):
                        import xml.etree.ElementTree as ET
                        from PIL import ImageDraw
                        try:
                            tree_tr = ET.parse(tr_xml_path)
                            root_tr = tree_tr.getroot()
                            
                            draw_ctx_tr = ImageDraw.Draw(tr_draw_img)
                            for obj in root_tr.findall('object'):
                                c_name = obj.find('name').text
                                bndbox = obj.find('bndbox')
                                xmin = float(bndbox.find('xmin').text)
                                ymin = float(bndbox.find('ymin').text)
                                xmax = float(bndbox.find('xmax').text)
                                ymax = float(bndbox.find('ymax').text)
                                
                                w_val = xmax - xmin
                                h_val = ymax - ymin
                                area_val = w_val * h_val
                                
                                tr_bboxes.append({
                                    "결함 종류": c_name,
                                    "시작 X (xmin)": xmin,
                                    "시작 Y (ymin)": ymin,
                                    "너비 (width)": w_val,
                                    "높이 (height)": h_val,
                                    "면적 (area, px²)": area_val
                                })
                                
                                # 클래스별 매핑된 컬러로 사각형 테두리와 이름 드로잉
                                color_hex = tr_colors.get(c_name, "#FFFFFF")
                                draw_ctx_tr.rectangle([xmin, ymin, xmax, ymax], outline=color_hex, width=3)
                                draw_ctx_tr.text((xmin + 2, ymin + 2), c_name, fill=color_hex)
                        except Exception as e_xml:
                            st.warning(f"업로드 이미지의 XML 어노테이션을 파싱하는 도중 에러가 발생했습니다: {e_xml}")
                    # ----------------- 결함 Bounding Box XML 검사 및 파싱 끝 -----------------
                    
                    tr_up_col1, tr_up_col2 = st.columns(2)
                    with tr_up_col1:
                        if len(tr_bboxes) > 0:
                            st.markdown("##### 📸 Ground Truth 결함 박스 시각화")
                            st.image(tr_draw_img, caption=f"결함 영역이 오버레이된 이미지 (파일명: {tr_up_file.name})", use_container_width=True)
                        else:
                            st.markdown("##### 📸 업로드 검증 이미지")
                            st.image(tr_up_orig, caption=f"매핑된 XML 결함 데이터가 없습니다 (파일명: {tr_up_file.name})", use_container_width=True)
                    with tr_up_col2:
                        st.markdown("##### 🤖 MobileNetV2 전이학습 예측")
                        st.info(f"💡 결함 최종 예측: **{tr_up_pred_label}** (신뢰도: **{tr_up_pred_pct:.2f}%**)")
                        
                        fig_tr_up, ax_tr_up = plt.subplots(figsize=(6, 4))
                        y_pos_comp = range(len(classes))
                        bars_tr_up = ax_tr_up.barh(y_pos_comp, tr_up_probs * 100, align='center', color='teal', edgecolor='black')
                        ax_tr_up.set_yticks(y_pos_comp)
                        ax_tr_up.set_yticklabels(classes)
                        ax_tr_up.invert_yaxis()
                        ax_tr_up.set_xlabel("예측 확률 (%)")
                        ax_tr_up.set_xlim(0, 100)
                        ax_tr_up.set_title("MobileNetV2 예측 분포도")
                        
                        for bar in bars_tr_up:
                            w_val = bar.get_width()
                            ax_tr_up.text(w_val + 2, bar.get_y() + bar.get_height()/2.0, f'{w_val:.1f}%',
                                         ha='left', va='center', fontsize=9, fontweight='bold')
                                         
                        st.pyplot(fig_tr_up)
                        plt.close(fig_tr_up)
                        
                    # ----------------- 결함 상세 데이터 테이블 추가 시작 -----------------
                    if len(tr_bboxes) > 0:
                        st.subheader("📊 검출된 결함 객체 스펙 테이블 및 요약 리포트")
                        df_tr_box = pd.DataFrame(tr_bboxes)
                        st.dataframe(df_tr_box, use_container_width=True)
                        
                        total_area_ratio = df_tr_box["면적 (area, px²)"].sum() / (tr_up_orig.width * tr_up_orig.height) * 100
                        st.success(f"검출 요약: 업로드된 이미지에는 총 **{len(tr_bboxes)}개**의 실제 결함 객체가 정의되어 있으며, "
                                   f"이미지 내 전체 면적 대비 결함 총 크기 비율은 **{total_area_ratio:.2f}%**입니다.")
                    # ----------------- 결함 상세 데이터 테이블 추가 끝 -----------------
                except Exception as e_tr_up:
                    st.error(f"예측 검증 중 오류: {e_tr_up}")
            else:
                st.info("💡 검증하고 싶은 새로운 철판 표면 이미지를 올려주시면 MobileNetV2 전이학습 분류기가 작동합니다.")
        else:
            st.warning("⚠️ 아직 학습 완료된 전이학습 모델이 메모리에 탑재되지 않았습니다. '💾 저장된 전이학습 가중치 불러오기' 버튼을 클릭해 불러오거나, '⚙️ 전이학습 모델 학습 및 가중치 저장' 탭에서 학습을 완료해 주세요.")

