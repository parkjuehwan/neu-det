# -*- coding: utf-8 -*-
"""
CNN 모델 구조 플로우차트, 학습/평가 시퀀스 다이어그램, 훈련 전처리 타임라인 간트차트 등의 
Mermaid 차트 렌더링을 처리하는 대시보드 페이지 모듈입니다.
"""

import streamlit as st

def show_flowchart_page(render_mermaid):
    """
    CNN 모델의 레이어 구조와 텐서 차원 흐름을 시각화하는 플로우차트 페이지입니다.

    인자:
        render_mermaid (function): Mermaid 코드를 렌더링하는 함수
    """
    st.header("🎨 CNN 모델 구조 플로우차트 (Flowchart)")
    st.markdown("각 레이어별 상세 연산 흐름과 텐서의 차원 변화를 나타낸 다이어그램입니다.")
    st.info("입력 이미지 64x64 그레이스케일 텐서가 2층 합성곱 및 풀링을 통과하며 특징 맵 크기 변화 및 최종 다중 분류로 연결되는 전방향 계산 그래프입니다.")
    
    cnn_structure_code = """
    flowchart TD
        subgraph InputLayer ["1. 입력 레이어"]
            In["입력 이미지 텐서: 64 x 64 x 1"]
        end

        subgraph ConvBlock1 ["2. 합성곱 블록 1"]
            Conv1["합성곱층 (Conv2D): 16채널, 3x3 커널"] --> Act1["활성화함수 (ReLU)"]
            Act1 --> Pool1["맥스풀링 (MaxPool2D): 2x2 (출력: 32x32x16)"]
        end

        subgraph ConvBlock2 ["3. 합성곱 블록 2"]
            Conv2["합성곱층 (Conv2D): 32채널, 3x3 커널"] --> Act2["활성화함수 (ReLU)"]
            Act2 --> Pool2["맥스풀링 (MaxPool2D): 2x2 (출력: 16x16x32)"]
        end

        subgraph FlatBlock ["4. 차원 플랫화"]
            Flat["텐서 플랫화: 32 x 16 x 16 -> 8192차원 벡터"]
        end

        subgraph DenseBlock1 ["5. 전결합 블록 1"]
            FC1["전결합층 (Dense): 128 노드"] --> Act3["활성화함수 (ReLU)"]
        end

        subgraph OutputBlock ["6. 출력 블록"]
            FC2["출력층 (Dense): 6 클래스"] --> Out["결함 분류 예측 확률 (Softmax)"]
        end

        In --> ConvBlock1
        ConvBlock1 --> ConvBlock2
        ConvBlock2 --> FlatBlock
        FlatBlock --> DenseBlock1
        DenseBlock1 --> OutputBlock
    """
    # 세로 배치를 고려하여 충분한 높이(750px) 설정
    render_mermaid(cnn_structure_code, height=750)


def show_sequence_page(render_mermaid):
    """
    훈련 실행 시 각 구성요소(UI, DataLoader, Model, Optimizer)간의 시퀀스 흐름을 나타내는 페이지입니다.

    인자:
        render_mermaid (function): Mermaid 코드를 렌더링하는 함수
    """
    st.header("🔄 학습/평가 시퀀스 다이어그램 (Sequence)")
    st.markdown("학습 시작 버튼 트리거 이후 일어나는 데이터 로드, 순전파, 역전파 가중치 갱신의 상호작용 프로세스입니다.")
    st.info("사용자의 학습 버튼 트리거 시점부터 미니배치 로딩, 순전파 계산, 손실 산출, 오차 역전파 및 Adam 가중치 업데이트로 이어지는 학습 시스템의 시퀀스 라이프사이클입니다.")
    
    sequence_code = """
    sequenceDiagram
        participant User as "사용자"
        participant UI as "대시보드 UI"
        participant DL as "DataLoader"
        participant Model as "CNN 모델"
        participant Optimizer as "최적화 도구 (Adam)"
        
        User->>UI: "'학습 시작' 클릭"
        UI->>DL: "데이터 분할 및 미니배치 구성"
        loop "에폭 학습 루프"
            DL->>Model: "배치 데이터 입력 (Forward)"
            Model->>Model: "손실값 계산 (CrossEntropy)"
            Model->>Optimizer: "역전파 오차 계산 (Backward)"
            Optimizer->>Model: "모델 가중치 업데이트 (Step)"
            UI->>User: "실시간 학습 곡선 갱신"
        end
        Model->>UI: "평가셋 검증 완료 리포트"
    """
    render_mermaid(sequence_code, height=550)


def show_gantt_page(render_mermaid):
    """
    훈련 전처리 및 검증 일정 타임라인을 나타내는 간트차트 페이지입니다.

    인자:
        render_mermaid (function): Mermaid 코드를 렌더링하는 함수
    """
    st.header("📅 훈련 전처리 타임라인 간트차트 (Gantt)")
    st.markdown("데이터 전처리부터 학습 진행 및 검증 성능 평가까지의 흐름을 일정 타임라인으로 보여줍니다.")
    st.info("데이터 로딩, 모델 학습 및 검증, 최종 5대 성능 평가 및 혼동 행렬 시각화에 이르는 단위 작업들의 실행 시간 타임라인 일정입니다.")
    
    gantt_code = """
    gantt
        title "CNN 훈련 및 검증 타임라인"
        dateFormat  X
        axisFormat %s
        section "데이터 준비"
        "이미지 리사이즈 & 텐서변환"    :active, t1, 0, 2
        "훈련/검증/테스트 데이터셋 분할" : t2, after t1, 1
        section "모델 훈련"
        "에폭 1~N 실시간 경사하강 학습" : t3, after t2, 5
        "에폭별 검증셋 Loss/Acc 평가" : t4, after t2, 5
        section "최종 평가"
        "테스트셋 5대 평가지표 추출"   : t5, after t4, 2
        "혼동행렬 히트맵 렌더링"       : t6, after t5, 1
    """
    render_mermaid(gantt_code, height=480)
