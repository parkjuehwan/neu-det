# -*- coding: utf-8 -*-
"""
대시보드와 물리적으로 분리되어 설계된 AI 모델 아키텍처 및 기하학적 특징 분석용 XAI 코어가 수록된 
model.py 소스 코드를 웹 화면상에 제공하는 페이지 모듈입니다.
"""

import os
import streamlit as st

def show_code_viewer_page(SRC_DIR):
    """
    model.py 소스 코드를 로드하여 대시보드 화면상에 가독성 높게 출력해주는 함수입니다.

    인자:
        SRC_DIR (str): 소스 코드 디렉토리 경로
    """
    st.header("📂 분리된 모델 및 학습 엔진 소스 코드 뷰어")
    st.markdown("대시보드와 분리되어 훈련 아키텍처 및 XAI(Grad-CAM) 모듈을 전담하는 `model.py` 파일의 실제 전체 소스 코드입니다.")
    
    # model.py 파일 읽기
    model_path = os.path.join(SRC_DIR, "model.py")
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            code_content = f.read()
        
        st.markdown("### 📝 `model.py` 소스 코드")
        st.code(code_content, language="python")
    except Exception as e:
        st.error(f"model.py 파일을 읽어올 수 없습니다: {e}")
