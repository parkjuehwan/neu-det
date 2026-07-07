# -*- coding: utf-8 -*-
"""
NEU-DET 표면 결함 데이터셋 분석 및 CNN 모델링 대시보드의 메인 진입점 모듈입니다.
각 메뉴(페이지)는 src/pages/ 디렉토리 내의 개별 모듈로 분리되어 있으며,
사용자가 선택한 페이지에 따라 동적으로 지연 로딩(Lazy Loading)을 수행하여 초기 및 페이지 전환 속도를 극대화합니다.
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# 페이지 레이아웃 설정 (반드시 모든 Streamlit 명령 중 처음에 실행되어야 함)
st.set_page_config(
    page_title="NEU-DET 결함 데이터셋 분석 및 CNN 모델링 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 절대경로 기준 상대경로 설정 (어떤 경로에서 실행해도 파일 로드에 실패하지 않도록 설정)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data", "NEU-DET")
IMAGES_DIR = os.path.join(DATA_DIR, "IMAGES")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "ANNOTATIONS")
CHECKPOINT_DIR = os.path.join(SRC_DIR, "checkpoints")

# 클래스 색상 지정 (Bounding Box 그리기 용)
CLASS_COLORS = {
    "crazing": "#FF0000",       # 빨간색
    "inclusion": "#00FF00",     # 초록색
    "patches": "#0000FF",       # 파란색
    "pitted_surface": "#FFFF00", # 노란색
    "rolled-in_scale": "#FF00FF", # 마젠타
    "scratches": "#00FFFF"      # 시안
}

def render_mermaid(code: str, height: int = 350):
    """
    HTML iframe과 Mermaid.js CDN을 사용하여 
    대시보드 상에 머메이드 다이어그램을 강제 렌더링하는 헬퍼 함수입니다.
    
    인자:
        code (str): 렌더링할 Mermaid 마크다운 코드
        height (int): iframe의 높이 (픽셀 단위)
    """
    # Mermaid.js 초기화 시 간트차트 등의 좌측 여백을 넉넉히 주어 한글 라벨이 잘리지 않도록 조정합니다.
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #F8F9FA;
                overflow: auto;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }}
            .mermaid {{
                width: 100%;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid">
            {code}
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ 
                startOnLoad: true, 
                theme: 'default',
                securityLevel: 'loose',
                gantt: {{
                    leftPadding: 180
                }}
            }});
        </script>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=height, scrolling=True)

@st.cache_data
def load_and_parse_dataset():
    """
    NEU-DET 데이터셋 폴더 내의 이미지와 어노테이션 XML을 읽어 
    데이터프레임으로 변환하여 캐싱합니다.
    
    반환:
        df_images (DataFrame): 이미지별 메타데이터 정보
        df_bboxes (DataFrame): 어노테이션 결함 바운딩박스 정보
        classes (list): 데이터셋 내 고유 결함 클래스 목록
    """
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(ANNOTATIONS_DIR):
        return pd.DataFrame(), pd.DataFrame(), []
        
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    image_data = []
    bbox_data = []
    
    for img_name in image_files:
        base_name = os.path.splitext(img_name)[0]
        xml_name = base_name + ".xml"
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_name)
        img_path = os.path.join(IMAGES_DIR, img_name)
        
        # 이미지 기본 정보 획득
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                channels = len(img.getbands())
                img_np = np.array(img)
                pixel_mean = float(np.mean(img_np))
                pixel_std = float(np.std(img_np))
        except Exception:
            continue
            
        # 클래스 접두사 추출
        class_prefix = img_name.rsplit('_', 1)[0]
        
        image_data.append({
            "image_name": img_name,
            "class_prefix": class_prefix,
            "width": width,
            "height": height,
            "channels": channels,
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
            "has_xml": os.path.exists(xml_path)
        })
        
        # XML 어노테이션 파싱
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    bbox_data.append({
                        "image_name": img_name,
                        "class_name": class_name,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "bbox_width": xmax - xmin,
                        "bbox_height": ymax - ymin,
                        "bbox_area": (xmax - xmin) * (ymax - ymin)
                    })
            except Exception:
                pass
                
    df_images = pd.DataFrame(image_data)
    df_bboxes = pd.DataFrame(bbox_data)
    
    # 고유 클래스 목록 추출
    classes = list(df_bboxes['class_name'].unique()) if not df_bboxes.empty else []
    classes.sort()
    
    return df_images, df_bboxes, classes

# 데이터 로딩
df_images, df_bboxes, classes = load_and_parse_dataset()

# 만약 데이터가 없다면 경고 표시
if df_images.empty:
    st.error(f"데이터셋 경로를 찾을 수 없습니다. 경로를 확인해 주세요.\n조회 경로: {DATA_DIR}")
    st.stop()

# ==================== 가상 멀티페이지용 사이드바 제어 패널 ====================
st.sidebar.title("🛠️ 제어 패널")
page = st.sidebar.radio(
    "원하는 대시보드 페이지를 선택하세요:", 
    [
        "📊 데이터셋 EDA 및 브라우징", 
        "🧠 CNN 모델 학습 및 평가", 
        "🚀 학습 완료 모델 예측 및 평가", 
        "🎯 경량 모델 전이학습 및 진단",
        "🎨 CNN 모델 구조 플로우차트",
        "🔄 학습/평가 시퀀스 다이어그램",
        "📅 훈련 전처리 타임라인 간트차트",
        "📂 모델 소스 코드 뷰어"
    ]
)

# 세션 상태 초기화 (상세보기용 이미지 설정)
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = df_images['image_name'].iloc[0]

# 학습 완료 상태 저장을 위한 세션 상태 정의
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False
if 'train_history' not in st.session_state:
    st.session_state.train_history = None
if 'eval_metrics' not in st.session_state:
    st.session_state.eval_metrics = None
if 'test_predictions' not in st.session_state:
    st.session_state.test_predictions = None
if 'is_ckpt_loaded' not in st.session_state:
    st.session_state.is_ckpt_loaded = False

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

# ==================== 페이지 렌더링 파트 (동적 임포트 적용) ====================
if page == "📊 데이터셋 EDA 및 브라우징":
    from pages.eda_page import show_eda_page
    show_eda_page(df_images, df_bboxes, classes, IMAGES_DIR, CLASS_COLORS)
    
elif page == "🧠 CNN 모델 학습 및 평가":
    from pages.train_page import show_train_page
    show_train_page(df_images, df_bboxes, classes, IMAGES_DIR, CHECKPOINT_DIR, CLASS_COLORS)

elif page == "🚀 학습 완료 모델 예측 및 평가":
    from pages.predict_page import show_predict_page
    show_predict_page(df_images, df_bboxes, classes, IMAGES_DIR, CHECKPOINT_DIR, CLASS_COLORS)

elif page == "🎯 경량 모델 전이학습 및 진단":
    from pages.transfer_page import show_transfer_page
    show_transfer_page(df_images, df_bboxes, classes, IMAGES_DIR, CHECKPOINT_DIR)

elif page == "🎨 CNN 모델 구조 플로우차트":
    from pages.visual_pages import show_flowchart_page
    show_flowchart_page(render_mermaid)

elif page == "🔄 학습/평가 시퀀스 다이어그램":
    from pages.visual_pages import show_sequence_page
    show_sequence_page(render_mermaid)

elif page == "📅 훈련 전처리 타임라인 간트차트":
    from pages.visual_pages import show_gantt_page
    show_gantt_page(render_mermaid)

elif page == "📂 모델 소스 코드 뷰어":
    from pages.code_viewer_page import show_code_viewer_page
    show_code_viewer_page(SRC_DIR)
