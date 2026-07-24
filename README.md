# NEU-DET 강판 표면 결함 분류

PyTorch로 강판 표면 결함을 분류하고, 전체 분석 및 모델링 흐름을 Streamlit 대시보드로 보여주는 포트폴리오 프로젝트입니다.

이 프로젝트는 NEU-DET annotation XML 파일을 사용해 결함 영역을 crop하고, Baseline CNN과 ResNet18 전이학습 모델을 학습합니다. 이후 두 모델의 성능을 비교하고, ResNet18의 판단 근거를 Grad-CAM으로 시각화합니다.

## 결함 클래스

- `crazing`
- `inclusion`
- `patches`
- `pitted_surface`
- `rolled-in_scale`
- `scratches`

## 프로젝트 구조

```text
NEU-DET/
|-- app.py
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_baseline_cnn.ipynb
|   |-- 03_resnet18_transfer.ipynb
|   `-- 04_gradcam.ipynb
|-- outputs/
|   |-- figures/
|   `-- results/
|-- src/
|   `-- dashboard_utils.py
|-- tests/
|   `-- test_dashboard_utils.py
`-- requirements.txt
```

다음 대용량 로컬 산출물은 Git에 포함하지 않습니다.

- `train/`
- `validation/`
- `crops/`
- `outputs/models/`
- `*.pth`

## 작업 흐름

1. `01_eda.ipynb`
   - XML annotation을 파싱합니다.
   - `outputs/results/bbox_annotations.csv`를 생성합니다.
   - 클래스 분포, bounding box 형태, 밝기, 대비에 대한 EDA figure를 생성합니다.
   - 결함 영역 crop 이미지를 `crops/` 아래에 생성합니다.

2. `02_baseline_cnn.ipynb`
   - crop된 결함 이미지를 사용해 간단한 grayscale CNN을 학습합니다.
   - baseline 성능 지표와 confusion matrix를 저장합니다.

3. `03_resnet18_transfer.ipynb`
   - grayscale crop 이미지를 3채널로 변환해 ResNet18 전이학습을 수행합니다.
   - 모델 비교 결과와 ResNet18 성능 지표를 저장합니다.

4. `04_gradcam.ipynb`
   - 가장 성능이 좋은 ResNet18 checkpoint를 불러옵니다.
   - validation crop 이미지에 대한 Grad-CAM 시각화를 생성합니다.

## 대시보드

Streamlit 대시보드는 노트북 기반 워크플로우를 인터랙티브 포트폴리오 데모로 구성합니다.

구성 섹션:

- Overview
- EDA
- Model Performance
- Error Analysis
- Grad-CAM
- Try Model

`Try Model` 섹션은 로컬에 `outputs/models/resnet18_best.pth` 파일이 있을 때 validation crop 선택과 이미지 업로드 예측을 지원합니다. checkpoint가 없어도 저장된 figure와 CSV 파일을 사용하는 나머지 대시보드 섹션은 정상적으로 동작합니다.

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 대시보드 실행

```bash
streamlit run app.py
```

실행 후 아래 주소를 엽니다.

```text
http://localhost:8501
```

Windows에서는 다음 스크립트로도 실행할 수 있습니다.

```powershell
.\scripts\run_streamlit.ps1
```

## 테스트

```bash
python -m unittest tests.test_dashboard_utils -v
python -m py_compile app.py src\dashboard_utils.py
```

## 현재 결과

저장된 모델 비교 결과는 다음과 같습니다.

- Baseline CNN accuracy: `0.89344`
- ResNet18 transfer-learning accuracy: `0.99297`

대표 시각화 산출물은 `outputs/figures/` 아래에 저장되어 있습니다. 여기에는 confusion matrix, EDA plot, 오분류 예시, Grad-CAM sample이 포함됩니다.
