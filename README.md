# NEU-DET Steel Defect Classification

Portfolio project for classifying steel surface defects with PyTorch and presenting the full workflow in a Streamlit dashboard.

The project uses NEU-DET annotation XML files to crop defect regions, trains a baseline CNN and a ResNet18 transfer-learning model, compares their performance, and visualizes ResNet18 decisions with Grad-CAM.

## Defect Classes

- `crazing`
- `inclusion`
- `patches`
- `pitted_surface`
- `rolled-in_scale`
- `scratches`

## Project Structure

```text
NEU-DET/
├── app.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_resnet18_transfer.ipynb
│   └── 04_gradcam.ipynb
├── outputs/
│   ├── figures/
│   └── results/
├── src/
│   └── dashboard_utils.py
├── tests/
│   └── test_dashboard_utils.py
└── requirements.txt
```

Large local artifacts are intentionally excluded from Git:

- `train/`
- `validation/`
- `crops/`
- `outputs/models/`
- `*.pth`

## Workflow

1. `01_eda.ipynb`
   - Parses XML annotations.
   - Builds `outputs/results/bbox_annotations.csv`.
   - Creates EDA figures for class counts, bounding-box geometry, brightness, and contrast.
   - Generates cropped defect regions under `crops/`.

2. `02_baseline_cnn.ipynb`
   - Trains a simple grayscale CNN on cropped defect images.
   - Saves baseline metrics and confusion matrix.

3. `03_resnet18_transfer.ipynb`
   - Fine-tunes ResNet18 with grayscale crops converted to 3 channels.
   - Saves model comparison results and ResNet18 metrics.

4. `04_gradcam.ipynb`
   - Loads the best ResNet18 checkpoint.
   - Generates Grad-CAM visualizations for validation crops.

## Dashboard

The Streamlit dashboard packages the notebook workflow as an interactive portfolio demo.

Sections:

- Overview
- EDA
- Model Performance
- Error Analysis
- Grad-CAM
- Try Model

`Try Model` supports validation crop selection and image upload when `outputs/models/resnet18_best.pth` exists locally. If the checkpoint is missing, the rest of the dashboard still works from saved figures and CSV files.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Dashboard

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

On Windows, you can also run:

```powershell
.\scripts\run_streamlit.ps1
```

## Tests

```bash
python -m unittest tests.test_dashboard_utils -v
python -m py_compile app.py src\dashboard_utils.py
```

## Current Results

The saved model comparison artifact reports:

- Baseline CNN accuracy: `0.893443`
- ResNet18 transfer-learning accuracy: `0.992974`

Representative visual artifacts are stored under `outputs/figures/`, including confusion matrices, EDA plots, misclassification examples, and Grad-CAM samples.
