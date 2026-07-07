from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.dashboard_utils import (
    get_project_paths,
    gradcam_overlay,
    list_class_names,
    load_resnet18_checkpoint,
    model_comparison_records,
    predict_image,
    report_accuracy,
    safe_read_csv,
    sample_images_for_class,
    top_probabilities,
)


ROOT = Path(__file__).resolve().parent
PATHS = get_project_paths(ROOT)


st.set_page_config(
    page_title="NEU-DET Steel Defect Classifier",
    page_icon="N",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 1280px; }
    div[data-testid="stMetric"] {
        background: #f7f8fa;
        border: 1px solid #e3e6ea;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .section-note {
        color: #5c6670;
        font-size: 0.96rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_tables():
    bbox = safe_read_csv(PATHS.results_dir / "bbox_annotations.csv")
    crops = safe_read_csv(PATHS.results_dir / "crop_annotations.csv")
    comparison = safe_read_csv(PATHS.results_dir / "model_comparison.csv")
    baseline_report = safe_read_csv(
        PATHS.results_dir / "baseline_classification_report.csv",
        index_col=0,
    )
    resnet_report = safe_read_csv(
        PATHS.results_dir / "resnet18_classification_report.csv",
        index_col=0,
    )
    return bbox, crops, comparison, baseline_report, resnet_report


@st.cache_resource(show_spinner=False)
def load_model():
    model_path = PATHS.models_dir / "resnet18_best.pth"
    if not model_path.exists():
        return None, [], 224
    return load_resnet18_checkpoint(model_path)


def show_image(path: Path, caption: str | None = None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing figure: {path.relative_to(ROOT)}")


def percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def page_overview(bbox: pd.DataFrame, crops: pd.DataFrame, comparison: pd.DataFrame):
    st.title("NEU-DET Steel Defect Classification")
    st.markdown(
        """
        <p class="section-note">
        A portfolio dashboard for crop-based steel surface defect classification.
        The workflow starts with XML bounding-box annotations, converts defect
        regions into grayscale crops, compares a baseline CNN with a transfer-
        learned ResNet18, and explains ResNet18 decisions with Grad-CAM.
        </p>
        """,
        unsafe_allow_html=True,
    )

    records = model_comparison_records(comparison)
    best_model = max(records, key=lambda item: item.get("Accuracy", 0), default={})
    classes = sorted(bbox["class"].dropna().unique()) if "class" in bbox else []

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Defect classes", len(classes))
    col2.metric("Annotated objects", f"{len(bbox):,}" if not bbox.empty else "N/A")
    col3.metric("Crop samples", f"{len(crops):,}" if not crops.empty else "N/A")
    col4.metric("Best accuracy", percent(best_model.get("Accuracy")))

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Defect Classes")
        if classes:
            st.dataframe(pd.DataFrame({"class": classes}), hide_index=True, use_container_width=True)
        else:
            st.info("Class metadata is not available.")
    with right:
        st.subheader("Model Result")
        if records:
            st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)
        else:
            st.info("Model comparison CSV is not available.")

    st.subheader("Representative Annotated Samples")
    show_image(PATHS.figures_dir / "sample_images_with_bbox.png")


def page_eda(bbox: pd.DataFrame, crops: pd.DataFrame):
    st.title("EDA")
    st.markdown(
        '<p class="section-note">Distribution checks show whether each defect type is balanced and how object geometry varies by class.</p>',
        unsafe_allow_html=True,
    )

    if not bbox.empty and {"split", "class"}.issubset(bbox.columns):
        counts = bbox.groupby(["split", "class"]).size().reset_index(name="objects")
        st.dataframe(counts, hide_index=True, use_container_width=True)

    fig_cols = st.columns(2)
    with fig_cols[0]:
        show_image(PATHS.figures_dir / "class_object_count.png", "Object count by class")
        show_image(PATHS.figures_dir / "bbox_area_distribution.png", "Bounding-box area")
    with fig_cols[1]:
        show_image(PATHS.figures_dir / "class_crop_count.png", "Crop count by class")
        show_image(PATHS.figures_dir / "bbox_aspect_ratio_distribution.png", "Bounding-box aspect ratio")

    st.subheader("Image Quality Signals")
    q1, q2 = st.columns(2)
    with q1:
        show_image(PATHS.figures_dir / "class_brightness_distribution.png")
    with q2:
        show_image(PATHS.figures_dir / "class_contrast_distribution.png")

    st.subheader("Crop Preview")
    show_image(PATHS.figures_dir / "sample_crop_images.png")


def page_model_performance(
    comparison: pd.DataFrame,
    baseline_report: pd.DataFrame,
    resnet_report: pd.DataFrame,
):
    st.title("Model Performance")
    baseline_acc = report_accuracy(baseline_report)
    resnet_acc = report_accuracy(resnet_report)

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline CNN", percent(baseline_acc))
    col2.metric("ResNet18 Transfer", percent(resnet_acc))
    if baseline_acc is not None and resnet_acc is not None:
        col3.metric("Accuracy lift", f"{(resnet_acc - baseline_acc) * 100:.2f} pp")
    else:
        col3.metric("Accuracy lift", "N/A")

    if not comparison.empty:
        chart_df = comparison.rename(columns={"model": "Model", "accuracy": "Accuracy"})
        st.bar_chart(chart_df.set_index("Model")["Accuracy"])

    st.subheader("Classification Reports")
    tab1, tab2 = st.tabs(["Baseline CNN", "ResNet18"])
    with tab1:
        st.dataframe(baseline_report, use_container_width=True)
        show_image(PATHS.figures_dir / "baseline_confusion_matrix.png")
    with tab2:
        st.dataframe(resnet_report, use_container_width=True)
        show_image(PATHS.figures_dir / "resnet18_confusion_matrix.png")


def page_error_analysis():
    st.title("Error Analysis")
    st.markdown(
        '<p class="section-note">The baseline model mistakes reveal which defect textures overlap visually before transfer learning improves separation.</p>',
        unsafe_allow_html=True,
    )
    show_image(PATHS.figures_dir / "misclassified_samples.png")


def page_gradcam():
    st.title("Grad-CAM")
    st.markdown(
        '<p class="section-note">Grad-CAM highlights the crop regions that most influenced the ResNet18 prediction.</p>',
        unsafe_allow_html=True,
    )
    show_image(PATHS.figures_dir / "gradcam_samples.png")


def selected_validation_image() -> Image.Image | None:
    classes = list_class_names(PATHS.validation_crops_dir)
    if not classes:
        st.warning("Validation crop images are not available.")
        return None

    class_name = st.selectbox("Validation class", classes)
    samples = sample_images_for_class(PATHS.validation_crops_dir, class_name, limit=80)
    if not samples:
        st.warning("No images found for this class.")
        return None

    image_path = st.selectbox("Sample image", samples, format_func=lambda path: path.name)
    return Image.open(image_path).convert("RGB")


def uploaded_image() -> Image.Image | None:
    uploaded = st.file_uploader("Upload a defect crop image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is None:
        return None
    return Image.open(uploaded).convert("RGB")


def page_try_model():
    st.title("Try Model")
    st.markdown(
        '<p class="section-note">Run the trained ResNet18 on a validation crop or your own crop image, then inspect the Grad-CAM overlay.</p>',
        unsafe_allow_html=True,
    )

    model, class_names, img_size = load_model()
    if model is None:
        st.error("ResNet18 checkpoint is missing: outputs/models/resnet18_best.pth")
        return

    mode = st.radio("Input source", ["Validation sample", "Upload image"], horizontal=True)
    image = selected_validation_image() if mode == "Validation sample" else uploaded_image()
    if image is None:
        return

    pred_class, pred_score, probabilities = predict_image(model, image, class_names, img_size)
    target_idx = class_names.index(pred_class)
    overlay = gradcam_overlay(model, image, target_idx, img_size)
    prob_df = pd.DataFrame(
        top_probabilities(class_names, probabilities),
        columns=["Class", "Probability"],
    )

    col1, col2 = st.columns([0.9, 1.1])
    with col1:
        st.metric("Prediction", pred_class)
        st.metric("Confidence", percent(pred_score))
        st.image(image, caption="Input", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
        st.bar_chart(prob_df.set_index("Class")["Probability"])


def main():
    bbox, crops, comparison, baseline_report, resnet_report = load_tables()

    pages = {
        "Overview": lambda: page_overview(bbox, crops, comparison),
        "EDA": lambda: page_eda(bbox, crops),
        "Model Performance": lambda: page_model_performance(comparison, baseline_report, resnet_report),
        "Error Analysis": page_error_analysis,
        "Grad-CAM": page_gradcam,
        "Try Model": page_try_model,
    }

    st.sidebar.title("NEU-DET Dashboard")
    st.sidebar.caption("Notebook workflow packaged as a portfolio demo")
    page_name = st.sidebar.radio("Section", list(pages.keys()))
    st.sidebar.divider()
    st.sidebar.caption("Run locally with `streamlit run app.py`.")

    pages[page_name]()


if __name__ == "__main__":
    main()
