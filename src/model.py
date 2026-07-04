# -*- coding: utf-8 -*-
"""
NEU-DET 철강 결함 분류를 위한 PyTorch CNN 모델 구조(BatchNorm 및 Dropout 적용 고도화), 
커스텀 데이터셋 로더, 모델 훈련/평가 엔진, 디스크 입출력 체크포인트 함수, 
그리고 3대 XAI 기법(Grad-CAM, Saliency Map, Feature Map) 모듈을 정의하는 스크립트입니다.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.models as models

class NEUDataset(Dataset):
    """
    NEU-DET 철강 결함 이미지 데이터를 읽어와 64x64 크기의 
    그레이스케일 텐서(0~1 정규화)로 변환하고 라벨을 매핑하는 PyTorch Dataset 클래스입니다.
    """
    def __init__(self, image_names, class_prefixes, class_list, images_dir):
        """
        NEUDataset 초기화 함수.
        
        인자:
            image_names (list): 이미지 파일명 리스트
            class_prefixes (list): 이미지 파일명에서 잘라낸 결함 클래스 접두사 리스트
            class_list (list): 전체 고유 결함 클래스 정렬 리스트
            images_dir (str): 이미지 파일들이 위치한 디렉토리 절대경로
        """
        self.image_names = image_names
        self.class_prefixes = class_prefixes
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}
        self.images_dir = images_dir
        
    def __len__(self):
        """데이터셋에 등록된 전체 샘플 개수를 반환합니다."""
        return len(self.image_names)
        
    def __getitem__(self, idx):
        """
        인덱스에 해당하는 텐서 이미지와 라벨 인덱스 및 이미지 이름을 로드합니다.
        
        인자:
            idx (int): 데이터 샘플의 인덱스
        반환:
            img_tensor (Tensor): (1, 64, 64) 크기의 정규화된 이미지 텐서
            label (int): 결함 클래스 인덱스 정수
            img_name (str): 원본 이미지 파일명
        """
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # 이미지 로드 및 그레이스케일 변환, 64x64 축소 리사이즈
        img = Image.open(img_path).convert("L").resize((64, 64))
        img_np = np.array(img, dtype=np.float32) / 255.0  # 픽셀값 0~1 사이로 정규화
        img_tensor = torch.tensor(img_np).unsqueeze(0)    # (1, 64, 64) 채널 차원 추가
        
        prefix = self.class_prefixes[idx]
        label = self.class_to_idx[prefix]
        
        return img_tensor, label, img_name


class SimpleCNN(nn.Module):
    """
    배치 정규화(BatchNorm2d) 및 드롭아웃(Dropout)을 포함한 3층의 2D 합성곱 풀링 블록과 
    전결합 레이어로 설계된 고성능 철강 표면 결함 분류용 CNN 모델입니다.
    """
    def __init__(self, num_classes=6):
        """
        SimpleCNN 네트워크 레이어들을 초기화합니다.
        
        인자:
            num_classes (int): 분류할 총 클래스(결함 종류) 개수
        """
        super(SimpleCNN, self).__init__()
        # 1. Conv Block 1: 1ch -> 32ch (3x3 Kernel, Padding 1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        
        # 2. Conv Block 2: 32ch -> 64ch (3x3 Kernel, Padding 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # 3. Conv Block 3: 64ch -> 128ch (3x3 Kernel, Padding 1) - XAI 타겟 최종 Conv 층
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # 최종 특징 맵 크기: 128ch * 8 * 8 = 8192차원
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """
        모델의 순전파(Forward propagation) 계산 흐름을 정의합니다.
        
        인자:
            x (Tensor): (batch_size, 1, 64, 64) 크기의 입력 텐서
        반환:
            x (Tensor): (batch_size, num_classes) 크기의 로짓(Logit) 출력 텐서
        """
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 1차원 텐서 플랫화 (Batch, 8192)
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully Connected 1 -> Dropout -> Fully Connected 2
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) 기법을 활용하여
    최종 합성곱 레이어의 특징 맵과 그래디언트 훅을 수집해 결함 영역의 활성도를 추적하는 클래스입니다.
    """
    def __init__(self, model, target_layer):
        """
        Grad-CAM 시각화 엔진 초기화 및 포워드/백워드 훅 등록.
        
        인자:
            model (nn.Module): 훈련 완료된 PyTorch 모델
            target_layer (nn.Module): 분석 대상이 될 최종 합성곱 레이어 (예: model.conv3)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 레이어 포워드/백워드 훅 등록
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        """포워드 단계에서 대상 레이어의 피처 맵 활성화 값을 캐싱합니다."""
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        """역전파 단계에서 대상 레이어의 피처 맵에 대한 가중치 기울기(그래디언트)를 캐싱합니다."""
        self.gradients = grad_output[0].detach()
        
    def __call__(self, x, class_idx=None):
        """
        입력 이미지 텐서에 대한 Grad-CAM 활성화 히트맵을 생성합니다.
        
        인자:
            x (Tensor): (1, 1, 64, 64) 크기의 단일 배치 입력 텐서
            class_idx (int, optional): 히트맵을 추출할 타겟 결함 클래스 인덱스. 지정 안 할 시 최고 예측값 사용.
        반환:
            cam (ndarray): 2차원 Grad-CAM 가중치 맵 (가로 8 x 세로 8 국소 좌표 0~1 정규화값)
            class_idx (int): 타겟 결함 클래스 인덱스
        """
        self.model.eval()
        output = self.model(x)
        
        # 지정된 타겟 클래스가 없을 시, 가장 큰 스코어를 갖는 예측값 인덱스로 타게팅
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        score = output[0, class_idx]
        # 해당 스코어 역전파 진행
        score.backward()
        
        # 훅으로 캡처한 특징 맵과 그래디언트 추출 (shape: (128, 8, 8))
        gradients = self.gradients[0]      
        activations = self.activations[0]  
        
        # 픽셀 영역에 대해 글로벌 에버리지 풀링(Global Average Pooling) 적용하여 채널별 가중치 산출
        weights = torch.mean(gradients, dim=(1, 2))  # shape: (128,)
        
        # 피처 맵 각 채널과 가중치 곱 결합
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # 0보다 큰 양의 기여만 남기는 ReLU 연산 수행
        cam = torch.clamp(cam, min=0)
        
        # 0 ~ 1 정규화
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = cam - cam_min
            
        return cam.numpy(), class_idx
        
    def release(self):
        """메모리 누수 방지를 위해 등록했던 포워드/백워드 훅을 해제합니다."""
        self.forward_hook.remove()
        self.backward_hook.remove()


def generate_saliency(model, image_tensor, target_class_idx):
    """
    입력 이미지의 픽셀값 극소 변화에 따른 예측 스코어 변화량(기울기 절대값)을 산출하여 
    픽셀 수준 해상도의 기여도 Saliency Map을 생성합니다. (XAI 기법 2)
    
    인자:
        model (nn.Module): 훈련 완료된 PyTorch 모델
        image_tensor (Tensor): (1, 1, 64, 64) 크기의 단일 입력 텐서
        target_class_idx (int): 타겟 결함 클래스 인덱스
    반환:
        saliency (ndarray): 2차원 Saliency Map (가로 64 x 세로 64 크기, 0~1 정규화값)
    """
    model.eval()
    # 입력 텐서의 그래디언트 역전파를 추적하도록 요구 설정
    image_tensor.requires_grad_()
    
    output = model(image_tensor)
    score = output[0, target_class_idx]
    
    model.zero_grad()
    score.backward()
    
    # 입력 이미지 기울기의 절대값을 추출하여 이미지 형상의 기여도 파악
    saliency = image_tensor.grad.data.abs()
    saliency = saliency[0, 0].numpy()  # 배치 및 채널 차원 축소하여 (64, 64) 2D 맵 획득
    
    # 0 ~ 1 정규화
    saliency_min, saliency_max = saliency.min(), saliency.max()
    if saliency_max - saliency_min > 1e-8:
        saliency = (saliency - saliency_min) / (saliency_max - saliency_min)
    else:
        saliency = saliency - saliency_min
        
    return saliency


def get_feature_maps(model, image_tensor):
    """
    첫 번째 합성곱 레이어(conv1)를 통과한 출력 피처 맵들의 채널별 활성화 맵을 가져옵니다. (XAI 기법 3)
    
    인자:
        model (nn.Module): PyTorch 모델
        image_tensor (Tensor): (1, 1, 64, 64) 크기의 단일 입력 텐서
    반환:
        feature_maps (ndarray): 첫 번째 레이어가 활성화한 32개 필터 맵 (shape: (32, 64, 64))
    """
    model.eval()
    with torch.no_grad():
        # conv1 통과 후 BatchNorm 및 ReLU 적용 후의 활성화 상태 추출
        x = model.conv1(image_tensor)
        x = F.relu(model.bn1(x))
        feature_maps = x[0].cpu().numpy()  # batch 차원 해제하여 (32, 64, 64) numpy 변환
    return feature_maps


def save_checkpoint(model, checkpoint_dir, eval_metrics, test_predictions_df):
    """
    학습이 완료된 모델 가중치 파일(.pth)과 평가지표 및 예측 히스토리가 기록된 
    메타데이터 파일(.json)을 지정된 폴더 경로에 저장합니다.
    
    인자:
        model (nn.Module): 훈련 완료된 PyTorch 모델
        checkpoint_dir (str): 파일들을 저장할 로컬 디렉토리 경로
        eval_metrics (dict): 5대 평가지표 및 혼동행렬 데이터가 포함된 딕셔너리
        test_predictions_df (DataFrame): 테스트셋 전체 이미지별 개별 예측 레코드 판정 데이터프레임
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    # 1. 모델 가중치 state_dict 저장
    model_path = os.path.join(checkpoint_dir, "improved_cnn.pth")
    torch.save(model.state_dict(), model_path)
    
    # 2. 평가지표 및 테스트 예측 판정 데이터프레임을 JSON 직렬화할 수 있게 가공하여 저장
    # numpy array(혼동행렬 등)를 list로 변환하여 json 쓰기 에러 예방
    processed_metrics = eval_metrics.copy()
    if "conf_matrix" in processed_metrics:
        processed_metrics["conf_matrix"] = processed_metrics["conf_matrix"].tolist()
        
    # 데이터프레임을 JSON 친화적인 사전형태로 직렬화
    predictions_dict = test_predictions_df.to_dict(orient="records")
    
    # prob_dist 등 numpy ndarray 타입의 데이터들을 list로 재인코딩
    for row in predictions_dict:
        if "prob_dist" in row and isinstance(row["prob_dist"], np.ndarray):
            row["prob_dist"] = row["prob_dist"].tolist()
            
    meta_data = {
        "eval_metrics": processed_metrics,
        "test_predictions": predictions_dict
    }
    
    meta_path = os.path.join(checkpoint_dir, "train_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)


def load_checkpoint(model, checkpoint_dir):
    """
    로컬 디렉토리에서 기존 학습 완료된 가중치 파일과 평가지표 메타데이터 파일을 감지하여 
    모델에 이식하고 복원용 사전 데이터를 반환합니다.
    
    인자:
        model (nn.Module): 가중치를 주입받을 모델 인스턴스
        checkpoint_dir (str): 가중치와 메타데이터가 보관된 디렉토리 경로
    반환:
        dict: 복원된 eval_metrics와 test_predictions 데이터프레임이 포함된 사전 (체크포인트 없을 시 None 반환)
    """
    model_path = os.path.join(checkpoint_dir, "improved_cnn.pth")
    meta_path = os.path.join(checkpoint_dir, "train_meta.json")
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None
        
    try:
        # 모델 state_dict 주입 (CPU 디바이스 맵핑 강제 지정으로 안전성 보장)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # 메타데이터 JSON 파싱
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            
        eval_metrics = meta_data["eval_metrics"]
        # 혼동행렬 list 구조를 다시 numpy array로 환원
        if "conf_matrix" in eval_metrics:
            eval_metrics["conf_matrix"] = np.array(eval_metrics["conf_matrix"])
            
        test_pred_list = meta_data["test_predictions"]
        # prob_dist 리스트 구조를 numpy array로 다시 래핑
        for row in test_pred_list:
            if "prob_dist" in row:
                row["prob_dist"] = np.array(row["prob_dist"])
                
        test_predictions_df = pd.DataFrame(test_pred_list)
        
        return {
            "eval_metrics": eval_metrics,
            "test_predictions": test_predictions_df
        }
    except Exception:
        # 로드 중 데이터 정합성 등의 에러 발생 시 안전하게 None 반환하여 재학습 유도
        return None


def evaluate_model(model, dataloader, criterion, classes):
    """
    주어진 DataLoader에 대해 모델 예측을 평가하고 5대 핵심 평가지표와 혼동행렬을 반환합니다.
    
    인자:
        model (nn.Module): 검증할 PyTorch 모델
        dataloader (DataLoader): 평가용 데이터 로더
        criterion (nn.Module): 손실 기준 함수
        classes (list): 클래스 라벨 명칭 리스트
    반환:
        dict: 5대 평가지표 및 세부 예측 기록 데이터가 저장된 딕셔너리
    """
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    all_prob_dist = []
    test_running_loss = 0.0
    image_names_list = []
    
    with torch.no_grad():
        for test_images, test_labels, test_names in dataloader:
            test_outputs = model(test_images)
            test_loss = criterion(test_outputs, test_labels)
            test_running_loss += test_loss.item() * test_images.size(0)
            
            probs = F.softmax(test_outputs, dim=1)
            _, test_predicted = torch.max(test_outputs.data, 1)
            
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()
            
            all_preds.extend(test_predicted.numpy())
            all_targets.extend(test_labels.numpy())
            all_prob_dist.extend(probs.numpy())
            image_names_list.extend(test_names)
            
    final_test_loss = test_running_loss / len(dataloader.dataset)
    final_test_acc = test_correct / test_total
    
    # 클래스 개별 지표 수동 연산 (Weighted Average)
    num_classes = len(classes)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        conf_matrix[t, p] += 1
        
    class_precisions = np.zeros(num_classes)
    class_recalls = np.zeros(num_classes)
    class_f1s = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        
        class_precisions[i] = tp / (tp + fp + 1e-6)
        class_recalls[i] = tp / (tp + fn + 1e-6)
        class_f1s[i] = 2 * (class_precisions[i] * class_recalls[i]) / (class_precisions[i] + class_recalls[i] + 1e-6)
        
    class_counts_test = np.array([conf_matrix[i, :].sum() for i in range(num_classes)])
    total_support = class_counts_test.sum()
    
    final_precision = (class_precisions * class_counts_test).sum() / (total_support + 1e-6)
    final_recall = (class_recalls * class_counts_test).sum() / (total_support + 1e-6)
    final_f1 = (class_f1s * class_counts_test).sum() / (total_support + 1e-6)
    
    return {
        "accuracy": final_test_acc,
        "precision": final_precision,
        "recall": final_recall,
        "f1_score": final_f1,
        "test_loss": final_test_loss,
        "conf_matrix": conf_matrix,
        "predictions_df_data": {
            "image_name": image_names_list,
            "target_idx": all_targets,
            "target_label": [classes[t] for t in all_targets],
            "pred_idx": all_preds,
            "pred_label": [classes[p] for p in all_preds],
            "is_correct": [t == p for t, p in zip(all_targets, all_preds)],
            "prob_dist": all_prob_dist
        }
    }


class TransferDataset(Dataset):
    """
    사전 학습된 허깅페이스 MobileNetV2 모델 규격에 맞춰 
    그레이스케일 이미지를 3채널 RGB로 변환하고 224x224 리사이즈 및 ImageNet 정규화를 처리하는 데이터셋입니다.
    """
    def __init__(self, image_names, class_prefixes, class_list, images_dir):
        """
        TransferDataset 초기화 함수.
        
        인자:
            image_names (list): 이미지 파일명 리스트
            class_prefixes (list): 이미지 파일명 접두사 리스트
            class_list (list): 전체 고유 결함 클래스 정렬 리스트
            images_dir (str): 이미지 파일 디렉토리 경로
        """
        self.image_names = image_names
        self.class_prefixes = class_prefixes
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}
        self.images_dir = images_dir
        
    def __len__(self):
        """데이터셋 샘플 개수를 반환합니다."""
        return len(self.image_names)
        
    def __getitem__(self, idx):
        """
        인덱스에 해당하는 이미지를 (3, 224, 224) 텐서로 변환 및 정규화하여 라벨과 함께 로드합니다.
        
        인자:
            idx (int): 인덱스
        반환:
            img_tensor (Tensor): ImageNet 정규화가 적용된 (3, 224, 224) 이미지 텐서
            label (int): 결함 클래스 인덱스
            img_name (str): 이미지 파일명
        """
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # 3채널 RGB 변환 및 224x224 리사이즈
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_np = np.array(img, dtype=np.float32) / 255.0  # 0~1 정규화
        
        # HWC -> CHW 형태 변환
        img_tensor = torch.tensor(img_np).permute(2, 0, 1)
        
        # ImageNet 기준 평균과 표준편차 정규화
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        prefix = self.class_prefixes[idx]
        label = self.class_to_idx[prefix]
        
        return img_tensor, label, img_name


def get_transfer_model(num_classes=6, freeze_backbone=True):
    """
    torchvision으로부터 경량 MobileNetV2 모델 구조를 로드하고,
    네트워크 차단 환경(403 Forbidden)을 극복하기 위해 오프라인(weights=None)으로 모델 구조만 생성하여 반환합니다.
    
    인자:
        num_classes (int): 분류할 총 클래스 수 (기본값 6)
        freeze_backbone (bool): True일 경우 특징 추출기 레이어들을 동결 (기본값 True)
    반환:
        model (nn.Module): 분류층이 재조정된 파이토치 모델 객체
    """
    # 네트워크 연결 오류(403 Forbidden) 방지를 위해 weights=None으로 오프라인 로드 수행
    model = models.mobilenet_v2(weights=None)
    
    if freeze_backbone:
        # 백본 가중치 파라미터들의 그래디언트 역전파를 동결하여 학습 연산 최소화
        for param in model.parameters():
            param.requires_grad = False
            
    # 마지막 분류층(classifier)의 선형 레이어를 대체 (입력 1280 -> 클래스 수)
    # 이 교체 작업으로 교체된 레이어의 requires_grad는 자동으로 True가 됩니다.
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model


def save_transfer_checkpoint(model, checkpoint_dir, eval_metrics, test_predictions_df):
    """
    전이학습 완료 시 경량 모델의 가중치(.pth)와 성능 평가 JSON 메타데이터를
    전이학습 전용 하위 디렉토리에 격리 보관합니다.
    
    인자:
        model (nn.Module): 전이학습 완료된 모델 객체
        checkpoint_dir (str): 저장할 디렉토리 경로
        eval_metrics (dict): 5대 성능 지표
        test_predictions_df (DataFrame): 테스트 데이터 예측 상세 레코드
    """
    target_dir = os.path.join(checkpoint_dir, "transfer")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        
    model_path = os.path.join(target_dir, "mobilenet_v2.pth")
    torch.save(model.state_dict(), model_path)
    
    processed_metrics = eval_metrics.copy()
    if "conf_matrix" in processed_metrics:
        processed_metrics["conf_matrix"] = processed_metrics["conf_matrix"].tolist()
        
    predictions_dict = test_predictions_df.to_dict(orient="records")
    for row in predictions_dict:
        if "prob_dist" in row and isinstance(row["prob_dist"], np.ndarray):
            row["prob_dist"] = row["prob_dist"].tolist()
            
    meta_data = {
        "eval_metrics": processed_metrics,
        "test_predictions": predictions_dict
    }
    
    meta_path = os.path.join(target_dir, "transfer_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)


def load_transfer_checkpoint(model, checkpoint_dir):
    """
    저장된 전이학습 경량 모델의 가중치를 복원하고 평가 메타데이터를 파싱하여 복원 사전을 반환합니다.
    
    인자:
        model (nn.Module): 가중치를 주입할 모델 객체
        checkpoint_dir (str): 가중치와 메타데이터가 보관된 디렉토리 경로
    반환:
        dict: 복원된 평가 정보 딕셔너리 (실패 시 None)
    """
    target_dir = os.path.join(checkpoint_dir, "transfer")
    model_path = os.path.join(target_dir, "mobilenet_v2.pth")
    meta_path = os.path.join(target_dir, "transfer_meta.json")
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            
        eval_metrics = meta_data["eval_metrics"]
        if "conf_matrix" in eval_metrics:
            eval_metrics["conf_matrix"] = np.array(eval_metrics["conf_matrix"])
            
        test_pred_list = meta_data["test_predictions"]
        for row in test_pred_list:
            if "prob_dist" in row:
                row["prob_dist"] = np.array(row["prob_dist"])
                
        test_predictions_df = pd.DataFrame(test_pred_list)
        
        return {
            "eval_metrics": eval_metrics,
            "test_predictions": test_predictions_df
        }
    except Exception:
        return None


def evaluate_transfer_model(model, dataloader, criterion, classes):
    """
    전이학습된 경량 모델(Hugging Face logits 구조)의 예측 성능을 테스트 셋에 대해 검증하고 5대 평가지표를 연산합니다.
    
    인자:
        model (nn.Module): 검증할 전이학습 모델
        dataloader (DataLoader): 평가용 데이터 로더
        criterion (nn.Module): 로스 함수
        classes (list): 클래스 라벨 정렬 리스트
    반환:
        dict: 5대 평가지표 및 세부 예측 기록 딕셔너리
    """
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    all_prob_dist = []
    test_running_loss = 0.0
    image_names_list = []
    
    with torch.no_grad():
        for test_images, test_labels, test_names in dataloader:
            logits = model(test_images)
            
            loss = criterion(logits, test_labels)
            test_running_loss += loss.item() * test_images.size(0)
            
            probs = F.softmax(logits, dim=1)
            _, test_predicted = torch.max(logits.data, 1)
            
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()
            
            all_preds.extend(test_predicted.cpu().numpy())
            all_targets.extend(test_labels.cpu().numpy())
            all_prob_dist.extend(probs.cpu().numpy())
            image_names_list.extend(test_names)
            
    final_test_loss = test_running_loss / len(dataloader.dataset)
    final_test_acc = test_correct / test_total
    
    num_classes = len(classes)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        conf_matrix[t, p] += 1
        
    class_precisions = np.zeros(num_classes)
    class_recalls = np.zeros(num_classes)
    class_f1s = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        
        class_precisions[i] = tp / (tp + fp + 1e-6)
        class_recalls[i] = tp / (tp + fn + 1e-6)
        class_f1s[i] = 2 * (class_precisions[i] * class_recalls[i]) / (class_precisions[i] + class_recalls[i] + 1e-6)
        
    class_counts_test = np.array([conf_matrix[i, :].sum() for i in range(num_classes)])
    total_support = class_counts_test.sum()
    
    final_precision = (class_precisions * class_counts_test).sum() / (total_support + 1e-6)
    final_recall = (class_recalls * class_counts_test).sum() / (total_support + 1e-6)
    final_f1 = (class_f1s * class_counts_test).sum() / (total_support + 1e-6)
    
    return {
        "accuracy": final_test_acc,
        "precision": final_precision,
        "recall": final_recall,
        "f1_score": final_f1,
        "test_loss": final_test_loss,
        "conf_matrix": conf_matrix,
        "class_precisions": class_precisions.tolist() if isinstance(class_precisions, np.ndarray) else class_precisions,
        "class_recalls": class_recalls.tolist() if isinstance(class_recalls, np.ndarray) else class_recalls,
        "class_f1s": class_f1s.tolist() if isinstance(class_f1s, np.ndarray) else class_f1s,
        "class_supports": class_counts_test.tolist() if isinstance(class_counts_test, np.ndarray) else class_counts_test,
        "predictions_df_data": {
            "image_name": image_names_list,
            "target_idx": all_targets,
            "target_label": [classes[t] for t in all_targets],
            "pred_idx": all_preds,
            "pred_label": [classes[p] for p in all_preds],
            "is_correct": [t == p for t, p in zip(all_targets, all_preds)],
            "prob_dist": all_prob_dist
        }
    }
