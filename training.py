import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import os

# 데이터 준비, training data : validation data -> 8:2 분할
data_dir = './PokemonData'
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

full_dataset = datasets.ImageFolder(data_dir)
class_names = full_dataset.classes
num_classes = len(class_names)
targets = full_dataset.targets

train_idx, val_idx = train_test_split(list(range(len(targets))), test_size=0.2, stratify=targets, random_state=42)

image_datasets = {
    'train': Subset(datasets.ImageFolder(data_dir, transform=data_transforms['train']), train_idx),
    'val': Subset(datasets.ImageFolder(data_dir, transform=data_transforms['val']), val_idx)
}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=(x == 'train')) for x in ['train', 'val']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 4가지 설정 정의
def get_model(exp_name):
    if exp_name == "ResNet18_Finetune":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif exp_name == "ResNet18_FeatureExtract":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in model.parameters(): param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif exp_name == "MobileNetV2_Finetune":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif exp_name == "ResNet18_Scratch":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

experiments = ["ResNet18_Finetune", "ResNet18_FeatureExtract", "MobileNetV2_Finetune", "ResNet18_Scratch"]

# 학습&검증
for exp in experiments:
    print(f"\n{'='*40}\n🚀 Start Experiment: {exp}\n{'='*40}")
    model = get_model(exp)
    criterion = nn.CrossEntropyLoss()
    # Feature Extract 모드일 때는 학습 가능한 파라미터만 옵티마이저에 전달
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=0.001)
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} 완료")

    # 평가 (Test Precision, Recall 계산)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"\n📊 {exp} 최종 성능:")
    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    
    torch.save(model.state_dict(), f'{exp}.pth')