import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import os

st.title("⚡ Pokemon Classifier")
st.write("이미지를 업로드하면 전이 학습된 CNN 모델이 포켓몬의 이름을 예측합니다.")

data_dir = './PokemonData'
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
num_classes = len(class_names)

# 모델 로드 (성능 비교해서 제일 좋았던 MobileNetV2 적용)
device = torch.device("cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

try:
    # MobileNet 가중치 파일 로드
    model.load_state_dict(torch.load('MobileNetV2_Finetune.pth', map_location=device))
    model.eval()
except Exception as e:
    st.error("모델 가중치 파일을 찾을 수 없습니다. 학습 코드를 먼저 실행해주세요.")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("포켓몬 이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', width=300)
    
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    st.subheader("🎯 예측 결과 (Top-5)")
    for i in range(5):
        name = class_names[top5_catid[i].item()]
        prob = top5_prob[i].item() * 100
        st.write(f"**{i+1}. {name}** ({prob:.2f}%)")
        st.progress(top5_prob[i].item())
