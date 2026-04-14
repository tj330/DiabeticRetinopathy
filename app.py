import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODEL DEFINITION (UNCHANGED)
# ============================================================

class DRModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.backbone = models.efficientnet_b3(weights=None)

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = DRModel().to(device)
model.load_state_dict(torch.load("Aptos_EB3.pth", map_location=device))
model.eval()

# ============================================================
# IMAGE TRANSFORM (UNCHANGED)
# ============================================================

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================================
# PREPROCESSING FUNCTIONS (NEW)
# ============================================================

def apply_clahe(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def ben_graham_processing(image_np, sigmaX=10):
    return cv2.addWeighted(
        image_np, 4,
        cv2.GaussianBlur(image_np, (0,0), sigmaX),
        -4, 128
    )

# ============================================================
# GRAD-CAM IMPLEMENTATION (UNCHANGED)
# ============================================================

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer = self.model.backbone.features[-1]

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):

        output = self.model(input_tensor)
        class_idx = output.argmax(dim=1)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], device=input_tensor.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(cam, (300, 300), interpolation=cv2.INTER_CUBIC)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

# ============================================================
# CLASS LABELS
# ============================================================

classes = {
    0: "No Diabetic Retinopathy",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="DR Detection", layout="wide")

st.title("Diabetic Retinopathy Detection System")
st.markdown("Upload a retinal fundus image to predict DR severity level.")

# Sidebar preprocessing toggles
st.sidebar.header("Preprocessing Visualization")
show_clahe = st.sidebar.checkbox("Show CLAHE Enhancement")
show_ben = st.sidebar.checkbox("Show Ben Graham Processing")

uploaded_file = st.file_uploader("Upload Retinal Image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # ============================================================
    # SHOW PREPROCESSING VISUALIZATION
    # ============================================================

    if show_clahe or show_ben:

        st.markdown("## Preprocessing Comparison")

        cols = st.columns(3)

        with cols[0]:
            st.image(image_np, caption="Original", use_container_width=True)

        if show_clahe:
            clahe_img = apply_clahe(image_np)
            with cols[1]:
                st.image(clahe_img, caption="CLAHE Enhanced", use_container_width=True)

        if show_ben:
            ben_img = ben_graham_processing(image_np)
            with cols[2]:
                st.image(ben_img, caption="Ben Graham Processing", use_container_width=True)

    # ============================================================
    # MODEL PREDICTION (UNCHANGED)
    # ============================================================

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred = output.argmax(1).item()
        confidence = probabilities[0][pred].item()

    with col2:
        st.subheader("Prediction Result")
        st.success(classes[pred])
        st.write(f"Confidence: {confidence:.2%}")

    # ============================================================
    # GRAD-CAM VISUALIZATION (UNCHANGED)
    # ============================================================

    gradcam = GradCAM(model)
    heatmap = gradcam.generate(input_tensor)

    img_np = np.array(image.resize((300, 300))) / 255.0

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    threshold = 0.6
    mask = heatmap > threshold

    overlay = img_np.copy()
    overlay[mask] = (
        0.7 * img_np[mask] +
        0.3 * np.array([1, 0, 0])
    )
    overlay = np.clip(overlay, 0, 1)

    st.markdown("## Grad-CAM Visualization")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_np, caption="Original", use_container_width=True)

    with col2:
        st.image(heatmap_color, caption="Heatmap", use_container_width=True)

    with col3:
        st.image(overlay, caption="Clinical Overlay", use_container_width=True)
