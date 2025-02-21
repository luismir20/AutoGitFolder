import os
import io
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image

# Defines Constants
MODEL_PATH_RESNET = "resnet_ai_detector.pth"
MODEL_PATH_EFFICIENTNET = "efficientnet_ai_detector.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defines Model with Dynamic Feature Handling
class AIImageDetector(nn.Module):
    def __init__(self, base_model, feature_size):
        super(AIImageDetector, self).__init__()
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(feature_size, 1)  

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return torch.sigmoid(x)

# Loads Pretrained Models with Correct Feature Sizes
def load_models():
    """Load both ResNet and EfficientNet models for comparison."""
    
    # Loads ResNet18 Model
    if os.path.exists(MODEL_PATH_RESNET):
        print(f"✅ Loading trained ResNet18: {MODEL_PATH_RESNET}")
        base_model_resnet = resnet18(weights=None)
        feature_size_resnet = base_model_resnet.fc.in_features  # Extract correct feature size
        model_resnet = AIImageDetector(base_model_resnet, feature_size_resnet)
        model_resnet.load_state_dict(torch.load(MODEL_PATH_RESNET, map_location=device))
    else:
        print("Using default ResNet18.")
        base_model_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        feature_size_resnet = base_model_resnet.fc.in_features
        model_resnet = AIImageDetector(base_model_resnet, feature_size_resnet)

    # Load EfficientNet-B3 Model
    if os.path.exists(MODEL_PATH_EFFICIENTNET):
        print(f"✅ Loading trained EfficientNet-B3: {MODEL_PATH_EFFICIENTNET}")
        base_model_efficientnet = efficientnet_b3(weights=None)
        feature_size_efficientnet = base_model_efficientnet.classifier[1].in_features  # Extract correct feature size
        model_efficientnet = AIImageDetector(base_model_efficientnet, feature_size_efficientnet)
        model_efficientnet.load_state_dict(torch.load(MODEL_PATH_EFFICIENTNET, map_location=device))
    else:
        print("Using default EfficientNet-B3.")
        base_model_efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        feature_size_efficientnet = base_model_efficientnet.classifier[1].in_features
        model_efficientnet = AIImageDetector(base_model_efficientnet, feature_size_efficientnet)

    # Move models to device (GPU or CPU)
    model_resnet.to(device)
    model_resnet.eval()
    
    model_efficientnet.to(device)
    model_efficientnet.eval()

    return model_resnet, model_efficientnet

# ✅ Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def decode_base64_image(base64_string):
    """Decodes a Base64 string into a PIL image with forced RGB mode."""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        print(f"❌ Error decoding Base64 image: {e}")
        return None

def preprocess_image(image):
    """Converts a PIL image into a PyTorch tensor."""
    return transform(image).unsqueeze(0).to(device)  

def classify_image(base64_string):
    """Classifies an image as AI-generated or real using two models."""
    
    image = decode_base64_image(base64_string)
    if image is None:
        return {"error": "Invalid Image", "confidence": 0.0}  

    processed_image = preprocess_image(image)

    # ✅ Loads Both Models
    model_resnet, model_efficientnet = load_models()

    # ✅ Runs Inference for Both Models
    with torch.no_grad():
        resnet_output = model_resnet(processed_image).squeeze().item()
        efficientnet_output = model_efficientnet(processed_image).squeeze().item()

    # Apply Sigmoid to Convert Logits to Probabilities
    resnet_output = torch.sigmoid(torch.tensor(resnet_output)).item()
    efficientnet_output = torch.sigmoid(torch.tensor(efficientnet_output)).item()

    # Prepares Detailed Results as a Dictionary
    threshold = 0.65
    result_details = {
        "ResNet18": {
            "classification": "AI-generated" if resnet_output > threshold else "Real",
            "confidence": round(resnet_output, 4),
        },
        "EfficientNet-B3": {
            "classification": "AI-generated" if efficientnet_output > threshold else "Real",
            "confidence": round(efficientnet_output, 4),
        }
    }

    return result_details