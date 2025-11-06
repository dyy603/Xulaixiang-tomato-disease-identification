import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import AlexNet
import sys
import torch.nn as nn
import json
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
            transforms.RandomResizedCrop(180, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def load_model(model_path, num_classes=5):
    model = AlexNet(num_classes).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(model, image_path, class_names=None):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  
        
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            confidence = F.softmax(outputs, dim=1)[0][predicted.item()].item() * 100
        
        if class_names:
            predicted_class = class_names[predicted.item()]
        else:
            predicted_class = predicted.item()
        
        return predicted_class, confidence
    
    except Exception as e:
        print(f"Error predicting image {image_path}: {str(e)}")
        return None, 0.0

def get_class_names(json_path='class_indices.json'):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            cla_dict = json.load(f)
        class_names = [cla_dict[str(i)] for i in range(len(cla_dict))]
        return class_names
    print(f"Warning: {json_path} not found, will return class indices instead of names")
    return None

def batch_predict(model, image_dir, class_names=None):
    results = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            pred_class, confidence = predict_image(model, img_path, class_names)
            results.append({
                'image': img_name,
                'predicted_class': pred_class,
                'confidence': f"{confidence:.2f}%"
            })
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Garlic Leaf Disease Identification Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--image_path', type=str, help='Path to a single image for prediction')
    parser.add_argument('--image_dir', type=str, help='Directory containing images for batch prediction')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of disease classes')
    
    args = parser.parse_args()
    
    if not args.image_path and not args.image_dir:
        print("Error: Either --image_path or --image_dir must be provided")
        sys.exit(1)
    
    class_names = get_class_names()
    if class_names:
        print(f"Class names: {class_names}")
    else:
        print("Warning: Could not load class names from dataset directory")
    
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args.num_classes)
    print("Model loaded successfully")
    
    if args.image_path:
        if os.path.exists(args.image_path):
            pred_class, confidence = predict_image(model, args.image_path, class_names)
            print(f"\nPrediction for {os.path.basename(args.image_path)}:")
            print(f"Class: {pred_class}")
            print(f"Confidence: {confidence:.2f}%")
        else:
            print(f"Error: Image file {args.image_path} not found")
    
    if args.image_dir:
        if os.path.isdir(args.image_dir):
            print(f"\nPerforming batch prediction on images in {args.image_dir}...")
            results = batch_predict(model, args.image_dir, class_names)
            
            print("\nBatch prediction results:")
            for result in results:
                print(f"{result['image']}: {result['predicted_class']} ({result['confidence']})")
        else:
            print(f"Error: Directory {args.image_dir} not found")