import os
import json
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
from torchvision import transforms
from model import AlexNet


def specificity_score(y_true, y_pred, labels):
    """计算多分类特异度"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    fp = cm.sum(axis=0) - np.diag(cm)
    return np.mean(tn / (tn + fp + 1e-7))


def category_accuracy(cm):
    """计算每一类别的预测准确率"""
    total = cm.sum(axis=1, keepdims=True)
    return np.diag(cm) / total.flatten()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预处理（与train.py保持一致）
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 加载类别映射
    with open('./class_indices.json', 'r') as f:
        class_indict = json.load(f)
    class_labels = list(class_indict.values())

    # 初始化模型
    model = AlexNet(num_classes=len(class_labels)).to(device)
    model.load_state_dict(torch.load('./AlexNet.pth', map_location=device))
    model.eval()

    # 测试数据收集
    all_preds = []
    all_labels = []
    test_dir = "D:\\team\\alexnet learning\\deep-learning-for-image-processing-master-juanji-msa\\data_set\\fanqie\\test"

    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = data_transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)  # 计算概率
                    confidence, pred_idx = torch.max(probabilities, 1)  # 获取置信度和索引
                    confidence = confidence.item()  # 转换为Python数值

                # 记录预测结果和真实标签
                predicted_class = class_indict[str(pred_idx.item())]
                all_preds.append(predicted_class)
                all_labels.append(class_name)

                # 输出每张图片的置信率
                print(f"Image: {img_path}")
                print(f"True Label: {class_name}")
                print(f"Predicted Label: {predicted_class}")
                print(f"Confidence: {confidence:.4f}\n")

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

    # 计算评估指标
    cm = confusion_matrix(all_labels, all_preds, labels=class_labels)
    metrics = {
        'Accuracy': accuracy_score(all_labels, all_preds),
        'Recall': recall_score(all_labels, all_preds, average='weighted'),
        'Precision': precision_score(all_labels, all_preds, average='weighted'),
        'F1': f1_score(all_labels, all_preds, average='weighted'),
        'Specificity': specificity_score(all_labels, all_preds, class_labels)
    }

    # 计算并添加类别准确率
    cat_acc = category_accuracy(cm)
    for i, acc in enumerate(cat_acc):
        metrics[f'{class_labels[i]}_Accuracy'] = round(acc, 4)

    # 输出结果
    print("Confusion Matrix:\n", cm)
    print("\nEvaluation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == '__main__':
    main()