import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
from torchvision import transforms
from model import AlexNet
from torch.nn.functional import softmax

def specificity_score(y_true, y_pred, labels):
    """计算多分类特异度"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    fp = cm.sum(axis=0) - np.diag(cm)
    return tn / (tn + fp + 1e-7)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize((180, 180)), #224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载类别映射
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    class_labels = list(class_indict.values())

    # 初始化模型
    model = AlexNet(num_classes=len(class_labels)).to(device)
    weights_path = "./MLSA.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' does not exist."
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # 测试数据收集
    all_preds = []
    all_labels = []
    results = []
    # test_dir = "D:\\team\\alexnet learning\\deep-learning-for-image-processing-master-juanji-msa\\data_set\\car_data\\test"
    test_dir = "D:\\team\\alexnet learning\\deep-learning-for-image-processing-master-juanji-msa\\data_set\\fanqie\\test"
    assert os.path.exists(test_dir), f"folder: '{test_dir}' does not exist."

    # 初始化统计变量
    correct = 0
    total = 0
    supported_formats = ('.png', '.jpg', '.jpeg')

    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            # 跳过非图片文件
            if not img_name.lower().endswith(supported_formats):
                print(f"跳过不支持的文件: {img_name}")
                continue

            try:
                # 图像预处理
                img = Image.open(img_path).convert('RGB')
                img_tensor = data_transform(img).unsqueeze(0).to(device)

                # 推理预测
                with torch.no_grad():
                    output = model(img_tensor)
                    predict = softmax(output, dim=1)
                    confidence, pred_idx = torch.max(predict, 1)

                # 转换结果
                pred_label = class_indict[str(pred_idx.item())]
                true_label = class_name
                is_correct = (true_label == pred_label)

                # 更新统计
                total += 1
                if is_correct:
                    correct += 1
                current_acc = 100.0 * correct / total

                # 保存结果
                all_preds.append(pred_label)
                all_labels.append(true_label)
                results.append([img_name, true_label, pred_label, f"{confidence.item():.4f}"])

                # 输出详细信息
                print(f"图片: {img_name:15} | 真实: {true_label:10} | 预测: {pred_label:10} | "
                      f"置信度: {confidence.item():.4f} | 当前准确率: {current_acc:6.3f}%")

            except Exception as e:
                print(f"处理图片 {img_name} 时出错: {str(e)}")
                continue

    # 计算评估指标
    cm = confusion_matrix(all_labels, all_preds, labels=class_labels)

    # 计算每类指标
    class_metrics = {}
    specificity = specificity_score(all_labels, all_preds, class_labels)
    for i, label in enumerate(class_labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        class_metrics[label] = {
            'Accuracy': (tp + tn) / (tp + tn + fp + fn + 1e-7),
            'Recall': tp / (tp + fn + 1e-7),
            'Precision': tp / (tp + fp + 1e-7),
            'F1-score': 2 * tp / (2 * tp + fp + fn + 1e-7),
            'Specificity': specificity[i]
        }

    # 计算宏平均
    macro_avg = {
        'Accuracy': np.mean([m['Accuracy'] for m in class_metrics.values()]),
        'Recall': np.mean([m['Recall'] for m in class_metrics.values()]),
        'Precision': np.mean([m['Precision'] for m in class_metrics.values()]),
        'F1-score': np.mean([m['F1-score'] for m in class_metrics.values()]),
        'Specificity': np.mean([m['Specificity'] for m in class_metrics.values()])
    }

    # 计算加权平均
    weighted_avg = {
        'Accuracy': accuracy_score(all_labels, all_preds),
        'Recall': recall_score(all_labels, all_preds, average='weighted'),
        'Precision': precision_score(all_labels, all_preds, average='weighted'),
        'F1-score': f1_score(all_labels, all_preds, average='weighted'),
        'Specificity': np.mean(specificity)
    }

    # 输出结果
    print("\nConfusion Matrix:\n", cm)
    print("\nClass-wise Metrics:")
    for label, metrics in class_metrics.items():
        print(f"\n{label}:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    print("\nMacro Average Metrics:")
    for name, value in macro_avg.items():
        print(f"{name}: {value:.4f}")

    print("\nWeighted Average Metrics:")
    for name, value in weighted_avg.items():
        print(f"{name}: {value:.4f}")

    # 保存预测结果到CSV
    df = pd.DataFrame(results, columns=['Image', 'True Label', 'Predicted Label', 'Confidence'])
    df.to_csv('prediction_results.csv', index=False)
    print("\nPrediction results saved to prediction_results.csv")

    # 输出最终统计
    final_accuracy = 100.0 * correct / total if total != 0 else 0.0
    print("\n测试结果汇总:")
    print(f"总测试样本数: {total}")
    print(f"正确预测数: {correct}")
    print(f"最终测试准确率: {final_accuracy:.3f}%")


if __name__ == '__main__':
    main()
