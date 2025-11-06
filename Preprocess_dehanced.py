import os
import torch
import random
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2


class EnhancedPreprocessor:
    def __init__(self, img_size=(224, 224)):
        self.train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.Lambda(lambda img: self.random_brightness(img, factor_range=(0.3, 1.7))),
            transforms.Lambda(lambda img: self.random_contrast(img, factor_range=(0.3, 1.7))),
            transforms.Lambda(lambda img: self.add_random_noise(img, prob=0.4)),
            transforms.Lambda(lambda img: self.add_random_shadow(img, prob=0.5)),
            transforms.ColorJitter(
                saturation=0.3,
                hue=0.15
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.infer_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def process_train(self, img):
        return self.train_transform(img)

    def process_infer(self, img):
        return self.infer_transform(img)

    @staticmethod
    def random_brightness(img, factor_range=(0.5, 1.5)):
        factor = random.uniform(*factor_range)
        enhancer = transforms.ColorJitter(brightness=factor)
        return enhancer(img)

    @staticmethod
    def random_contrast(img, factor_range=(0.5, 1.5)):
        factor = random.uniform(*factor_range)
        enhancer = transforms.ColorJitter(contrast=factor)
        return enhancer(img)

    @staticmethod
    def add_random_noise(img, prob=0.5, noise_level=(5, 20)):
        if random.random() > prob:
            return img

        img_np = np.array(img)
        h, w, c = img_np.shape

        noise_type = random.choice(['gaussian', 'salt_pepper'])

        if noise_type == 'gaussian':
            mean = 0
            var = random.randint(*noise_level)
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (h, w, c))
            noisy = img_np + gauss
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        else:
            s_vs_p = 0.5
            amount = random.uniform(0.001, 0.02)
            noisy = np.copy(img_np)

            num_salt = np.ceil(amount * img_np.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_np.shape]
            noisy[coords] = 255

            num_pepper = np.ceil(amount * img_np.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_np.shape]
            noisy[coords] = 0

        return Image.fromarray(noisy)

    @staticmethod
    def add_random_shadow(img, prob=0.5, num_shadows=1):
        if random.random() > prob:
            return img

        img_np = np.array(img, dtype=np.float32) / 255.0
        h, w, c = img_np.shape

        num = random.randint(1, num_shadows)

        for _ in range(num):
            points_num = random.randint(3, 6)
            points = []

            for _ in range(points_num):
                x = random.randint(0, w)
                y = random.randint(0, h)
                points.append((x, y))

            mask = np.zeros((h, w), dtype=np.float32)
            polygon = np.array(points, np.int32)
            polygon = polygon.reshape((-1, 1, 2))

            cv2.fillPoly(mask, [polygon], 1)

            shadow_strength = random.uniform(0.3, 0.7)

            for channel in range(c):
                img_np[:, :, channel] = np.where(
                    mask == 1,
                    img_np[:, :, channel] * shadow_strength,
                    img_np[:, :, channel]
                )

        img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)


def load_example_images(data_dir="example_data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        for i in range(3):
            img = np.ones((256, 256, 3), dtype=np.uint8) * 240
            cv2.circle(img, (128, 128), 80, (34, 139, 34), -1)
            for _ in range(50):
                x1, y1 = random.randint(50, 200), random.randint(50, 200)
                x2, y2 = random.randint(50, 200), random.randint(50, 200)
                cv2.line(img, (x1, y1), (x2, y2), (20, 100, 20), 1)

            Image.fromarray(img).save(os.path.join(data_dir, f"leaf_example_{i}.png"))

    images = []
    for fname in os.listdir(data_dir):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(data_dir, fname)).convert('RGB')
            images.append(img)
    return images


def visualize_enhancements(images, preprocessor, num_variants=2):
    fig, axes = plt.subplots(
        len(images),
        num_variants + 1,
        figsize=(5 * (num_variants + 1), 5 * len(images))
    )

    for i, img in enumerate(images):
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        for j in range(num_variants):
            processed_img = preprocessor.process_train(img)
            img_np = processed_img.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

            axes[i, j + 1].imshow(img_np)
            axes[i, j + 1].set_title(f"Enhanced #{j + 1}")
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.savefig("enhancements_demo.png")
    print("Enhancement comparison saved as enhancements_demo.png")


if __name__ == "__main__":
    preprocessor = EnhancedPreprocessor(img_size=(224, 224))

    example_images = load_example_images()
    print(f"Loaded {len(example_images)} example leaf images")

    if example_images:
        visualize_enhancements(example_images, preprocessor, num_variants=3)

    if example_images:
        sample_img = example_images[0]
        processed_tensor = preprocessor.process_infer(sample_img)
        print(f"Processed tensor shape: {processed_tensor.shape}")