import clearml
from clearml import Task, Logger
from clearml import Dataset as ClearMLDataset
import fiftyone as fo
from fiftyone.types import FiftyOneDataset
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import psutil
import random

# Создание задачи в ClearML
task = Task.init(project_name="Belt Classification", task_name="MobileNetV2 Training", task_type=Task.TaskTypes.training)

# Определение и логирование конфигурации гиперпараметров
params = {
    "dataset_name": "e2337bfcb3644857b7212766c39e3991",
    "dataset_view": "v1_dataset",
    "num_epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.001,
    "train_resize": (224, 224),
    "val_resize": (224, 224),
    "random_horizontal_flip": True,
    "random_rotation_degrees": 30,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "class_weights": [0.1, 0.1, 0.8], 
    "train_mode": "fine_tune"
}


task.connect(params)

dataset_id = params.get("dataset_name", "e2337bfcb3644857b7212766c39e3991")
dataset_view = params.get("dataset_view", "v1_dataset")
num_epochs = params.get("num_epochs", 20)
batch_size = params.get("batch_size", 64)
learning_rate = params.get("learning_rate", 0.001)
train_resize = tuple(params.get("train_resize", (224, 224)))
val_resize = tuple(params.get("val_resize", (224, 224)))
random_horizontal_flip = params.get("random_horizontal_flip", True)
random_rotation_degrees = params.get("random_rotation_degrees", 30)
normalize_mean = params.get("normalize_mean", [0.485, 0.456, 0.406])
normalize_std = params.get("normalize_std", [0.229, 0.224, 0.225])
class_weights = torch.tensor(params.get("class_weights", [0.1, 0.1, 0.8]))
train_mode = params.get("train_mode", "fine_tune")

dataset_path = ClearMLDataset.get(dataset_id="e2337bfcb3644857b7212766c39e3991").get_local_copy()

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_path,
    dataset_type=fo.types.FiftyOneDataset
)
print(f"Dataset is downloaded to: {dataset_path}")

print(f"Total samples in FiftyOne dataset: {len(dataset)}")

view = dataset.load_saved_view(dataset_view)

print(f"Loaded view '{dataset_view}' with {len(view)} samples")

train_transform = transforms.Compose([
    transforms.Resize(train_resize),
    transforms.RandomHorizontalFlip() if random_horizontal_flip else transforms.Lambda(lambda x: x),
    transforms.RandomRotation(random_rotation_degrees),
    #transforms.ColorJitter(**color_jitter_params),
    #transforms.RandomResizedCrop(train_resize, scale=random_resized_crop_scale),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std),
])

val_transform = transforms.Compose([
    transforms.Resize(val_resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std),
])

class FiftyOneDataset(Dataset):
    def __init__(self, fiftyone_view, transform=None):
        self.sample_ids = fiftyone_view.values("id")
        self.fiftyone_view = fiftyone_view
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample = self.fiftyone_view[sample_id]
        img_path = sample.filepath
        image = Image.open(img_path).convert("RGB")

        # Получаем метку из ground_truth
        label = sample.ground_truth.label

        # Преобразование метки в числовое значение
        label_map = {"not_visible": 0, "with_belt": 1, "without_belt": 2}
        label = label_map[label]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_dataset(dataset_name):
    dataset = fo.load_dataset(dataset_name)
    return dataset


# Split
train_ratio = 0.8

print(f"train ratio {0.8}")

sample_ids = dataset.values("id")
random.shuffle(sample_ids)
num_train_samples = int(train_ratio * len(sample_ids))

train_ids = sample_ids[:num_train_samples]
test_ids = sample_ids[num_train_samples:]

train_view = dataset.select(train_ids)
val_view = dataset.select(test_ids)

print(f"Len train_view {len(train_view)}")
print(f"Len val_view {len(val_view)}")

# Pytorch Dataset
train_dataset = FiftyOneDataset(fiftyone_view=train_view, transform=train_transform)
val_dataset = FiftyOneDataset(fiftyone_view=val_view, transform=val_transform)

# DataLoader Pytorch
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Test probe
data_iter = iter(train_loader)
images, labels = next(data_iter)

print(f"Images size: {images.size()}")
print(f"Labels size: {labels.size()}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weights=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(weight=self.weights, reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
criterion = FocalLoss(alpha=1, gamma=2, weights=class_weights)

model = models.mobilenet_v2(pretrained=True)

# Замените последний слой на ваш классификатор
num_classes = 3
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Настройка обучения в зависимости от режима
if train_mode == "from_scratch":
    for param in model.parameters():
        param.requires_grad = True
elif train_mode == "last_layer":
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
elif train_mode == "fine_tune":
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-1].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    
print("Model is ready")
        
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

print("Optimizer is ready")

model.train()


for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    task.get_logger().report_scalar("Loss", "train", iteration=epoch, value=train_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss}")

    # Валидация модели
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_loss /= len(val_loader)
    task.get_logger().report_scalar("Loss", "val", iteration=epoch, value=val_loss)
    print(f"Validation Loss: {val_loss}")

    # Обновление планировщика
    scheduler.step(val_loss)

    # Вычисление метрик
    cm = confusion_matrix(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
    avg_precision = average_precision_score(all_labels, all_preds)

    # Отчет метрик в ClearML
    task.get_logger().report_scalar("AUC", "val", iteration=epoch, value=auc)
    task.get_logger().report_scalar("Average Precision", "val", iteration=epoch, value=avg_precision)
    
    # Отрисовка матрицы ошибок
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix at Epoch {epoch + 1}")
    task.get_logger().report_matplotlib_figure(f"Confusion Matrix at Epoch {epoch + 1}", series="confusion_matrix", figure=plt.gcf())
    plt.clf()

    model.train()
    
    
