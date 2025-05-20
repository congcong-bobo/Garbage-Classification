import torch
from torch.utils.data import DataLoader
from dataset import GarbageDataset, get_transform
from model import ResNetClassifier
import torch.optim as optim
from torch import nn
import os
import logging

# 配置日志记录
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 配置参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'data'  # 数据集根目录（需包含train/val子文件夹）
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 3  # 提前停止的耐心值

# 1. 加载数据集
transform = get_transform()
train_dataset = GarbageDataset(os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset = GarbageDataset(os.path.join(DATA_DIR, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 2. 初始化模型
num_classes = len(train_dataset.classes)
model = ResNetClassifier(num_classes).to(device)

# 3. 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# 4. 训练函数
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, early_stopping_patience):
    best_val_acc = 0.0
    patience = 0
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        logging.info(f'Epoch {epoch+1}/{epochs}')
        logging.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        logging.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        logging.info('-' * 50)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print('-' * 50)

        # 提前停止机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            # 保存最佳模型
            model_path = os.path.join('models', f'model_best_{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)
            logging.info(f'Saved best model to {model_path}')
        else:
            patience += 1
            if patience >= early_stopping_patience:
                logging.info(f'Early stopping at epoch {epoch+1}')
                print(f'Early stopping at epoch {epoch+1}')
                break

# 5. 开始训练
if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')
    train_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, EARLY_STOPPING_PATIENCE)

    # 6. 保存最终模型（训练完成后执行）
    torch.save(model.state_dict(), os.path.join('models', 'model_final.pth'))