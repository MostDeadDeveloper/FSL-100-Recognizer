#%%
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
# CONFIGURATION

# -------- Dataset Class --------
class VideoDataset(Dataset):
    def __init__(self, video_folder, csv_file, num_frames=16, frame_size=(112, 112)):
        self.video_folder = video_folder
        self.labels_df = pd.read_csv(csv_file)
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        video_path = os.path.join(self.video_folder, row['filename'])
        label = int(row['label'])

        frames = self._load_video_frames(video_path)
        return frames, label

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Choose frame indices uniformly
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)

        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame / 255.0  # normalize
                frame = np.transpose(frame, (2, 0, 1))  # CxHxW
                frames.append(frame)

        cap.release()

        # If video was too short, pad with zeros
        while len(frames) < self.num_frames:
            frames.append(np.zeros((3, *self.frame_size)))

        frames = np.stack(frames)  # Shape: [num_frames, 3, H, W]
        return torch.tensor(frames, dtype=torch.float32)

# -------- GRU Model --------
class VideoGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VideoGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.feature_dim = input_size[0] * input_size[1] * input_size[2]

        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size, seq_len, -1)  # flatten each frame
        out, _ = self.gru(x)
        out = out[:, -1, :]   # last hidden state
        out = self.fc(out)
        return out

# -------- Train Function --------
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Train Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}%")

        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader, device)
        print(f"Test Acc: {test_acc:.2f}%\n")

# -------- Evaluation Function --------
def evaluate_model(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    return acc

# -------- Main --------
if __name__ == '__main__':
    # Parameters
    # video_folder = '/path/to/videos'
    # csv_file = '/path/to/labels.csv'

    video_folder = 'workspace/clips/'  # Change this
    csv_file = 'spliced_train.csv'  # Change this
    num_frames = 16
    frame_size = (112, 112)
    batch_size = 4
    hidden_size = 128
    num_layers = 1
    num_classes = 3
    num_epochs = 30
    learning_rate = 0.001
    test_split_ratio = 0.2  # 20% for test


    # PREPROCESSING =======================================================
    # Full dataset 
    full_dataset = VideoDataset(video_folder, csv_file, num_frames, frame_size)

    # Split into train/test
    test_size = int(len(full_dataset) * test_split_ratio)
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # PREPROCESSING =======================================================

    # TRAINING ============================================================
    # Model
    model = VideoGRUModel(input_size=(3, *frame_size), hidden_size=hidden_size,
                          num_layers=num_layers, num_classes=num_classes)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # TRAINING ============================================================

    # Train & Evaluate
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs,
                device='cuda' if torch.cuda.is_available() else 'cpu')
