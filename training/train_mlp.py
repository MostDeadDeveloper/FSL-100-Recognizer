#!/usr/bin/env python
# coding: utf-8

# In[2]:


# !pip install gdown
# !pip install torch opencv-python
# pip install mediapipe
# !pip install pandas


# In[1]:


# import gdown
# # https://drive.google.com/file/d/1VvAI_EmjfPl-MHsyEOlLc9rE2OZ4jRtY/view?usp=sharing
# file_id = '1VvAI_EmjfPl-MHsyEOlLc9rE2OZ4jRtY'
# gdown_url = f'https://drive.google.com/uc?id={file_id}'
# output_path = '../dataset/'


# In[2]:


# gdown.download(gdown_url, output_path, quiet=False)


# In[3]:


# !unzip ../dataset/clips.zip -d ../dataset/


# In[4]:


#%%

import pandas as pd
df = pd.read_csv('train.csv')


# assume df is your DataFrame
# allowed = ["COLOR", "CALENDAR", "GREETING"]
allowed = ["GOOD EVENING", "COLD","JUICE"]
# allowed = ['GOOD EVENING']

# filtered_df = df[df['category'].isin(allowed)].reset_index(drop=True)
filtered_df = df[df['label'].isin(allowed)].reset_index(drop=True)
# filtered_df = df
# filtered_df=  filtered_df.drop(columns='label')
filtered_df=  filtered_df.drop(columns='category')
# optional: verify
# print(filtered_df['category'].value_counts())

filtered_df= filtered_df.rename(columns={'label': 'label_str'})

filtered_df['label_str'] = filtered_df['label_str'].astype('category')

filtered_df['label'] = filtered_df['label_str'].cat.codes

filtered_df = filtered_df.rename(columns={'vid_path':'filename'})

filtered_df = filtered_df.drop(columns=['id_label','label_str'])

# filtered_df.drop(filtered_df.columns[0], axis=1, inplace=True)

filtered_df.to_csv('spliced_train.csv', index=False)


# In[15]:


# --------------------
# Logging Configuration  # CHANGE
# --------------------
import logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),      # log to file  # CHANGE
        logging.StreamHandler()              # also print to console  # CHANGE
    ]
)  # CHANGE


# In[ ]:


import os
import csv
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp

# --------------------
# Dataset Definition with Statistical Features
# --------------------
class HandSignFeatureDataset(Dataset):
    def __init__(self, videos_dir, csv_path, seq_length=50, transform=None):
        self.videos_dir = videos_dir
        self.seq_length = seq_length
        self.transform = transform

        # Read CSV mapping video filenames to labels
        self.samples = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for fname, label in reader:
                path = os.path.join(videos_dir, fname)
                path = path.replace("\\", "/")
                self.samples.append((path, int(label)))
                print(path)
                logging.info(path)  # CHANGE

        # Initialize MediaPipe hand detector
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        landmarks_seq = []

        while len(landmarks_seq) < self.seq_length:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb)
            if results.multi_hand_landmarks:
                coords = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                landmarks_seq.append(coords)
        cap.release()

        # pad with zeros if needed
        seq = torch.zeros(self.seq_length, 21*3)
        for i, coords in enumerate(landmarks_seq[:self.seq_length]):
            seq[i] = torch.tensor(coords)

        # ------- Feature extraction: statistical pooling -------  # CHANGE
        # compute mean, std, max, min across time dimension
        mean_feats = seq.mean(dim=0)
        std_feats = seq.std(dim=0)
        max_feats, _ = seq.max(dim=0)
        min_feats, _ = seq.min(dim=0)
        features = torch.cat([mean_feats, std_feats, max_feats, min_feats], dim=0)  # CHANGE

        if self.transform:
            features = self.transform(features)
        print(f'Sucessful: {video_path}')
        logging.info(f'Sucessful: {video_path}')  # CHANGE
        return features, label

# --------------------
# Model Definition: Small MLP Classifier  # CHANGE
# --------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_size=63*4, hidden_size=64, num_classes=10):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # CHANGE
            nn.ReLU(),                            # CHANGE
            nn.Dropout(0.3),                      # CHANGE
            nn.Linear(hidden_size, hidden_size), # CHANGE
            nn.ReLU(),                            # CHANGE
            nn.Dropout(0.3),                      # CHANGE
            nn.Linear(hidden_size, num_classes)   # CHANGE
        )

    def forward(self, x):
        return self.net(x)

# --------------------
# Training Loop
# --------------------
def train_model(videos_dir, csv_path, num_classes,
                batch_size=8, lr=1e-3, epochs=20, seq_length=50, device='cuda'):
    dataset = HandSignFeatureDataset(videos_dir, csv_path, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLPClassifier(input_size=63*4, hidden_size=64, num_classes=num_classes).to(device)  # CHANGE
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for features, labels in dataloader:  # CHANGE
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)  # CHANGE
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        msg = f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
        print(msg)
        logging.info(msg)  # CHANGE

    torch.save(model.state_dict(), 'mlp_hand_sign_model.pth')
    print("Training complete. Model saved to mlp_hand_sign_model.pth")
    logging.info("Training complete. Model saved to mlp_hand_sign_model.pth")  # CHANGE

# --------------------
# Single-Sample Inference (adapted)  # CHANGE
# --------------------
def infer_one_sample(videos_dir, csv_path, model_path,
                     sample_idx=0, seq_length=50, device='cuda'):
    dataset = HandSignFeatureDataset(videos_dir, csv_path, seq_length)
    feat, true_label = dataset[sample_idx]  # CHANGE

    model = MLPClassifier(input_size=63*4, hidden_size=64, num_classes=max([lbl for _, lbl in dataset.samples])+1).to(device)  # CHANGE
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    feat = feat.unsqueeze(0).to(device)  # CHANGE
    with torch.no_grad():
        outputs = model(feat)  # CHANGE
        probs = torch.softmax(outputs, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()

    print(f"Sample index: {sample_idx}")
    print(f"Expected label: {true_label}")
    print(f"Predicted label: {pred_label}")
    print(f"Probabilities: {probs.cpu().numpy()}")
    logging.info(f"Sample index: {sample_idx}")
    logging.info(f"Expected label: {true_label}")
    logging.info(f"Predicted label: {pred_label}")
    logging.info(f"Probabilities: {probs.cpu().numpy()}")

# --------------------
# Example Usage
# --------------------
# if __name__ == '__main__':
# VIDEOS_DIR = 'data/videos'
# CSV_PATH = 'data/labels.csv'
MODEL_PATH = 'mlp_hand_sign_model.pth'
NUM_CLASSES = 3
VIDEOS_DIR = '../dataset/'  # Change this
CSV_PATH = 'spliced_train.csv'  # Change this
BATCH_SIZE = 32
lr_change_based_on_batch_size = 1e-3 * (BATCH_SIZE / 4)

train_model(
    videos_dir=VIDEOS_DIR,
    csv_path=CSV_PATH,
    num_classes=NUM_CLASSES,
    batch_size=BATCH_SIZE,
    lr=lr_change_based_on_batch_size,
    epochs=50,
    seq_length=50,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

infer_one_sample(
    videos_dir=VIDEOS_DIR,
    csv_path=CSV_PATH,
    model_path=MODEL_PATH,
    sample_idx=0,
    seq_length=50,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


# In[ ]:




