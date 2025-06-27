#!/usr/bin/env python
# coding: utf-8

# In[2]:


# !pip install gdown
# !pip install torch opencv-python
# pip install mediapipe
# !pip install pandas


# In[ ]:


# pip install torch opencv-python
# pip install mediapipe
# pip install pandas


# In[4]:


# import gdown
# # https://drive.google.com/file/d/1VvAI_EmjfPl-MHsyEOlLc9rE2OZ4jRtY/view?usp=sharing
# file_id = '1VvAI_EmjfPl-MHsyEOlLc9rE2OZ4jRtY'
# gdown_url = f'https://drive.google.com/uc?id={file_id}'
# output_path = '../dataset/'


# In[5]:


# gdown.download(gdown_url, output_path, quiet=False)


# In[8]:


# !unzip ../dataset/clips.zip -d ../dataset/


# In[1]:


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


# In[2]:


import logging  # CHANGE: imported logging

# --------------------
# Logging Configuration
# --------------------
logging.basicConfig(  # CHANGE: configured logging to file + console
    level=logging.INFO,  # CHANGE
    format='%(asctime)s [%(levelname)s] %(message)s',  # CHANGE
    handlers=[  # CHANGE
        logging.FileHandler("app.log", mode='a'),  # CHANGE
        logging.StreamHandler()  # CHANGE
    ]
)  # CHANGE
logger = logging.getLogger(__name__)  # CHANGE


# In[ ]:


import os
import csv
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
import torchvision.models as models

# --------------------
# Dataset Definition (sequence of landmarks)  # CHANGE
# --------------------
class HandSignDataset(Dataset):  # CHANGE: renamed class
    def __init__(self, videos_dir, csv_path, seq_length=50, transform=None):
        self.videos_dir = videos_dir
        self.seq_length = seq_length
        self.transform = transform

        # Read CSV mapping video filenames to labels
        self.samples = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for fname, label in reader:
                path = os.path.join(videos_dir, fname).replace('\\', '/')  # CHANGE: normalize path
                print(path)
                logger.info(f"Loaded sample path: {path}")  # CHANGE: replaced print() with logger
                self.samples.append((path, int(label)))

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
            # print(type(rgb))
            results = self.mp_hands.process(rgb)
            if results.multi_hand_landmarks:
                coords = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                landmarks_seq.append(coords)
        cap.release()

        # pad or truncate to fixed length
        seq = torch.zeros(self.seq_length, 21*3)
        for i, coords in enumerate(landmarks_seq[:self.seq_length]):
            seq[i] = torch.tensor(coords)

        if self.transform:
            seq = self.transform(seq)
        return seq, label  # CHANGE


class IV3_GRUClasssifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=3):  # CHANGE: default num_classes
        super(IV3_GRUClasssifier, self).__init__()
        inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.embed = nn.Embedding(num_embeddings=10000, embedding_dim=256)
        self.dropout = nn.Dropout(0.5)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)  # out: (batch, seq_len, hidden_size)
        last = out[:, -1, :]
        return self.classifier(last)

# --------------------
# Training Loop  # CHANGE
# --------------------
def train_model(videos_dir, csv_path, num_classes,
                batch_size=8, lr=1e-3, epochs=20, seq_length=50, device='cuda'):
    logger.info(f"Executing Training.")
    dataset = HandSignDataset(videos_dir, csv_path, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Model and Data Loaded.")
    model = IV3_GRUClasssifier(input_size=63, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for seq, labels in dataloader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * seq.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        logger.info(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    torch.save(model.state_dict(), 'gru_hand_sign_model.pth')
    print("Training complete. Model saved to gru_hand_sign_model.pth")
    logger.info("Training complete. Model saved to gru_hand_sign_model.pth")

def infer_one_sample(videos_dir, csv_path, model_path,
                     sample_idx=0, seq_length=50, device='cuda'):
    dataset = HandSignDataset(videos_dir, csv_path, seq_length)
    seq, true_label = dataset[sample_idx]  # CHANGE

    model = IV3_GRUClasssifier(input_size=63, hidden_size=128, num_layers=2,
                          num_classes=max([lbl for _, lbl in dataset.samples])+1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    seq = seq.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(seq)
        probs = torch.softmax(outputs, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()

    print(f"Sample index: {sample_idx}")
    print(f"Expected label: {true_label}")
    print(f"Predicted label: {pred_label}")
    print(f"Probabilities: {probs.cpu().numpy()}")
    logger.info(f"Sample index: {sample_idx}")
    logger.info(f"Expected label: {true_label}")
    logger.info(f"Predicted label: {pred_label}")
    logger.info(f"Probabilities: {probs.cpu().numpy()}") 


if __name__ == '__main__':
    VIDEOS_DIR = '../dataset/'
    CSV_PATH = 'spliced_train.csv'
    MODEL_PATH = 'gru_hand_sign_model.pth'
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    lr_change_based_on_batch_size = 1e-3 * (BATCH_SIZE / 4)

    train_model(
        videos_dir=VIDEOS_DIR,
        csv_path=CSV_PATH,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        lr=lr_change_based_on_batch_size,
        epochs=30,
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



