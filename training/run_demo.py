# app.py
import os
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import streamlit as st
import numpy as np

# --- Model definition (must match your training script) ---
class GRUClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.classifier(last)

# --- Helper: extract a fixed-length landmark sequence from video ---
def extract_landmarks(video_path, seq_length=50):
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(video_path)
    seq = []
    while len(seq) < seq_length:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_hands.process(rgb)
        if res.multi_hand_landmarks:
            coords = []
            for lm in res.multi_hand_landmarks[0].landmark:
                coords += [lm.x, lm.y, lm.z]
            seq.append(coords)
    cap.release()
    # pad/truncate
    data = np.zeros((seq_length, 63), dtype=np.float32)
    for i, coords in enumerate(seq[:seq_length]):
        data[i] = np.array(coords, dtype=np.float32)
    return data

# --- Load model once ---
@st.cache(allow_output_mutation=True)
def load_model(path="training/gru_hand_sign_model.pth", device="cpu"):
    model = GRUClassifier()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(device=device)

# --- UI ---
st.title("Filipino Hand-Sign IV3-GRU Demo")
st.write("Upload a short video (single sign) and see which of the three signs it predicts.")

uploaded = st.file_uploader("Choose a video file", type=["mp4","mov","avi"])
if uploaded:
    # save to temp
    tmp_path = os.path.join("temp_video.mp4")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.video(tmp_path)
    with st.spinner("Processing…"):
        seq = extract_landmarks(tmp_path)                # (seq_len,63)
        inp = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1,seq_len,63)
        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    classes = {0: "COLD", 1: "GOOD EVENING", 2: "JUICE"}
    top_idx = int(np.argmax(probs))
    st.subheader(f"Predicted Sign: {classes[top_idx]} ({top_idx})")
    st.write("### Probabilities")
    for i, p in enumerate(probs):
        st.write(f"- **{classes[i]}**: {p*100:.1f}%")
    
    os.remove(tmp_path)
