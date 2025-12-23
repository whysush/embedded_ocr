import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import IIIT5KDataset
from collate import ctc_collate
from model import OCRNet
from label_encoder import BLANK_INDEX

BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3

dataset = IIIT5KDataset(
    mat_path="data/IIIT5K/trainCharBound.mat",
    root_dir="data/IIIT5K"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=ctc_collate
)

model = OCRNet()
criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels, input_lengths, label_lengths in loader:
        optimizer.zero_grad()
        outputs = model(images)
        log_probs = outputs.log_softmax(2)

        loss = criterion(
            log_probs,
            labels,
            input_lengths,
            label_lengths
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss / len(loader):.4f}")

torch.save(model.state_dict(), "ocr_crnn.pth")
print("Model saved.")
