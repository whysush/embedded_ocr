import torch.nn as nn

NUM_CLASSES = 37  # 62 chars + CTC blank

class OCRNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        self.rnn = nn.GRU(
            input_size=64 * 8,
            hidden_size=128,
            bidirectional=True
        )

        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(3, 0, 2, 1)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
