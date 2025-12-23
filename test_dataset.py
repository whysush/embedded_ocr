from dataset import IIIT5KDataset
from collate import ctc_collate
from torch.utils.data import DataLoader

dataset = IIIT5KDataset(
    mat_path="data/IIIT5K/trainCharBound.mat",
    root_dir="data/IIIT5K"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=ctc_collate
)

images, labels, input_lengths, label_lengths = next(iter(loader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Input lengths:", input_lengths)
print("Label lengths:", label_lengths)
