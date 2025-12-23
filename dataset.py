import torch
from torch.utils.data import Dataset
import scipy.io as sio
from preprocess import preprocess_image
from label_encoder import encode_label

class IIIT5KDataset(Dataset):
    def __init__(self, mat_path, root_dir):
        self.root_dir = root_dir
        mat = sio.loadmat(mat_path)
        key = [k for k in mat.keys() if not k.startswith("__")][0]
        self.samples = mat[key]

    def __len__(self):
        return self.samples.shape[1] if self.samples.shape[0] == 1 else self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[0][idx] if self.samples.shape[0] == 1 else self.samples[idx][0]

        img_name = sample["ImgName"][0]
        img_path = f"{self.root_dir}/{img_name}"

        chars = sample["chars"][0]
        word = "".join([c[0] for c in chars]).lower()

        label = encode_label(word)
        if len(label) == 0:
            return self.__getitem__((idx + 1) % len(self))

        img = preprocess_image(img_path)
        img = torch.from_numpy(img).unsqueeze(0)  # (1,32,128)

        label = torch.tensor(label, dtype=torch.long)
        return img, label
