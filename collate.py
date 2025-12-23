import torch

def ctc_collate(batch):
    images, labels = zip(*batch)

    images = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat(labels)

    input_lengths = torch.full(
        size=(images.size(0),),
        fill_value=32,
        dtype=torch.long
    )

    return images, labels, input_lengths, label_lengths
