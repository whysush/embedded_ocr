import torch
from label_encoder import idx_to_char, BLANK_INDEX

def ctc_greedy_decode(logits):
    preds = torch.argmax(logits, dim=1)

    decoded = []
    prev = BLANK_INDEX

    for p in preds:
        p = p.item()
        if p != BLANK_INDEX and p != prev:
            decoded.append(idx_to_char[p])
        prev = p

    return "".join(decoded)
