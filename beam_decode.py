# beam_decode.py
import torch
from label_encoder import idx_to_char, BLANK_INDEX

def ctc_beam_search(logits, beam_width=15):
    """
    logits: (T, C)
    returns best decoded string
    """
    T, C = logits.shape
    beams = [("", 0.0)]  # (sequence, score)

    for t in range(T):
        new_beams = []
        probs = torch.log_softmax(logits[t], dim=0)

        for seq, score in beams:
            for c in range(C):
                p = probs[c].item()
                if c == BLANK_INDEX:
                    new_beams.append((seq, score + p))
                else:
                    char = idx_to_char[c]
                    if len(seq) == 0 or seq[-1] != char:
                        new_beams.append((seq + char, score + p))
                    else:
                        new_beams.append((seq, score + p))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

    return beams[0][0]
