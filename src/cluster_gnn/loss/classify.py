import torch

def bce_loss(pred, target, false_pos_weight=1.0, false_neg_weight=1.0):
    """Custom loss function with options to independently penalise
    false positives and false negatives.
    """
    norm = torch.tensor(1.0 / float(len(pred)))
    tiny = torch.finfo(pred.dtype).tiny
    false_neg = torch.sum(target * torch.log(
        torch.clip(pred, min=tiny) # preventing -inf
        ))
    false_pos = torch.sum(
            (torch.tensor(1.0) - target)
            * torch.log(torch.clip(torch.tensor(1.0) - pred, min=tiny))
        )
    return - norm * (false_pos_weight * false_pos
                     + false_neg_weight * false_neg)
