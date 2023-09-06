r"""
Performance evaluation metrics
"""

import torch
import scipy


def matching_metrics(similarity=None, x=None, y=None, **kwargs):
    if similarity is None:
        if x.shape != y.shape:
            raise ValueError("Shapes do not match!")
        similarity = 1 - scipy.spatial.distance_matrix(x, y, **kwargs)
    if not isinstance(similarity, torch.Tensor):
        similarity = torch.from_numpy(similarity)

    with torch.no_grad():
        # similarity = output.logits_per_atac
        batch_size = similarity.shape[0]
        acc_x = (
            torch.sum(
                torch.argmax(similarity, dim=1)
                == torch.arange(batch_size).to(similarity.device)
            )
            / batch_size
        )
        acc_y = (
            torch.sum(
                torch.argmax(similarity, dim=0)
                == torch.arange(batch_size).to(similarity.device)
            )
            / batch_size
        )
        foscttm_x = (
            (similarity > torch.diag(similarity)).float().mean(axis=1).mean().item()
        )
        foscttm_y = (
            (similarity > torch.diag(similarity)).float().mean(axis=0).mean().item()
        )
        # matchscore_x = similarity.softmax(dim=1).diag().mean().item()
        # matchscore_y = similarity.softmax(dim=0).diag().mean().item()
        X = similarity
        mx = torch.max(X, dim=1, keepdim=True).values
        hard_X = (mx == X).float()
        logits_row_sums = hard_X.clip(min=0).sum(dim=1)
        matchscore = hard_X.clip(min=0).diagonal().div(logits_row_sums).mean().item()

        acc = (acc_x + acc_y) / 2
        foscttm = (foscttm_x + foscttm_y) / 2
        # matchscore = (matchscore_x + matchscore_y)/2
        return acc, matchscore, foscttm
