import torch


class CrossEntropy(torch.nn.Module):
    def __init__(self, reduction='mean') -> None:
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(self, pred, labels):
        ce = self.ce(pred, labels)
        return ce


def ce(reduction='mean'):
    return CrossEntropy(reduction)
