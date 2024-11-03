import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        
        # inputs: 모델의 로짓 출력, targets: 실제 레이블
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)  # p_t를 계산
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        return F_loss.mean()

class Criterion(nn.Module):
    def __init__(self, loss_type):
        super(Criterion, self).__init__()
        self.loss_type = loss_type
        
        if self.loss_type == "classifier":
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_type == "regression":
            self.criterion = nn.L1Loss()
        elif self.loss_type == "focal":
            self.criterion = FocalLoss()

    def compute_loss(self, a, b):
        # loss = self.criterion(a, b)
        loss = self.criterion(a, b[:, 0])
        return loss

    def forward(self, a, b):
        loss = self.compute_loss(a, b)
        return loss
