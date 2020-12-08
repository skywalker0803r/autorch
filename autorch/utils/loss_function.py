import torch

def hingeLoss(y_pred,y_real,model):
    loss1 = torch.sum(torch.clamp(1 - torch.matmul(y_pred.t(),y_real),min=0))
    loss2 = 0
    for name,weight in model.named_parameters():
      if 'weight' in name:
        loss2 += torch.sum(weight ** 2) # l2 penalty
    return loss1 + loss2