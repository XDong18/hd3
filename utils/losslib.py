import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        P = F.sigmoid(inputs)
        log_P = P.log()
        log_P = torch.clamp(log_P, min=-100, max=0)
        # print(log_P.min())
        probs = log_P * targets
        batch_loss = -(torch.pow((1-P), self.gamma)) * probs

        neg_P = 1 - P
        log_neg_P = neg_P.log()
        log_neg_P = torch.clamp(log_neg_P, min=-100, max=0)
        neg_targets = 1 - targets
        neg_probs = log_neg_P * neg_targets
        batch_loss += -(torch.pow((1-neg_P), self.gamma)) * neg_probs

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        
        # print(loss.item())
        return loss

class edge_bce(nn.Module):
    def __init__(self, window_size=5, size_average=True):
        super(edge_bce, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        