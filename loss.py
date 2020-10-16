import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return -(log_softmax_outputs*softmax_targets).sum(dim=1).mean()

def L1_soft(outputs, targets):
    softmax_outputs = F.softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return F.l1_loss(softmax_outputs, softmax_targets)

def L2_soft(outputs, targets):
    softmax_outputs = F.softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return F.mse_loss(softmax_outputs, softmax_targets)


class betweenLoss(nn.Module):
    def __init__(self, gamma=[1, 1, 1, 1, 1, 1], loss=nn.L1Loss()):
        super(betweenLoss, self).__init__()
        self.gamma = gamma
        self.loss = loss

    def forward(self, outputs, targets):
        assert len(outputs)
        assert len(outputs) == len(targets)
        length = len(outputs)
        res = sum([self.gamma[i] * self.loss(outputs[i], targets[i]) for i in range(length)])
        return res


class discriminatorLoss(nn.Module):
    def __init__(self, models, eta=[1, 1, 1, 1, 1], loss=nn.BCEWithLogitsLoss()):
        super(discriminatorLoss, self).__init__()
        self.models = models
        self.eta = eta
        self.loss = loss

    def forward(self, outputs, targets):
        
        inputs = [torch.cat((i,j),0) for i, j in zip(outputs, targets)]
        batch_size = inputs[0].size(0)
        target = torch.FloatTensor([[1, 0] for _ in range(batch_size//2)] + [[0, 1] for _ in range(batch_size//2)])
        target = target.to(inputs[0].device)
        outputs = self.models(inputs)
        res = sum([self.eta[i] * self.loss(output, target) for i, output in enumerate(outputs)])
        return res


class discriminatorFakeLoss(nn.Module):
    def forward(self, outputs, targets):
        res = (0*outputs[0]).sum()
        return res

class KLLoss(loss._Loss):
    """The KL-Divergence loss for the model and soft labels output.

    output must be a pair of (model_output, soft_labels), both NxC tensors.
    The rows of soft_labels must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, output, target):
        if not self.training:
            # Loss is normal cross entropy loss between the model output and the
            # target.
            return F.cross_entropy(output, target)

        assert type(output) == tuple and len(output) == 2 and output[0].size() == \
            output[1].size(), "output must a pair of tensors of same size."

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the soft labels.
        model_output, soft_labels = output
        if soft_labels.requires_grad:
            raise ValueError("soft labels should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        del model_output

        # Loss is -dot(model_output_log_prob, soft_labels). Prepare tensors
        # for batch matrix multiplicatio
        soft_labels = soft_labels.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average for the batch.
        cross_entropy_loss = -torch.bmm(soft_labels, model_output_log_prob)
        cross_entropy_loss = cross_entropy_loss.mean()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        model_output_log_prob = model_output_log_prob.squeeze(2)
        return (cross_entropy_loss, model_output_log_prob)



