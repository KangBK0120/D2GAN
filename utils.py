import torchvision
import torch
import torchvision.datasets as dsets
import torchvision.utils as vutils
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def load_data_STL10():
    train_dataset =dsets.STL10(root='./data/', split='train+unlabeled', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader

def load_data_CIFAR10():
    train_dataset = dsets.CIFAR10(root='./data/', train=True,download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader

class Log_loss(torch.nn.Module):
    def __init__(self):
        # negation is true when you minimize -log(val)
        super(Log_loss, self).__init__()
       
    def forward(self, x, negation=True):
        # shape of x will be [batch size]
        log_val = torch.log(x)
        loss = torch.sum(log_val)
        if negation:
            loss = torch.neg(loss)
        return loss
    
class Itself_loss(torch.nn.Module):
    def __init__(self):
        super(Itself_loss, self).__init__()
        
    def forward(self, x, negation=True):
        # shape of x will be [batch size]
        loss = torch.sum(x)
        if negation:
            loss = torch.neg(loss)
        return loss
