import torch
import torch.nn as nn

nz = 100
nc = 3
ngf = 64
ndf = 64

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            
            # Z
            nn.ConvTranspose2d(nz, ngf*8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # (ngf * 8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # (ngf * 4) x 4 x 4
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            
            # (ngf * 2) x 8 x 8
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            
            # ngf x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        output = self.main(input)
        return output

class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
            # (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 2) x 8 x 8
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 8) x 2 x 2
            nn.Conv2d(ndf*8, 1, 2, 1, 0, bias=False),
            nn.Softplus()
        )
    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_netG():
    use_cuda = torch.cuda.is_available()
    netG = _netG()
    netG.apply(weights_init)
    if use_cuda:
        print("USE CUDA")
        netG.cuda()
    return netG

def get_netD():
    use_cuda = torch.cuda.is_available()
    netD = _netD()
    netD.apply(weights_init)
    if use_cuda:
        print("USE CUDA")
        netD.cuda()
    return netD
