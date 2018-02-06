import os
import utils
import models
import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision.utils as vutils

train_loader = utils.load_data_CIFAR10()

if not os.path.exists('./result'):
    os.mkdir('result/')

if not os.path.exists('./model'):
    os.mkdir('model/')
    
netG = models.get_netG()
netD1 = models.get_netD()
netD2 = models.get_netD()

# setup optimizer
optimizerD1 = torch.optim.Adam(netD1.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD2 = torch.optim.Adam(netD2.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion_log = utils.Log_loss()
criterion_itself = utils.Itself_loss()

input = torch.FloatTensor(64, 3, 64, 64)
noise = torch.FloatTensor(64, 100, 1, 1)
fixed_noise = torch.FloatTensor(64, 100, 1, 1).normal_(0, 1)
fixed_noise = Variable(fixed_noise)

use_cuda = torch.cuda.is_available()
if use_cuda:
    criterion_log, criterion_itself = criterion_log.cuda(),  criterion_itself.cuda()
    input= input.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

for epoch in range(200):
    for i, data in enumerate(train_loader):
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        ######################################
        # train D1 and D2
        #####################################
        
        netD1.zero_grad()
        netD2.zero_grad()
        # train with real
        if use_cuda:
            real_cpu = real_cpu.cuda()
            
        input.resize_as_(real_cpu).copy_(real_cpu)        
        inputv = Variable(input)
        
        # D1 sees real as real, minimize -logD1(x)
        output = netD1(inputv)
        errD1_real = 0.2 * criterion_log(output)#criterion(output1, labelv) * 0.2
        errD1_real.backward()
        
        # D2 sees real as fake, minimize D2(x)
        output = netD2(inputv)
        errD2_real = criterion_itself(output, False)
        errD2_real.backward()
        
        # train with fake
        noise.resize_(batch_size, 100, 1, 1).normal_(0,1)
        noisev = Variable(noise)
        fake = netG(noisev)
        
        # D1 sees fake as fake, minimize D1(G(z))
        output = netD1(fake.detach())
        errD1_fake = criterion_itself(output, False)
        errD1_fake.backward()
        
        # D2 sees fake as real, minimize -log(D2(G(z))
        output = netD2(fake.detach())
        errD2_fake = 0.1 * criterion_log(output)
        errD2_fake.backward()
        
        optimizerD1.step()
        optimizerD2.step()
        
        ##################################
        # train G
        ##################################
        netG.zero_grad()
        # G: minimize -D1(G(z)): to make D1 see fake as real
        output = netD1(fake)
        errG1 = criterion_itself(output)
        
        # G: minimize logD2(G(z)): to make D2 see fake as fake
        output = netD2(fake)
        errG2 = criterion_log(output, False)
        
        errG = errG2*0.1 + errG1
        errG.backward()
        optimizerG.step()
        
        if ((i+1) % 200 == 0):
            print(i+1, "step")
            print(str(errG1.data[0]) + " " + str(errG2.data[0]*0.1))
            fake = netG(fixed_noise)
            if use_cuda:
                vutils.save_image(fake.cpu().data, '%s/fake_samples_epoch_%s.png' % ('result', str(epoch)+"_"+str(i+1)), normalize=True)
            else:
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%s.png' % ('result', str(epoch)+"_"+str(i+1)), normalize=True)
    print("%s epoch finished" % (str(epoch)))
    print("-----------------------------------------------------------------\n")
    fake = netG(fixed_noise)
    if use_cuda:
        vutils.save_image(fake.cpu().data, '%s/fake_samples_epoch_%s.png' % ('result', str(epoch)+"_"+str(i+1)), normalize=True)
    else:
        vutils.save_image(fake.data, '%s/fake_samples_epoch_%s.png' % ('result', str(epoch)+"_"+str(i+1)), normalize=True)
    torch.save(netG.state_dict(), '%s/netG.pth' % ('model'))
    torch.save(netD1.state_dict(), '%s/netD1.pth' % ('model'))
    torch.save(netD2.state_dict(), '%s/netD2.pth' % ('model'))
