import torch
import numpy as np
import scipy.io as sio
from Spa_downs import *
from LoadDataset_Batch import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import os
import matplotlib.pyplot as plt
from torch.nn.functional import upsample

from ThreeBranch_3 import *


lr = 1e-4
batch_size = 6
factor = 3


model = ThreeBranch_Net().cuda()

Train_image_num = 12   # CAVE:12, Harvard:20, ICLV:20
save_path = './Ours_result/Models/CAVE_lr{}_X{}_{}Pic_K-P-Unknown/'.format(lr, factor, Train_image_num)
if not os.path.exists(save_path):
    os.mkdir(save_path)

#Load Dataset
dataset = LoadDataset(Path='/Datasets/CAVE/', datasets='CAVE', patch_size=128, stride=64, Data_Aug=True, up_mode='bicubic', Normlization=1, Train_image_num=Train_image_num)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)

#loss & optimizer
L1 = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#spatial and spectral downsampler
P = sio.loadmat('P_N_V2.mat')
P = Variable(torch.from_numpy(P['P'])).type(torch.cuda.FloatTensor)

WS = [[7, 1/2], [8, 3], [9, 2], [13, 4], [15, 1.5]]

#learning rate decay
def LR_Decay(optimizer, n):
    lr_d = lr * (0.7**n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_d


Minimum_Loss = 1000
k_num = 0
n_num = 1

n = 0

start_time = time.time()

f = open(save_path+'loss.txt', 'w+')
f.write('The total loss, loss1, loss2 is : \n\n\n')


#Training

for epoch in range(150):
    print('*'*10, 'The {}th epoch for training with DataAug_noSkip_3KS.'.format(epoch+1), '*'*10)
    running_loss = 0  #the total loss

    for iteration, Data in enumerate(data_loader, 1):

        GT = Data.type(torch.cuda.FloatTensor)

        # Randomly sampling the spatial downsampler
        ws = np.random.randint(0, 5, 1)[0]
        ws = WS[ws]
        ks = np.random.randint(0, 10, 1)[0]
        down_spa = Spa_Downs(
            31, factor, kernel_type=ks, kernel_width=ws[0],
            sigma=ws[1],preserve_size=True
        ).type(torch.cuda.FloatTensor)

        # Randomly sampling the spectral response matrix
        threshold = torch.rand(1)[0]
        if threshold > 0.3:
            R_deg = 1e-4 * torch.randint(50, 80, [1])[0]
            P_re = torch.unsqueeze((P + R_deg)/(P.sum(1).unsqueeze(1) + R_deg*GT.shape[1]), 0)
        else:
            P_re = P.unsqueeze(0)


        #Generate the LR_HSI
        LR_HSI = down_spa(GT)
        #Generate the HR_MSI
        HR_MSI = torch.matmul(P_re, GT.reshape(-1, GT.shape[1], GT.shape[2]*GT.shape[3])).reshape(-1, 3, GT.shape[2], GT.shape[3])
        #Generate the UP_HSI
        UP_HSI = upsample(LR_HSI, (GT.shape[2], GT.shape[3]), mode='bilinear')
        Input_1 = Variable(HR_MSI, requires_grad=False).type(torch.cuda.FloatTensor)
        Input_2 = Variable(torch.cat((UP_HSI, HR_MSI), 1), requires_grad=False).type(torch.cuda.FloatTensor)
        Input_3 = Variable(UP_HSI, requires_grad=False).type(torch.cuda.FloatTensor)

        # Calculating the output
        out = model(Input_1, Input_2, Input_3)

        HSI_out = down_spa(out)
        MSI_out = torch.reshape(torch.matmul(P_re, torch.reshape(out, (out.shape[0], out.shape[1], out.shape[2]*out.shape[3]))), (out.shape[0],HR_MSI.shape[1],HR_MSI.shape[2],HR_MSI.shape[3]))

        loss = L1(out, GT)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data.cpu()

    if epoch%10 == 0:
        LR_Decay(optimizer, n)
        n += 1
        print('Adjusting the learning rate by timing 0.8.')


    print('*'*10, 'The loss is {:.4f}.'.format(running_loss), '*'*10)
    f.write('The loss at {}th epoch is{:.4f}.\n'.format(epoch, running_loss))

    if epoch%10 == 9:
        torch.save(model, save_path+'model_'+str(int(epoch/10))+'.pth')

T = time.time()-start_time

print('Total training time is {}'.format(T))
f.write('Total traing time is {}.\n'.format(T))

f.close()







