import torch
import numpy as np
import scipy.io as sio
from torch.autograd import Variable
import os
from function import *
from downsampler import *
import torch.nn as nn
from torch.nn.functional import upsample
from Spa_downs import *
import time
from SSIM import *
import matplotlib.pyplot as plt


names_CAVE_test = [
    'real_and_fake_apples', 'superballs', 'chart_and_stuffed_toy', 'hairs',  'fake_and_real_lemons',
    'fake_and_real_lemon_slices', 'fake_and_real_sushi', 'egyptian_statue', 'glass_tiles', 'jelly_beans',
    'fake_and_real_peppers', 'clay', 'pompoms', 'watercolors', 'fake_and_real_tomatoes', 'flowers', 'paints',
    'photo_and_face', 'cloth', 'beads'
]
names_Harvard_test = [
    'imgb9', 'imgh3','imgc5', 'imga7', 'imgb4', 'imgh0', 'imgd7', 'imge7', 'imgb6', 'imga5', 
    'imgf7', 'imgc2', 'imgf5','imgb2', 'imge3', 'imgc1', 'imga1', 'imgc9', 'imgb5', 'img1', 
    'imgb0', 'imgd8', 'imgb8'
]

names_ICLV_test = [
    'BGU_HS_00003', 'BGU_HS_00007', 'BGU_HS_00013', 'BGU_HS_00017', 'BGU_HS_00023', 'BGU_HS_00027', 'BGU_HS_00033',
    'BGU_HS_00037', 'BGU_HS_00043', 'BGU_HS_00047', 'BGU_HS_00053', 'BGU_HS_00057', 'BGU_HS_00063', 'BGU_HS_00067',
    'BGU_HS_00073', 'BGU_HS_00077', 'BGU_HS_00083', 'BGU_HS_00087', 'BGU_HS_00093', 'BGU_HS_00097', 'BGU_HS_00103',
    'BGU_HS_00107', 'BGU_HS_00113', 'BGU_HS_00117', 'BGU_HS_00123', 'BGU_HS_00127', 'BGU_HS_00133', 'BGU_HS_00137',
    'BGU_HS_00143', 'BGU_HS_00147', 'BGU_HS_00153', 'BGU_HS_00157', 'BGU_HS_00163', 'BGU_HS_00167', 'BGU_HS_00173',
    'BGU_HS_00177', 'BGU_HS_00183', 'BGU_HS_00187', 'BGU_HS_00193', 'BGU_HS_00197', 'BGU_HS_00203', 'BGU_HS_00207',
    'BGU_HS_00213', 'BGU_HS_00217', 'BGU_HS_00223', 'BGU_HS_00227', 'BGU_HS_00233', 'BGU_HS_00237', 'BGU_HS_00243',
    'BGU_HS_00247'
]

U_spa = 1
U_spc = 1


Stages = 40 
num_steps = 10
factor = 8
K_spa = 'K1'
lr = 1e-3
lr_dc = 1e-4
lr_da = 1e-4

WD = 1e-4
WD_Dspa = 1e-4


H_N = 40
M_N = 40

data1_path = ''  # LR_HSI
data2_path = ''  # HR_MSI
GT_path = '/Datasets/Harvard/'
names = names_Harvard_test
save_path = 'Ours_result/'

if not os.path.exists(save_path):
    os.mkdir(save_path)

model = torch.load('./Models/model.pth'.format(factor))


L1Loss = nn.L1Loss()

k = 1

F = open(save_path+'result.txt', 'w+')
F.write('The whole result of this experiment is here:\n\n\n')

PSNR_SUM=0
SAM_SUM=0
SSIM_SUM=0
Time_Sum = 0

for name in names:

    start_time = time.time()
    if U_spc == 0:
        P = sio.loadmat('P_N_V2.mat')
        P = Variable(torch.unsqueeze(torch.from_numpy(P['P']), 0)).type(torch.cuda.FloatTensor)
    else:
        # Learnable spectral downsampler
        P_N = sio.loadmat('P_N_V2.mat')
        P = torch.from_numpy(P_N['P'])
        down_spc = L_Dspec(31, 3, P).type(torch.cuda.FloatTensor).cuda()
        optimizer_spc = torch.optim.Adam(down_spc.parameters(), lr=lr_dc, weight_decay=1e-5)

    if U_spa == 0:
        down_spa = Downsampler(31, factor, kernel_type='gauss83', phase=0, preserve_size=True).type(torch.cuda.FloatTensor)
    else:
        # Learnable spatial downsampler
        KS = 32
        kernel = torch.rand(1, 1, KS, KS)
        kernel[0, 0, :, :] = torch.from_numpy(get_kernel(factor, 'gauss', 0, KS, sigma=3))
        Conv = nn.Conv2d(1, 1, KS, factor)
        Conv.weight = nn.Parameter(kernel)
        dow = nn.Sequential(nn.ReplicationPad2d(int((KS - factor) / 2.)), Conv)
        downs = Apply(dow, 1).cuda()
        optimizer_d = torch.optim.Adam(downs.parameters(), lr=lr_da, weight_decay=WD_Dspa)

    print('*' * 10, 'Proceding the {}th image {}.'.format(k,name), '*' * 10)
    F.write('Proceding the {}th image {}.\n'.format(k,name))
    k += 1


    GT = sio.loadmat(GT_path+name+'.mat')
    GT = torch.from_numpy(GT['data']).type(torch.cuda.FloatTensor)
    GT = GT / (torch.max(GT) - torch.min(GT))

    data1 = sio.loadmat(data1_path + name)
    LR_HSI = torch.unsqueeze(torch.from_numpy(data1['LR_HSI']), 0).type(torch.cuda.FloatTensor)
    data2 = sio.loadmat(data2_path + name)
    HR_MSI = torch.unsqueeze(torch.from_numpy(data2['HR_MSI']), 0).type(torch.cuda.FloatTensor)
    if U_spc == 0:
        LR_MSI_spc = torch.matmul(P, LR_HSI.reshape(-1,LR_HSI.shape[1],LR_HSI.shape[2]*LR_HSI.shape[3])).reshape(-1,3,LR_HSI.shape[2],LR_HSI.shape[3])
    else:
        with torch.no_grad():
            LR_MSI_spc = down_spc(LR_HSI)

    if U_spa == 0:
        down_rgb = Downsampler(3, factor, kernel_type='gauss83', phase=0, preserve_size=True).type(torch.cuda.FloatTensor)
        LR_MSI_spa = down_rgb(HR_MSI)
    else:
        with torch.no_grad():
            LR_MSI_spa = downs(HR_MSI)

    UP_HSI = upsample(LR_HSI, (HR_MSI.shape[2],HR_MSI.shape[3]), mode='bilinear').type(torch.cuda.FloatTensor)
    Input_1 = HR_MSI
    Input_2 = torch.cat((UP_HSI, HR_MSI), 1)
    Input_3 = UP_HSI

    # Loading the Meta-trained Adaptor
    Net = torch.load('Path_')
    optimizer = torch.optim.Adam([{'params': Net.parameters(), 'initial_lr': lr}], lr=lr, weight_decay=WD)

    with torch.no_grad():
        Input = model(Input_1, Input_2, Input_3)

    out = Input

    for step in range(Stages):        
       
        for i in range(num_steps):
            out_d = downs(out.detach())
            optimizer_d.zero_grad()
            loss_d = L1Loss(out_d, LR_HSI)
            loss_d.backward()
            optimizer_d.step()

        LR_HSI = Variable(LR_HSI.detach(), requires_grad=False).type(torch.cuda.FloatTensor)
        optimizer_d.zero_grad()
        
        for i in range(num_steps):
            out_spc = down_spc(out.detach())
            optimizer_spc.zero_grad()
            loss_spc = L1Loss(out_spc, HR_MSI)
            loss_spc.backward()
            optimizer_spc.step()

        HR_MSI = Variable(HR_MSI.detach(), requires_grad=False).type(torch.cuda.FloatTensor)
        optimizer_spc.zero_grad()
        

        param_K = list(downs.named_parameters())
        K = torch.from_numpy(param_K[0][1][0].detach().cpu().numpy()).unsqueeze(0).cuda()
        param_P = list(down_spc.named_parameters())
        P = param_P[0][1].detach().permute(1,0).unsqueeze(0).unsqueeze(0)#.repeat(GT.shape[0], 1, 1, 1)

        for i in range(num_steps):

            out = Net(Input, K, P)

            if U_spa == 1:
                D_HSI = downs(out)
                D_MSI_spa = downs(HR_MSI)
            else:
                D_HSI = down_spa(out)

            if U_spc == 1:
                D_MSI = down_spc(out)
            else:
                D_MSI = torch.reshape(
                    torch.matmul(P, torch.reshape(out, (out.shape[0], out.shape[1], out.shape[2] * out.shape[3]))),
                    (out.shape[0], HR_MSI.shape[1], HR_MSI.shape[2], HR_MSI.shape[3]))
            Loss = L1Loss(D_HSI, LR_HSI) + L1Loss(D_MSI, HR_MSI) # + L1Loss(D_MSI_spa, LR_MSI_spc)

            optimizer.zero_grad()
            if U_spa == 1:
                optimizer_d.zero_grad()
            if U_spc == 1:
                optimizer_spc.zero_grad()

            Loss.backward()

            optimizer.step()
            if U_spa == 1:
                optimizer_d.step()
            if U_spc == 1:
                optimizer_spc.step()

            if i % 10 == 0:
                out_ = torch.squeeze(out)
                psnr = PSNR_GPU(GT.cpu(), out_.detach().cpu())
                sam = SAM_GPU(GT, out_.detach())
                ssim = ssim_GPU(GT.unsqueeze(0), out_.unsqueeze(0).detach())
                print('At the {0}th epoch the Loss,PSNR,SAM are {1:.8f}, {2:.2f}, {3:.2f}, {4:.4f}.'.format(i, Loss, psnr, sam,
                                                                                                            ssim))
                F.write('At the {0}th epoch the Loss, PSNR, SAM are {1:.2f}, {2:.2f}, {3:.4f}, {4:.4f}.\n'.format(i, Loss, psnr,
                                                                                                                  sam, ssim))


    if U_spa == 1:
        torch.save(downs, save_path + name + '_Dspa.pth')
    if U_spc == 1:
        torch.save(down_spc, save_path + name + '_Dspc.pth')
    
    PSNR_SUM += PSNR_GPU(GT.cpu(), out.detach().cpu())
    SAM_SUM += SAM_GPU(GT, out.detach())
    SSIM_SUM = 000

    torch.save(Net, save_path + name + '_FineNet1.pth')
    param_K = list(downs.named_parameters())
    K = param_K[0][1][0].detach().cpu().numpy()
    param_P = list(down_spc.named_parameters())
    P = param_P[0][1].detach().cpu().numpy()
    D = {}
    out = torch.squeeze(out)
    D['RE'] = np.array(out.detach().cpu())
    D['K'] = K
    D['P'] = P
    sio.savemat(save_path + name + '.mat', D)
    total_time = time.time()-start_time
    Time_Sum += total_time
    print('Total Time Used: {}.'.format(total_time))
    F.write('Total Time Used: {}.\n\n\n'.format(total_time))
print('The average PSNR,SAM,SSIM are: {0:.4f},{1:.4f},{2:.4f}'.format(PSNR_SUM/20, SAM_SUM/20, SSIM_SUM/20))
F.write('The average PSNR,SAM,SSIM are: {0:.4f},{1:.4f},{2:.4f}'.format(PSNR_SUM/20, SAM_SUM/20, SSIM_SUM/20))
print('The average time & total time  is : {}, {}'.format(Time_Sum/20, Time_Sum))
F.close()



