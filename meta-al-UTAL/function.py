import torch
import torch.nn as nn
import numpy as np
import math

class ReshapeTo2D(nn.Module):

    def __init__(self):
        super(ReshapeTo2D, self).__init__()

    def forward(self,x):
        return torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))

class ReshapeTo3D(nn.Module):
    def __init__(self):
        super(ReshapeTo3D, self).__init__()

    def forward(self,x):
        return  torch.reshape(x, (x.shape[0], x.shape[1], int(np.sqrt(x.shape[2])), int(np.sqrt(x.shape[2]))))

class TransDimen(nn.Module):
    def __init__(self):
        super(TransDimen, self).__init__()

    def forward(self,x):
        #print(x.shape)
        return torch.Tensor.permute(x,[0,2,1])

def PSNR_GPU(im_true, im_fake):
    im_true *= 255
    im_fake *= 255
    im_true = im_true.round()
    im_fake = im_fake.round()
    data_range = 255
    esp = 1e-12
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    Itrue = im_true.clone()
    Ifake = im_fake.clone()
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum() / (C*H*W)
    psnr = 10. * np.log((data_range**2)/(err.data + esp)) / np.log(10.)
    return psnr

def SAM_GPU(im_true, im_fake):
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    esp = 1e-12
    Itrue = im_true.clone()#.resize_(C, H*W)
    Ifake = im_fake.clone()#.resize_(C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=0)#.resize_(H*W)
    denominator = Itrue.norm(p=2, dim=0, keepdim=True).clamp(min=esp) * \
                  Ifake.norm(p=2, dim=0, keepdim=True).clamp(min=esp)
    denominator = denominator.squeeze()
    sam = torch.div(nom, denominator).acos()
    sam[sam != sam] = 0
    sam_sum = torch.sum(sam) / (H * W) / np.pi * 180
    return sam_sum


class L_Dspec(nn.Module):
    def __init__(self,in_channel,out_channel,P_init):
        super(L_Dspec, self).__init__()
        self.in_channle = in_channel
        self.out_channel = out_channel
        self.P = nn.Parameter(P_init)

    def forward(self,input):
        S = input.shape
        out = torch.reshape(input,[S[0],S[1],S[2]*S[3]])
        out = torch.matmul(self.P,out)

        return torch.reshape(out,[S[0],self.out_channel,S[2],S[3]])

class Apply(nn.Module):
    def __init__(self, what, dim, *args):
        super(Apply, self).__init__()
        self.dim = dim
        self.what = what

    def forward(self, input):
        inputs = []
        for i in range(input.size(self.dim)):
            inputs.append(self.what(input.narrow(self.dim, i, 1)))
        return torch.cat(inputs, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class Apply_batch(nn.Module):
    def __init__(self, what, dim, batch_size, *args):
        super(Apply_batch, self).__init__()
        self.dim = dim
        self.what = what
        self.BS = batch_size

    def forward(self, input):
        inputs = []
        for i in range(input.size(self.dim)):
            if self.BS > 1:
                x = self.what(input.narrow(self.dim, i, 1).squeeze().unsqueeze(0))
                inputs.append(x.squeeze().unsqueeze(1))
            else:
                x = self.what(input.narrow(self.dim, i, 1))
                inputs.append(x)

        return torch.cat(inputs, dim=self.dim)

    def __len__(self):
        return len(self._modules)



class FineNet_SelfAtt(nn.Module):

    def __init__(self):
        super(FineNet_SelfAtt, self).__init__()
        self.Conv1 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv2 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv3 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv4 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv5 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Relu = nn.ReLU()
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Conv2(self.Relu(out))
        out = self.Conv3(self.Relu(out))

        Z = self.Conv5(self.Relu(out))
        M = self.Sig(self.Conv4(self.Relu(out)))

        out = M*out + (1-M)*Z

        return out + x

class FineNet_SelfAtt_ReHSI(nn.Module):

    def __init__(self):
        super(FineNet_SelfAtt_ReHSI, self).__init__()
        self.Conv1 = nn.Conv2d(31, 64, 3, 1, 1)
        self.Conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.Conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.Conv3_2 = nn.Conv2d(64, 31, 3, 1, 1)
        self.Conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.Conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.Relu = nn.ReLU()
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Conv2(self.Relu(out))

        Z = self.Conv5(self.Relu(out))
        M = self.Sig(self.Conv4(self.Relu(out)))

        out = M*out + (1-M)*Z
        out = self.Conv3_2(self.Relu(self.Conv3(self.Relu(out))))

        return out + x

class FineNet_SelfAtt_GenK(nn.Module):

    def __init__(self):
        super(FineNet_SelfAtt_GenK, self).__init__()
        self.Conv1 = nn.Conv2d(32, 31, 3, 1, 1)
        self.Conv2 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv3 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv4 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv5 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Relu = nn.ReLU()
        self.Sig  = nn.Sigmoid()


    def forward(self, x, K_Est):
        K = torch.nn.functional.upsample(K_Est, (x.shape[2], x.shape[3]), mode='bilinear')
        out = self.Conv1(torch.cat((x, K), 1))
        out = self.Conv2(self.Relu(out))
        out = self.Conv3(self.Relu(out))

        #out += x 

        Z = self.Conv5(self.Relu(out))
        M = self.Sig(self.Conv4(self.Relu(out)))

        out = M*out + (1-M)*Z

        return out + x




def Downsampler_2D(x, kernel, factor):

    _,_,m_k, n_k = kernel.shape

    if  m_k % 2 == 1: 
        pad = int((m_k - 1) / 2.)
    else:
        pad = int((m_k - factor) / 2.)
                
    padding = nn.ReplicationPad2d(pad)

    #x = padding(x)
    a,b,m,n = x.shape 
    x = padding(x)
    x_ = torch.zeros(a, b, int(m/factor), int(n/factor))

    for i in range(x.shape[1]):
        if i == 0:
             #x_ = nn.functional.conv2d(x[:, i, :, :].unsqueeze(1), kernel, stride=factor)
             x_ = nn.functional.conv2d(x[:, i, :, :].unsqueeze(1), kernel, stride=factor)
        else:
             x_0 = nn.functional.conv2d(x[:, i, :, :].unsqueeze(1), kernel, stride=factor)
             x_ = torch.cat([x_, x_0], 1)


    return x_



def GloblePooling(X):
    # X: B*C*M*N
    return X.mean(3).mean(2)

def Down_spa(X, K, factor):
    # X: B*C*M*N
    # K: B*1*k*k
    [B, _, ks, _] = K.shape

    output = []
    for i in range(B):
        output.append(torch.nn.functional.conv2d(X.narrow(0, i, 1), K.narrow(0, i, 1).repeat(X.shape[1],1,1,1), stride=factor, padding=int((ks-factor)/2), groups=X.shape[1]))
    return torch.cat(output, 0)



class GenKP_Net(nn.Module):

    def __init__(self,  Dim=[31, 3], KS=32, factor=8):
        super(GenKP_Net, self).__init__()

        self.KS    = KS
        self.Dim   = Dim
        self.factor= factor

        # Upsample K, P
        Mid_Chas = 256
        self.layerIn_K_1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.layerIn_K_2 = nn.Conv2d(64, Mid_Chas, 3, 1, 1)

        self.layerIn_P_1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.layerIn_P_2 = nn.Conv2d(64, Mid_Chas, 3, 1, 1)
        self.layerIn_P_3 = nn.Linear(Dim[1], Dim[0])

        self.PixShuf = nn.PixelShuffle(16)

        # Estimate Degeneration part
        K_Chas = self.KS**2
        P_Chas = Dim[0]**2
        KS_1 = 3

        self.layerEst_K_1 = nn.Conv2d(Dim[0], 64, 3, 1, 1)
        self.layerEst_K_2 = nn.Conv2d(64, K_Chas, 3, 1, 1)
        self.layerEst_P_1 = nn.Conv2d(Dim[0]+Dim[1], 64, 3, 1, 1)
        self.layerEst_P_2 = nn.Conv2d(64, P_Chas, 3, 1, 1)

        self.block_K = nn.Sequential(
            *[
                nn.Conv2d(K_Chas, K_Chas, KS_1, 1, int(KS_1 / 2)), nn.ReLU(),
                nn.Conv2d(K_Chas, K_Chas, KS_1, 1, int(KS_1 / 2))
            ]
        )
        self.block_P = nn.Sequential(
            *[
                nn.Conv2d(P_Chas, P_Chas, KS_1, 1, int(KS_1 / 2)), nn.ReLU(),
                nn.Conv2d(P_Chas, P_Chas, KS_1, 1, int(KS_1 / 2)),
            ]
        )
        self.Linear_P = nn.Linear(Dim[0], Dim[1])

        self.Relu = nn.ReLU()
        self.SM = nn.Softmax()

    def forward(self, X, x, y):
        # X: B*C*M*N
        # x: B*C*M*N
        # y: B*c*M*N
        # K: B*1*k*k
        # P: B*1*C*c

        [_, _, M, N] = X.shape
        k_s = self.KS
        [C, c] = self.Dim 
        [_, _, m, n] = x.shape
        #F_K = self.layerIn_K_2(self.Relu(self.layerIn_K_1(K)))
        #F_K = nn.functional.upsample(self.PixShuf(F_K), (m, n), mode='bilinear')
        #F_P = self.layerIn_P_3(self.layerIn_P_2(self.Relu(self.layerIn_P_1(P))))
        #F_P = nn.functional.upsample(self.PixShuf(F_P), (M, N), mode='bilinear')


        # ##### Part Two ##### Degeneration Estimation part
        # Estimate K from LR HSI
        out_K_1 = self.layerEst_K_2(self.Relu(self.layerEst_K_1(x)))
        out_K_2 = self.layerEst_K_2(self.Relu(self.layerEst_K_1(X)))
        #out_K = GloblePooling(self.block_K(out_K))
        out_K = torch.nn.functional.softmax(GloblePooling(out_K_2) - GloblePooling(out_K_1)).unsqueeze(2).unsqueeze(3)
        out_K = out_K.reshape(-1, 1, k_s, k_s)

        # Estimate P from HR MSI
        #y_ = torch.matmul(P.permute(0, 1, 3, 2).squeeze(1), X.reshape(-1, C, M*N)).reshape(-1, c, M, N)
        out_P = self.layerEst_P_2(self.Relu(self.layerEst_P_1(torch.cat((y, X), 1))))
        #out_P = GloblePooling(self.block_P(out_P))
        out_P = GloblePooling(out_P)
        out_P = nn.functional.softmax(self.Linear_P(out_P.reshape(-1, 1, C, C)), 2)

        return out_K, out_P


class FineNet_SelfAtt_InputK_P(nn.Module):

    def __init__(self, Dim=[31,3]):
        super(FineNet_SelfAtt_InputK_P, self).__init__()
        self.Conv1 = nn.Conv2d(33, 31, 3, 1, 1)
        self.Conv2 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv3 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv4 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv5 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Relu = nn.ReLU()
        self.Sig  = nn.Sigmoid()

        # Upsample K, P
        Mid_Chas = 256
        self.layerIn_K_1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.layerIn_K_2 = nn.Conv2d(64, Mid_Chas, 3, 1, 1)

        self.layerIn_P_1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.layerIn_P_2 = nn.Conv2d(64, Mid_Chas, 3, 1, 1)
        self.layerIn_P_3 = nn.Linear(Dim[1], Dim[0])

        self.PixShuf = nn.PixelShuffle(16)

    def forward(self, x, K, P):
        # Upsample K, P into feature map
        [_, _, m, n] = x.shape
        F_K = self.layerIn_K_2(self.Relu(self.layerIn_K_1(K)))
        F_K = nn.functional.upsample(self.PixShuf(F_K), (m, n), mode='bilinear')
        F_P = self.layerIn_P_3(self.layerIn_P_2(self.Relu(self.layerIn_P_1(P))))
        F_P = nn.functional.upsample(self.PixShuf(F_P), (m, n), mode='bilinear')

        out = self.Conv1(torch.cat((x, F_K, F_P), 1))
        out = self.Conv2(self.Relu(out))
        out = self.Conv3(self.Relu(out))

        #out += x 

        Z = self.Conv5(self.Relu(out))
        M = self.Sig(self.Conv4(self.Relu(out)))

        out = M*out + (1-M)*Z

        return out + x


class FineNet_SelfAtt_InputK_P_V2(nn.Module):

    def __init__(self, Dim=[31,3]):
        super(FineNet_SelfAtt_InputK_P_V2, self).__init__()
        self.Conv1 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv2 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv3 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv4 = nn.Conv2d(32, 31, 3, 1, 1)
        self.Conv5 = nn.Conv2d(32, 31, 3, 1, 1)
        self.Conv6 = nn.Conv2d(62, 31, 3, 1, 1)
        self.Conv7 = nn.Conv2d(62, 31, 3, 1, 1)
        #self.Conv6 = nn.Conv2d(93, 31, 3, 1, 1)
        #self.Conv7 = nn.Conv2d(93, 31, 3, 1, 1)
        #self.Conv6 = nn.Conv2d(31, 31, 3, 1, 1)
        #self.Conv7 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Relu = nn.ReLU(inplace=True)
        self.Linear_1 = nn.Linear(31, 16)
        self.Linear_2 = nn.Linear(16, 31)
        self.Sig  = nn.Sigmoid()

        # Upsample K, P
        Mid_Chas = 256
        self.layerIn_K_1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.layerIn_K_2 = nn.Conv2d(64, Mid_Chas, 3, 1, 1)

        self.layerIn_P_1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.layerIn_P_2 = nn.Conv2d(64, Mid_Chas, 3, 1, 1)
        self.layerIn_P_3 = nn.Linear(Dim[1], Dim[0])

        self.PixShuf = nn.PixelShuffle(16)

    def forward(self, x, K, P):
        # Upsample K, P into feature map
        [_, _, m, n] = x.shape
        F_K = self.layerIn_K_2(self.Relu(self.layerIn_K_1(K)))
        F_K = nn.functional.upsample(self.PixShuf(F_K), (m, n), mode='bilinear')
        F_P = self.layerIn_P_3(self.layerIn_P_2(self.Relu(self.layerIn_P_1(P))))
        F_P = nn.functional.upsample(self.PixShuf(F_P), (m, n), mode='bilinear')

        #out = self.Conv1(torch.cat((x, F_K, F_P), 1))
        out = self.Conv1(x)
        out = self.Conv2(self.Relu(out))
        out = self.Conv3(self.Relu(out))

        out_a = self.Conv5(torch.cat((out, F_K), 1)).mean(1).unsqueeze(1).repeat(1,x.shape[1],1,1)
        out_b = self.Conv4(torch.cat((out, F_P), 1))
        SE_b  = self.Sig(self.Linear_2(self.Relu(self.Linear_1(out_b.mean(3).mean(2))))).unsqueeze(2).unsqueeze(3)
        out_b = SE_b*out_b

        #Z = out_a*out + out_b*out
        Z = out_a*out + out_b

        Z = self.Conv6(self.Relu(torch.cat((out, Z), 1)))
        M = self.Sig(self.Conv7(self.Relu(torch.cat((out, Z), 1))))
        #Z = self.Conv6(self.Relu(torch.cat((out, out_a, out_b), 1)))
        #M = self.Sig(self.Conv7(self.Relu(torch.cat((out, out_a, out_b), 1))))
        #Z = self.Conv6(self.Relu(out+ Z))
        #M = self.Sig(self.Conv7(self.Relu(out+Z)))

        out = M*out + (1-M)*Z

        return out + x


class FineNet_SelfAtt_InputK_P_V2_Ablation(nn.Module):

    def __init__(self, Dim=[31,3]):
        super(FineNet_SelfAtt_InputK_P_V2_Ablation, self).__init__()
        self.Conv1 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv2 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv3 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv4 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv5 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv6 = nn.Conv2d(62, 31, 3, 1, 1)
        self.Conv7 = nn.Conv2d(62, 31, 3, 1, 1)
        #self.Conv6 = nn.Conv2d(93, 31, 3, 1, 1)
        #self.Conv7 = nn.Conv2d(93, 31, 3, 1, 1)
        #self.Conv6 = nn.Conv2d(31, 31, 3, 1, 1)
        #self.Conv7 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Relu = nn.ReLU()
        self.Linear_1 = nn.Linear(31, 16)
        self.Linear_2 = nn.Linear(16, 31)
        self.Sig  = nn.Sigmoid()

        # Upsample K, P
        Mid_Chas = 256
        self.layerIn_K_1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.layerIn_K_2 = nn.Conv2d(64, Mid_Chas, 3, 1, 1)

        self.layerIn_P_1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.layerIn_P_2 = nn.Conv2d(64, Mid_Chas, 3, 1, 1)
        self.layerIn_P_3 = nn.Linear(Dim[1], Dim[0])

        self.PixShuf = nn.PixelShuffle(16)

    def forward(self, x, K, P):
        # Upsample K, P into feature map
        '''[_, _, m, n] = x.shape
        F_K = self.layerIn_K_2(self.Relu(self.layerIn_K_1(K)))
        F_K = nn.functional.upsample(self.PixShuf(F_K), (m, n), mode='bilinear')
        F_P = self.layerIn_P_3(self.layerIn_P_2(self.Relu(self.layerIn_P_1(P))))
        F_P = nn.functional.upsample(self.PixShuf(F_P), (m, n), mode='bilinear')
        '''

        #out = self.Conv1(torch.cat((x, F_K, F_P), 1))
        out = self.Conv1(x)
        out = self.Conv2(self.Relu(out))
        out = self.Conv3(self.Relu(out))

        out_a = self.Conv5(out).mean(1).unsqueeze(1).repeat(1,x.shape[1],1,1)
        out_b = self.Conv4(out)
        SE_b  = self.Sig(self.Linear_2(self.Relu(self.Linear_1(out_b.mean(3).mean(2))))).unsqueeze(2).unsqueeze(3)
        out_b = SE_b*out_b

        #Z = out_a*out + out_b*out
        Z = out_a*out + out_b

        Z = self.Conv6(self.Relu(torch.cat((out, Z), 1)))
        M = self.Sig(self.Conv7(self.Relu(torch.cat((out, Z), 1))))
        #Z = self.Conv6(self.Relu(torch.cat((out, out_a, out_b), 1)))
        #M = self.Sig(self.Conv7(self.Relu(torch.cat((out, out_a, out_b), 1))))
        #Z = self.Conv6(self.Relu(out+ Z))
        #M = self.Sig(self.Conv7(self.Relu(out+Z)))

        out = M*out + (1-M)*Z

        return out + x


class FineNet_SelfAtt_InputK_P_V3(nn.Module):

    def __init__(self, Dim=[31,3]):
        super(FineNet_SelfAtt_InputK_P_V3, self).__init__()
        self.Conv1 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv2 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv3 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv1_x = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv2_x = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv3_x = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv1_Y = nn.Conv2d(3, 31, 3, 1, 1)
        self.Conv2_Y = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv3_Y = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv4 = nn.Conv2d(32, 31, 3, 1, 1)
        self.Conv5 = nn.Conv2d(32, 31, 3, 1, 1)
        #self.Conv6 = nn.Conv2d(62, 31, 3, 1, 1)
        #self.Conv7 = nn.Conv2d(62, 31, 3, 1, 1)
        self.Conv6 = nn.Conv2d(93, 31, 3, 1, 1)
        self.Conv7 = nn.Conv2d(93, 31, 3, 1, 1)
        #self.Conv6 = nn.Conv2d(31, 31, 3, 1, 1)
        #self.Conv7 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Relu = nn.ReLU()
        self.Linear_1 = nn.Linear(31, 16)
        self.Linear_2 = nn.Linear(16, 31)
        self.Sig  = nn.Sigmoid()

        # Upsample K, P
        Mid_Chas = 256
        self.layerIn_K_1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.layerIn_K_2 = nn.Conv2d(64, Mid_Chas, 3, 1, 1)

        self.layerIn_P_1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.layerIn_P_2 = nn.Conv2d(64, Mid_Chas, 3, 1, 1)
        self.layerIn_P_3 = nn.Linear(Dim[1], Dim[0])

        self.PixShuf = nn.PixelShuffle(16)

    def forward(self, X, x, Y, K, P):
        # Upsample K, P into feature map
        [_, _, m, n] = X.shape
        F_K = self.layerIn_K_2(self.Relu(self.layerIn_K_1(K)))
        F_K = nn.functional.upsample(self.PixShuf(F_K), (m, n), mode='bilinear')
        F_P = self.layerIn_P_3(self.layerIn_P_2(self.Relu(self.layerIn_P_1(P))))
        F_P = nn.functional.upsample(self.PixShuf(F_P), (m, n), mode='bilinear')

        #out = self.Conv1(torch.cat((x, F_K, F_P), 1))
        out_X = self.Conv3(self.Relu(self.Conv2(self.Relu(self.Conv1(X)))))
        out_x = self.Conv3_x(self.Relu(self.Conv2_x(self.Relu(self.Conv1_x(x)))))
        out_Y = self.Conv3_Y(self.Relu(self.Conv2_Y(self.Relu(self.Conv1_Y(Y)))))

        out_a = self.Conv5(torch.cat((out_x, F_K), 1)).mean(1).unsqueeze(1).repeat(1,X.shape[1],1,1)
        out_b = self.Conv4(torch.cat((out_Y, F_P), 1))
        SE_b  = self.Sig(self.Linear_2(self.Relu(self.Linear_1(out_b.mean(3).mean(2))))).unsqueeze(2).unsqueeze(3)
        out_b = SE_b*out_b

        #Z = out_a*out + out_b*out

        Z = self.Conv6(self.Relu(torch.cat((out_X, out_a*out_x, out_b*out_Y), 1)))
        M = self.Sig(self.Conv7(self.Relu(torch.cat((out_X, out_a*out_x, out_b*out_Y), 1))))
        #Z = self.Conv6(self.Relu(torch.cat((out, out_a, out_b), 1)))
        #M = self.Sig(self.Conv7(self.Relu(torch.cat((out, out_a, out_b), 1))))
        #Z = self.Conv6(self.Relu(out+ Z))
        #M = self.Sig(self.Conv7(self.Relu(out+Z)))

        out = M*out_X + (1-M)*Z

        return out + x






