import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_ssim import SSIM

from fastai.conv_learner import *

# part of architecture is copied from fastai library

# transfer learning from pretrain resnet34
f = resnet34
cut,lr_cut = model_meta[f]

def get_base():
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)

# a warped forward hook 
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()

class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p))

class Pose(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(inc, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
    def forward(self, x):
        x = self.body(x)
        batch, c, h, w = x.size()
        return x.view(batch, 2, c//2)

class Depth34(nn.Module):
    def __init__(self, rn, ochannel):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,256)
        self.up2 = UnetBlock(256,128,256)
        self.up3 = UnetBlock(256,64,256)
        self.up4 = UnetBlock(256,64,256)
        self.up5 = UnetBlock(256,3,16)
        self.up6 = nn.ConvTranspose2d(16, ochannel, 1)
        
    def forward(self,x):
        inp = x
        x = F.relu(self.rn(x))
        
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x, inp)
        x = self.up6(x)
        return x, sfs
    
    def close(self):
        for sf in self.sfs: sf.remove()

class TriDepth(nn.Module):
    def __init__(self, rn, ochannel):
        super().__init__()
        self.depth = Depth34(rn, ochannel) 
        self.pose = Pose(256*3)

    def forward(self, x1, x2, x3):
        d1, ft1 = self.depth(x1)
        d2, ft2 = self.depth(x2) 
        d3, ft3 = self.depth(x3)
        poses_x2 = self.pose(torch.cat((ft1,ft2,ft3), dim=1))

        return d1, d2, d3, poses_x2




class TriDepthModel():
    def __init__(self,model,name='tridepth'):
        self.model,self.name = model,name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.depth.rn), [lr_cut]))
        return lgs + [children(self.model.depth)[1:]] + [children(self.model.pose)]

class Offset(nn.Module):
    '''
        xnew = Rx + td
        where R is determined by camera relative pose change using Rodrigues Rotation Formular
    '''
    def __init__(self):
        super().__init__()
        self.register_buffer('o', torch.zeros([1,1]))
        self.register_buffer('eye', torch.eyes(3).unsqueeze())

    def factorize(self, vecs, dim):
        mags = vecs.norm(p=2, dim=dim, keepdim=True)
        return vecs/mag, mags

    def rot_vec2mat(self, rot_vecs):
        batch, _ = rot_vecs.size()
        directs, angles = self.factorize(vecs, 1)
        
        K0 = directs[:,:1]
        K1 = directs[:,:2]
        K2 = directs[:,2:]
        
        o = self.o.repeat(batch, dim=0)
        eye = self.eye.repeat(batch, dim=0)
        K = torch.cat([o, -K2, K1, K2, o, K0, -K1, K0, o]).view(-1, 3, 3)
        return eye + K * angles.sin() + torch.bmm(K,K) * (1-angles.cos())
    
    def forward(self, pose, inv_depth):
        rot_vecs = pose[:,:3]
        tran_vecs = pose[:, 3:]

        rot_mats = self.rot_vec2mat(rot_vecs)

        offset_xyz = torch.matmul(rot_mats[:, :2].unsqueeze(1).unsqueeze(2), xy.unsqueeze(-1)).squeeze(-1) +
            rot_mats[:, 2:] +
            tran_vecs.view( *(tran_vecs.size()), 1, 1)  * inv_depth       
        offset = offset_xyz[:, :2] / offset_xyz[:, 2]
        return offset

class BilinearProj(nn.Module):
    """
        bilinear sampler
        warp the input image to the target image given the offset
    """
    def __init__(self):
        super().__init__()
        self.padding = 1
        self.pad = nn.ReflectionPad2d(self.padding)

    def forward(self, imgs, offset):
        # offset in pixel level
        batch, c, height , width = imgs.size()
        
        x, y = torch.meshgrid([torch.arange(height), torch.arange(width)])
        x = (x + offset[:,0] + self.padding[:, 0])/(height/2) + 1
        y = (y + offset[:,1] + self.padding[:, 1])/(width/2) + 1
        xy = torch.stack([x, y], dim=0).expand(batch, 1, height, width)
        
        sampled = F.grid_sample(imgs, xy.expand(batch,3,height,width))
        return sampled

class TriAppearanceLoss(nn.Module):
    def __init__(self, scale=0.01, ws=11):

        self.offset= Offset()
        self.sampler = BilinearProj()

        self.SSIM = SSIM(window_size = ws)
        self.L1 = nn.L1Loss()
        self.scale = scale

    def forward(self, d1, d3, poses_x2, x1, x2, x3):

        offset1 = self.offset.forward( d1, poses_x2[:,0])
        offset3 = self.offset.forward( d3, poses_x2[:,1])

        x12 = self.sampler.forward(x1, offset1)
        x32 = self.sampler.forward(x3, offset3)

        ssimloss = self.SSIM(x12, x2) + self.SSIM(x32, x2)
        l1loss = self.L1(x12, x2) + self.L1(x32, x2)

        return ssimloss + scale * l1loss

class LaplacianLayer(nn.Module):
    def __init__(self):
        super(LaplacianLayer, self).__init__()
        w_nom = torch.FloatTensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1,1,3,3)
        w_den = torch.FloatTensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]]).view(1,1,3,3)
        self.register_buffer('w_nom', w_nom)
        self.register_buffer('w_den', w_den)

    def forward(self, input, do_normalize=True):
        assert(input.dim() == 2 or input.dim()==3 or input.dim()==4)
        input_size = input.size()
        if input.dim()==4:
            x = input.view(input_size[0]*input_size[1], 1,
                            input_size[2], input_size[3])
        elif input.dim()==3:
            x = input.unsqueeze(1)
        else:
            x = input.unsqueeze(0).unsqueeze(0)
        x_nom = torch.nn.functional.conv2d(input=x,
                        weight=Variable(self.w_nom),
                        stride=1,
                        padding=0)
        if do_normalize:
            x_den = torch.nn.functional.conv2d(input=x,
                        weight=Variable(self.w_den),
                        stride=1,
                        padding=0)
            # x_den = x.std() + 1e-5
            x = (x_nom.abs()/x_den)
        else:
            x = x_nom.abs()
        if input.dim() == 4:
            return x.view(input_size[0], input_size[1], input_size[2]-2, input_size[3]-2)
        elif input.dim() == 3:
            return x.squeeze(1)
        elif input.dim() == 2:
            return x.squeeze(0).squeeze(0)

