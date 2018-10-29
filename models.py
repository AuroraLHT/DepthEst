import torch
import torch.nn as nn
import torch.nn.functional as F

import fastai
from fastai.conv_learner import *

from math import exp

# part of architecture is copied from fastai library

# transfer learning from pretrain resnet34 or resnet18

EPS = 1e-10
# f = resnet34
f = resnet18
cut,lr_cut = model_meta[f]

# the scalling factor for the disparity is one of the most misterious thing in the universe

# Geonet use 5 for resnet50, 10 for vgg https://github.com/yzcjtr/GeoNet/blob/master/geonet_nets.py

# Left-Right consistency use 0.3 ???

# SfMLearner use 10 for vgg https://github.com/tinghuiz/SfMLearner/blob/master/nets.py
"""
# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01
"""

def meshgrid_fromHW(H, W, dtype=torch.FloatTensor):
    x = torch.arange(W).type(dtype)
    y = torch.arange(H).type(dtype)
    return meshgrid(x, y)

def xy_fromHW(H, W):
    x = torch.arange(W)
    y = torch.arange(H)
    return x, y

def meshgrid(x ,y):
    imW = x.size(0)
    imH = y.size(0)
    X = x.unsqueeze(0).repeat(imH, 1)
    Y = y.unsqueeze(1).repeat(1, imW)
    return X, Y

def get_base(f, cut):
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)

def get_resnet():
    return get_base(f, cut)

# a warped forward hook 
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()


class Conv(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride, padding, activation_func=nn.ELU()):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_nc,
                              out_channels=output_nc,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=0,
                              bias=True)
        self.activation_fn = activation_func
        self.pad_fn = nn.ReplicationPad2d(padding)

    def forward(self, input):
        if self.activation_fn == None:
            return self.conv(self.pad_fn(input))
        else:
            return self.activation_fn(self.conv(self.pad_fn(input)))

class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p))

class Pose(nn.Module):
    def __init__(self, inc, mag_scalor = 1):
        """
            According to the "Digging Into Self-Supervised Monocular Depth Estimation", the pose decoder should be the same as the last three layer
            of the pose net defined in https://github.com/tinghuiz/SfMLearner/blob/master/nets.py line:18 
            def pose_exp_net(tgt_image, src_image_stack, do_exp=True, is_training=True)
        """
        super().__init__()
        self.ps = 6
        self.multi = 2
        self.mag_scalor = mag_scalor

        self.body = nn.Sequential(
            nn.Conv2d(inc, 256, 3, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.ps*self.multi, 1, bias=True), # set biad = True would make the model remember the common driving speed in non-shuffle training set
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.tran_mag = 0.001
        self.rot_mag= 0.01
        
    def forward(self, x):
        x = self.body(x)
        batch, c, h, w = x.size()
        x = x.view(batch, self.multi, self.ps)
        
        transistion = x[:, :, :3] * self.tran_mag
        rotation = x[:, :, 3:] * self.rot_mag
        return transistion, rotation

class Depth34(nn.Module):
    def __init__(self, rn, ochannel):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,256)
        self.up2 = UnetBlock(256+1,128,128)
        self.up3 = UnetBlock(128+1,64,64)
        self.up4 = UnetBlock(64+1,64,64)
        # self.up5 = UnetBlock(64+1,3,16)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64+1, 16, 2, stride=2),
            nn.ELU(inplace=True)
        )

        self.d1 =  Conv( 256, 1, 3, 1, 1, activation_func=nn.Sigmoid() )
        self.d2 =  Conv( 128, 1, 3, 1, 1, activation_func=nn.Sigmoid() )
        self.d3 =  Conv( 64, 1, 3, 1, 1, activation_func=nn.Sigmoid() )
        self.d4 =  Conv( 64, 1, 3, 1, 1, activation_func=nn.Sigmoid() )
        self.d5 =  Conv( 16, 1, 3, 1, 1, activation_func=nn.Sigmoid() )   
    #    self.fuse2 = DepthFuseBlock()
    #    self.fuse3 = DepthFuseBlock()
    #    self.fuse4 = DepthFuseBlock()
    #    self.fuse5 = DepthFuseBlock()
    #     self.op_norm = torch.nn.InstanceNorm2d(1)
    
        self.MIN_DISP = 0.01
        self.DISP_SCALING = 5

    def forward(self, x, enc_only=False):
        inp = x
        x = F.elu(self.rn(x))        
        depthmaps = []

        if not enc_only:
            x = self.up1(x, self.sfs[3].features)
            d1 = self.d1(x)
#             depthmaps.append(d1)
            
            x = torch.cat((x, d1), dim=1)
            x = self.up2(x, self.sfs[2].features)
#            d2 = self.fuse2( d1, self.d2(x) )
            d2 = self.d2(x)
#             depthmaps.append(d2)
            
            x = torch.cat((x, d2), dim=1)
            x = self.up3(x, self.sfs[1].features)
#            d3 = self.fuse3( d2, self.d3(x) )
            d3 = self.d3(x)        
#             depthmaps.append(d3)
            
            x = torch.cat((x, d3), dim=1)
            x = self.up4(x, self.sfs[0].features)
#            d4 = self.fuse4( d3, self.d4(x) )
            d4 = self.d4(x)
#             depthmaps.append(d4)
            
            x = torch.cat((x, d4), dim=1)
            x = self.up5(x, inp)
#             d5 = self.fuse5( d4, self.d5(x) )
            d5 = self.d5(x)
            depthmaps.append(d5)
            
            depthmaps = [ d * self.DISP_SCALING + self.MIN_DISP for d in depthmaps ]
                            
            depthmaps.reverse()

        # return disparity map and the output of the encoder
        return depthmaps, self.sfs[3].features
        # return DISP_SCALING * F.sigmoid(x) + MIN_DISP, self.sfs[3].features #, x
        #return F.sigmoid(x), self.sfs[3].features #, x
        #return F.sigmoid(self.op_norm(x)), self.sfs[3].features #, x
        #return 1/torch.clamp(F.relu(x), 0.1, 500), self.sfs[3].features 
        #return x, self.sfs[3].features 
    
    def close(self):
        for sf in self.sfs: sf.remove()

class Depth18(nn.Module):
    def __init__(self, rn, ochannel):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,256)
        self.up2 = UnetBlock(256+1,128,128)
        self.up3 = UnetBlock(128+1,64,64)
        self.up4 = UnetBlock(64+1,64,64)
        # self.up5 = UnetBlock(64+1,3,16)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64+1, 16, 2, stride=2),
            nn.ELU(inplace=True)
        )

        self.d1 =  Conv( 256, 1, 3, 1, 1, activation_func=nn.Sigmoid() )
        self.d2 =  Conv( 128, 1, 3, 1, 1, activation_func=nn.Sigmoid() )
        self.d3 =  Conv( 64, 1, 3, 1, 1, activation_func=nn.Sigmoid() )
        self.d4 =  Conv( 64, 1, 3, 1, 1, activation_func=nn.Sigmoid() )
        self.d5 =  Conv( 16, 1, 3, 1, 1, activation_func=nn.Sigmoid() )   
    
        self.MIN_DISP = 0.01
        self.DISP_SCALING = 5

    def forward(self, x, enc_only=False):
        inp = x
        x = F.elu(self.rn(x))        
        depthmaps = []

        if not enc_only:
            x = self.up1(x, self.sfs[3].features)
            d1 = self.d1(x)
#             depthmaps.append(d1)
            
            x = torch.cat((x, d1), dim=1)
            x = self.up2(x, self.sfs[2].features)
#            d2 = self.fuse2( d1, self.d2(x) )
            d2 = self.d2(x)
#             depthmaps.append(d2)
            
            x = torch.cat((x, d2), dim=1)
            x = self.up3(x, self.sfs[1].features)
#            d3 = self.fuse3( d2, self.d3(x) )
            d3 = self.d3(x)        
            depthmaps.append(d3)
            
            x = torch.cat((x, d3), dim=1)
            x = self.up4(x, self.sfs[0].features)
#            d4 = self.fuse4( d3, self.d4(x) )
            d4 = self.d4(x)
            depthmaps.append(d4)
            
            x = torch.cat((x, d4), dim=1)
            x = self.up5(x)
#             x = self.up5(x, inp)
#             d5 = self.fuse5( d4, self.d5(x) )
            d5 = self.d5(x)
            depthmaps.append(d5)
            
            depthmaps = [ d * self.DISP_SCALING + self.MIN_DISP for d in depthmaps ]
                            
            depthmaps.reverse()

        return depthmaps, self.sfs[3].features

    
    def close(self):
        for sf in self.sfs: sf.remove()            
            
class TriDepth(nn.Module):
    def __init__(self, rn, ochannel):
        super().__init__()
        self.depth = Depth18(rn, ochannel) 
#         self.depth = Depth34(rn, ochannel) 
        self.pose = Pose(256*3)
        #self.train = train
        
    def forward(self, x1, x2, x3):

        d1, ft1 = self.depth(x1, enc_only=True) # src
        d2, ft2 = self.depth(x2, enc_only=False) # target
        d3, ft3 = self.depth(x3, enc_only=True) # src

#         if self.train:
#             d1, ft1 = self.depth(x1, enc_only=True) # src
#             d2, ft2 = self.depth(x2, enc_only=False) # target
#             d3, ft3 = self.depth(x3, enc_only=True) # src
#         else:
#             d1, ft1 = self.depth(x1, enc_only=False) # src
#             d2, ft2 = self.depth(x2, enc_only=False) # target
#             d3, ft3 = self.depth(x3, enc_only=False) # src            
        trans, rotation = self.pose(torch.cat((ft1,ft2,ft3), dim=1))

        return d1, d2, d3, trans, rotation

class TriDepthModel():
    def __init__(self,model,name='tridepth'):
        self.model,self.name = model,name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.depth.rn), [lr_cut]))
        return lgs + [children(self.model.depth)[1:]] + [children(self.model.pose)]

class Offset3(nn.Module):
    '''
        xnew = Rx + td
        where R is determined by camera relative pose change using Rodrigues Rotation Formular
    '''
    def __init__(self):
        super().__init__()
        self.register_buffer('o', torch.zeros([1,1]).type(torch.FloatTensor))
        self.register_buffer('eye', torch.eye(3).type(torch.FloatTensor).unsqueeze(0))
        self.register_buffer('filler', torch.FloatTensor([0,0,0,1]).unsqueeze(0))
        
    def factorize(self, vecs, dim):
        mags = vecs.norm(p=2, dim=dim, keepdim=True)
        return vecs/mags, mags

    def rot_vec2mat(self, rot_vecs):
        b, _ = rot_vecs.size()
        directs, angles = self.factorize(rot_vecs, 1)
        
        K0 = directs[:,:1]
        K1 = directs[:,1:2]
        K2 = directs[:,2:]
        
        o = Variable(self.o.repeat(b, 1))
        eye = Variable(self.eye.repeat(b, 1, 1))
        
        #print(K0.type, K2.type, K1.type, o.type, eye.type)
        angles = angles.unsqueeze(-1)
        K = torch.cat((o, -K2, K1, K2, o, K0, -K1, K0, o), 1).view(-1, 3, 3) # form a cpro matrix
        return eye + K * angles.sin() + torch.bmm(K,K) * (1-angles.cos()) # using the R formular
    
    def pose_vec2mat(self, trans, rotation):
        b = trans.size(0)
        rot_vecs = rotation
        tran_vecs = trans

        rot_mats = self.rot_vec2mat(rot_vecs)

        # stack these trans together w.r.t the column
        pose = torch.cat((rot_mats, tran_vecs.unsqueeze(-1)), dim=-1)
        pose = torch.cat((pose, V(self.filler.repeat(b, 1, 1))), dim=-2)

        return pose
        
    def pixel2cam(self, inv_depth, pixel_coors, inv_intrinsics, ishomo=True):
        b, c, h , w = inv_depth.size()
        inv_depth = inv_depth.permute(0,2,3,1)
        pixel_coors = pixel_coors.unsqueeze(-1)
        inv_intrinsics = inv_intrinsics.unsqueeze(1).unsqueeze(2)
        
        cam_coors = torch.matmul(inv_intrinsics, pixel_coors).squeeze(-1)
        #pdb.set_trace()
        # cam_coors = cam_coors * inv_depth
        cam_coors = cam_coors / inv_depth
        if ishomo:
            cam_coors = torch.cat((cam_coors, V(torch.ones(b, h, w, 1))), dim=-1)
        return cam_coors
    
    def cam2pixel(self, cam_coords, proj):
        
        b, h, w, c = cam_coords.size()
        proj = proj.unsqueeze(1).unsqueeze(2)
        cam_coords = cam_coords.unsqueeze(-1)
        #pdb.set_trace()
        unnormalized_pixel_coords = torch.matmul(proj, cam_coords).squeeze(-1)
        
        x_u = unnormalized_pixel_coords[:, :, :, 0]
        y_u = unnormalized_pixel_coords[:, :, :, 1]
        z_u = unnormalized_pixel_coords[:, :, :, 2]
        
        x_n = x_u / (z_u + EPS)
        y_n = y_u / (z_u + EPS)
        
        #pdb.set_trace()
        return x_n, y_n, V(z_u.data>EPS)
    
    def forward(self, trans, rotation, inv_depth, camera):
        
        """
            Params:
                pose: relative pose, N X 6 vectors,
                    1-3 is the transition vector
                    4-6 is the rotation vector in eular representation
                inv_depth: invered depth map
                camera: intrinsic camera parameters NX4: (fx, fy, cx, cy)
            Return:
                tkx: transformed camera pixel coordinate - x-component
                tky: transformed camera pixel coordinate - y-component
                dmask: binary map of pixel that keeps track in the future
        """
        b, c, h, w = inv_depth.size()
        
        # build the camera intrinsic matrix
        camera = camera.data
        cx = camera[:, 2:3].contiguous()
        cy = camera[:, 3:4].contiguous()
        fx = camera[:, 0:1].contiguous()
        fy = camera[:, 1:2].contiguous()
        
        o = self.o.repeat(b,1)
        intrinsics = torch.cat(
            [fx, o, cx, o,
             o, fy, cy, o,
             o, o, o+1, o,
             o, o, o, o+1], dim=-1).view(b,4,4)
        
        inv_intrinsics = torch.cat(
            [1/fx, o, -cx/fx,
             o, 1/fy, -cy/fy,
             o, o, o+1], dim=-1).view(b,3,3)
        
        intrinsics = V(intrinsics)
        inv_intrinsics = V(inv_intrinsics)

        pose = self.pose_vec2mat(trans, rotation)
       
        # grip points preperation
        px, py = meshgrid_fromHW(h, w, dtype=type(inv_depth.data))
        pixel_coords = torch.stack([px, py, torch.ones_like(px)], dim=-1).repeat(b, 1, 1, 1)
        pixel_coords = V(pixel_coords)
        
        cam_coords = self.pixel2cam(inv_depth, pixel_coords, inv_intrinsics)
        
        proj_tgt_cam_to_src_pixel = torch.matmul(intrinsics, pose)
        #pdb.set_trace()
        x_n, y_n, dmask = self.cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
        #pdb.set_trace()
        return x_n, y_n, dmask.type_as(inv_depth)
    
class BilinearProj(nn.Module):
    """
        bilinear sampler
        warp the input image to the target image given the offset
    """
    def __init__(self):
        super().__init__()
        #self.padding = 1
        #self.pad = nn.ReflectionPad2d(self.padding)

    def forward(self, imgs, kx, ky):
        """
            Param:
                imgs : batch of images in Variable Type
                kx: the new location of the tranformed pixel on camera x axis 
                ky: the new location of the tranformed pixel on camera y axis
            Return:
                sampled : sampled image from imgs
                in_view_mask : binary masks show whether the pixel is out of boundary
        """
        batch, c, h , w = imgs.size()

        # n_kx stands for normalized camera points x component, range from (-1, 1)       
        n_kx = kx/((w-1)/2) - 1
        n_ky = ky/((h-1)/2) - 1
        # shape of rcxy should be B X H X W X 2
        n_kxy = torch.stack([n_kx, n_ky], dim=-1)
        
        sampled = F.grid_sample(imgs, n_kxy, mode='bilinear')  
        in_view_mask = V(((n_kx.data > -1+2/w) & (n_kx.data < 1-2/w) & (n_ky.data > -1+2/h) & (n_ky.data < 1-2/h)).type_as(imgs.data))
        return sampled, in_view_mask

def l1_loss(x1, x2, mask):
#     size = mask.size()
#     masksum = mask.view(size[0], size[1], -1).sum(-1, keepdim=True) + 1
#     diffs = torch.abs(mask*(x1-x2)).view(size[0], size[1], -1).sum(-1, keepdim=True)
#     diffs = torch.sum(diffs/masksum, 1)
#     return torch.mean(diffs)
    return F.l1_loss(x1, x2)
    


# Copy from pytorch_ssim repo

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    #  filters tensor (out_channels x in_channels/groups x kH x kW)
    window = V(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, mask, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1*mask, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2*mask, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    ssim_map = 1 - ssim_map
#    size = mask.size()
#    masksum = mask.view(size[0], size[1], -1).sum(-1, keepdim=False) + 1
#     pdb.set_trace()
#    ssim_map = ssim_map.view(size[0], size[1], -1).sum(-1, keepdim=False)
#     pdb.set_trace()
#    ssim_map = (ssim_map/masksum).sum(-1, keepdim=False)
#     pdb.set_trace()
#    return torch.mean(ssim_map)
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size = 11, channel = 3, size_average = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        #self.window = create_window(window_size, self.channel)
        self.register_buffer('window', create_window(window_size, self.channel) )

    def forward(self, img1, img2, mask):
        (_, channel, _, _) = img1.size()

        #if channel == self.channel and self.window.data.type() == img1.data.type():
        #    window = self.window
        #else:
        #    window = create_window(self.window_size, channel)
           
        #    if img1.is_cuda:
        #        window = window.cuda(img1.get_device())
        #    window = window.type_as(img1)
           
        #    self.window = window
        #    self.channel = channel
        return _ssim(img1, img2, mask, self.window, self.window_size, self.channel, self.size_average)


class TriAppearanceLoss(nn.Module):
    def __init__(self, scale=0.5):
        super().__init__()
        self.offset = Offset3()
        self.sampler = BilinearProj()       
        self.scale = float(scale)        
        self.ssim_loss = SSIM()
        self.l1_loss = l1_loss
        #self.imgds = DownSampleLayer(chan=3)
        
    def forward(self, d2s, trans, rotation, x1, x2, x3, camera):
        l1losses = []
        ssimlosses = []
        for i, d2 in enumerate(d2s):            
            
            cx12, cy12, d_mask12 = self.offset.forward(trans[:, 0], rotation[:, 0], inv_depth = d2, camera = camera)
            #cx32, cy32, d_mask32 = self.offset.forward(trans[:, 1], rotation[:, 1], inv_depth = d2, camera = camera)
            #cx32, cy32, d_mask32 = self.offset2.forward(trans[:, 1], rotation[:, 1], inv_depth = d2, camera = camera)

            x12, in_mask12 = self.sampler.forward(x1, cx12, cy12)
#             x32, in_mask32 = self.sampler.forward(x3, cx32, cy32)
#             x32, in_mask32 = self.sampler2.forward(x3, cx32, cy32)

            mask12 = (d_mask12*in_mask12).unsqueeze(1)
#             mask32 = (d_mask32*in_mask32).unsqueeze(1)
            
            mask12.requires_grad = False
#             mask32.requires_grad = False
            
            # loss on original scale
            l1losses.append(self.l1_loss(x12, x2, mask12))
            ssimlosses.append(self.ssim_loss(x12, x2, mask12))       
            
#             l1losses.append( self.l1_loss(x12, x2, mask12) + self.l1_loss(x32, x2, mask32) )
#             ssimlosses.append( self.ssim_loss(x12, x2, mask12) + self.ssim_loss(x32, x2, mask32) )       
        
        l1loss = torch.mean(torch.cat(l1losses, dim=0))
        ssimloss = torch.mean(torch.cat(ssimlosses, dim=0))

        return (1-self.scale) * ssimloss + self.scale * l1loss, ((1-self.scale) * ssimloss, self.scale * l1loss)
        #return ssimloss + self.scale * l1loss, (ssimloss, self.scale * l1loss)
        

class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def grad_x(self, target):
        return target[:,:,:,:-1] - target[:,:,:,1:]
    
    def grad_y(self, target):
        return target[:,:,:-1] - target[:,:,1:]
    
    def forward(self, imgs, ds):
        img_grad_y = self.grad_y(imgs)
        img_grad_x = self.grad_x(imgs)
        
        disp_grad_y = self.grad_y(ds)
        disp_grad_x = self.grad_x(ds)
        
        weight_x = torch.mean( torch.exp( -torch.abs(img_grad_x)), dim=1, keepdim=True ) 
        weight_y = torch.mean( torch.exp( -torch.abs(img_grad_y)), dim=1, keepdim=True ) 
        
        loss_x = torch.abs(disp_grad_x) * weight_x
        loss_y = torch.abs(disp_grad_y) * weight_y
#         pdb.set_trace()
        return torch.mean(loss_x) + torch.mean(loss_y)
        
class SmoothLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.laplacian = LaplacianLayer()

    def forward(self, imgs, masks=None):
        if masks is not None:
            return (masks * self.laplacian(imgs)).mean()
        else:
            return self.laplacian(imgs).mean()

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

        x = input.view(input_size[0]*input_size[1], 1, input_size[2], input_size[3])
        x_nom = F.conv2d(
            input=x,
            weight=V(self.w_nom),
            stride=1,
            padding=0
        )
        if do_normalize:
            x_den = F.conv2d(
                input=x,
                weight=V(self.w_den),
                stride=1,
                padding=0
            )
                      
            x = (x_nom.abs()/x_den)
        else:
            x = x_nom.abs()
            
        return x.view(input_size[0], input_size[1], input_size[2]-2, input_size[3]-2)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptor = get_base(f, 2)
        
    def forward(self, x1, x2):
        return torch.mse(self.perceptor(x1)-self.perceptor(x2))
    
class Loss(nn.Module):
    def __init__(self, scale=10, Tscale=2, ndown=2):
        super().__init__()
        self.appr = TriAppearanceLoss(scale=Tscale) #, ndown=ndown)
        self.smooth = EdgeAwareLoss()
#         self.smooth = SmoothLoss()
        self.scale = float(scale)
    def forward(self, d1s, d2s ,d3s, trans, rots, x1s, x2s, x3s, cameras):
        appr_loss, details = self.appr(d2s, trans, rots, x1s, x2s, x3s, cameras)
        d2s = [F.upsample(input=d2, scale_factor=2**i, mode='bilinear') if i>0 else d2 for d2 in d2s ]
        smooth_losses = [0.5**i * self.smooth(x2, d2) for x2, d2 in zip(x2s, d2s)]
#         smooth_losses = [ 0.5**(i) * self.smooth(d2) for i, d2 in enumerate(d2s) ]
        smooth_loss = torch.mean(torch.cat(smooth_losses, dim=0)) * self.scale
        #print(type(appr_loss))
        #print(type(smooth_loss))
        return appr_loss + smooth_loss, (appr_loss, smooth_loss, *details) 
               