import torch
import torch.nn as nn
import torch.nn.functional as F

import fastai
from fastai.conv_learner import *

# part of architecture is copied from fastai library

# transfer learning from pretrain resnet34

EPS = 1e-10
f = resnet34
cut,lr_cut = model_meta[f]

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

def get_base():
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)

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

class DepthFuseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.cont = Conv(1, 1, 3, 1, 1, None)
        self.fuse = Conv(2, 1, 3, 1, 1, nn.Sigmoid())
    def forward(self, up_d, d):    
        up_d = self.cont(self.up(up_d))
        return self.fuse(torch.cat((up_d, d), dim=1))
    
class Pose(nn.Module):
    def __init__(self, inc, mag_scalor = 1):
        super().__init__()
        self.ps = 6
        self.multi = 2
        self.mag_scalor = mag_scalor

        self.body = nn.Sequential(
            nn.Conv2d(inc, 128, 3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.ps*self.multi, 3, bias=True),
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
        self.up2 = UnetBlock(256,128,128)
        self.up3 = UnetBlock(128,64,64)
        self.up4 = UnetBlock(64,64,64)
        self.up5 = UnetBlock(64,3,16) 
        self.d1 =  Conv( 256, 1, 3, 1, 1, activation_func=nn.Sigmoid() )
        self.d2 =  Conv( 128, 1, 3, 1, 1, activation_func=nn.Sigmoid() )
        self.d3 =  Conv( 64, 1, 3, 1, 1, activation_func=nn.Sigmoid() )
        self.d4 =  Conv( 64, 1, 3, 1, 1, activation_func=nn.Sigmoid() )
        self.d5 =  Conv( 16, 1, 3, 1, 1, activation_func=nn.Sigmoid() )   
        self.fuse2 = DepthFuseBlock()
        self.fuse3 = DepthFuseBlock()
        self.fuse4 = DepthFuseBlock()
        self.fuse5 = DepthFuseBlock()
        #self.op_norm = torch.nn.InstanceNorm2d(1)
    
        self.MIN_DISP = 0.01
        self.DISP_SCALING = 10

    def forward(self, x, enc_only=False):
        inp = x
        x = F.elu(self.rn(x))        
        depthmaps = []

        if not enc_only:
            x = self.up1(x, self.sfs[3].features)
            d1 = self.d1(x)
#             depthmaps.append(d1)
            
            x = self.up2(x, self.sfs[2].features)
            d2 = self.fuse2( d1, self.d2(x) )
#             depthmaps.append(d2)
            
            x = self.up3(x, self.sfs[1].features)
            d3 = self.fuse3( d2, self.d3(x) )
            depthmaps.append(d3)
            
            x = self.up4(x, self.sfs[0].features)
            d4 = self.fuse4( d3, self.d4(x) )
            depthmaps.append(d4)
            
            x = self.up5(x, inp)
            d5 = self.fuse5( d4, self.d5(x) )
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

class TriDepth(nn.Module):
    def __init__(self, rn, ochannel, train=True):
        super().__init__()
        self.depth = Depth34(rn, ochannel) 
        self.pose = Pose(256*3)
        self.train = train
        
    def forward(self, x1, x2, x3):
        if self.train:
            d1, ft1 = self.depth(x1, enc_only=True) # src
            d2, ft2 = self.depth(x2, enc_only=False) # target
            d3, ft3 = self.depth(x3, enc_only=True) # src
        else:
            d1, ft1 = self.depth(x1, enc_only=False) # src
            d2, ft2 = self.depth(x2, enc_only=False) # target
            d3, ft3 = self.depth(x3, enc_only=False) # src            
        trans, rotation = self.pose(torch.cat((ft1,ft2,ft3), dim=1))

        return d1, d2, d3, trans, rotation

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
        self.register_buffer('o', torch.zeros([1,1]).type(torch.FloatTensor))
        self.register_buffer('eye', torch.eye(3).type(torch.FloatTensor).unsqueeze(0))
        
    def factorize(self, vecs, dim):
        mags = vecs.norm(p=2, dim=dim, keepdim=True)
        return vecs/mags, mags

    def rot_vec2mat(self, rot_vecs):
        batch, _ = rot_vecs.size()
        directs, angles = self.factorize(rot_vecs, 1)
        
        K0 = directs[:,:1]
        K1 = directs[:,1:2]
        K2 = directs[:,2:]
        
        o = V(self.o.repeat(batch, 1))
        eye = V(self.eye.repeat(batch, 1, 1))
        
        #print(K0.type, K2.type, K1.type, o.type, eye.type)
        angles = angles.unsqueeze(-1)
        K = torch.cat((o, -K2, K1, K2, o, K0, -K1, K0, o), 1).view(-1, 3, 3) # form a cpro matrix
        return eye + K * angles.sin() + torch.bmm(K,K) * (1-angles.cos()) # using the R formular
    
    def forward(self, pose, inv_depth, camera):
        camera = camera.data
        """
            Params:
                pose: relative pose, N X 6 vectors,
                    1-3 is the transition vector
                    4-6 is the rotation vector
                inv_depth: invered depth map
                camera: intrinsic camera parameters NX4: (fx, fy, cx, cy)
            Return:
                tkx: transformed camera pixel points - x-component
                tky: transformed camera pixel points - y-component
                dmask: binary map of pixel that keeps track in the future
        """
        batch, c, h, w = inv_depth.size()

        cxy = camera[:, 2:].contiguous().view(batch, 1, 1, 2)
        fxy = camera[:, :2].contiguous().view(batch, 1, 1, 2)

        rot_vecs = pose[:,:3]
        tran_vecs = pose[:, 3:]

        rot_mats = self.rot_vec2mat(rot_vecs)

        # grip points preperation
        kx, ky = meshgrid_fromHW(h, w, dtype=type(inv_depth.data))
        #kx, ky = kx+0.5, ky+0.5
        kxy = torch.stack([kx, ky], dim=-1).repeat(batch, 1, 1, 1)
        
        hxy = (kxy - cxy)/fxy
        hxy = V(hxy)
        # transformation : Output Size NX3XHXW 
        # calculate the transformed extended homogenerous coordinate(in projective space) of the camera screen
        thxyz = torch.matmul(rot_mats[:, :, :2].unsqueeze(1).unsqueeze(2), hxy.unsqueeze(-1)).squeeze(-1).permute(0,3,1,2)
        thxyz = thxyz + rot_mats[:, :, 2].unsqueeze(-1).unsqueeze(-1) 
        thxyz = thxyz + tran_vecs.unsqueeze(-1).unsqueeze(-1) * inv_depth
        
        # project the pixel in "tilted" projective space to projective space       
        thxy_warp = thxyz[:, :2] / thxyz[:, 2:].clamp(min=EPS)
        
        # projective space to camera space
        cxy = V(cxy.view(batch, 2, 1, 1))
        fxy = V(fxy.view(batch, 2, 1, 1))
        
        #tkxy = (thxy_warp * fxy) + cxy - V(kxy.permute(0,3,1,2))
        tkxy = (thxy_warp * fxy) + cxy
        tkx, tky = tkxy[:, 0], tkxy[:, 1]

        dmask = V((thxyz[:, 2].data>EPS).type_as(inv_depth.data))

        return tkx, tky, dmask

class Offset2(nn.Module):
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
    
    def forward(self, pose, inv_depth, camera):
        camera = camera.data
        """
            Params:
                pose: relative pose, N X 6 vectors,
                    1-3 is the transition vector
                    4-6 is the rotation vector
                inv_depth: invered depth map
                camera: intrinsic camera parameters NX4: (fx, fy, cx, cy)
            Return:
                tkx: transformed camera pixel points - x-component
                tky: transformed camera pixel points - y-component
                dmask: binary map of pixel that keeps track in the future
        """
        b, c, h, w = inv_depth.size()

        cxy = camera[:, 2:].contiguous().view(b, 1, 1, 2)
        fxy = camera[:, :2].contiguous().view(b, 1, 1, 2)

        rot_vecs = pose[:,:3]
        tran_vecs = pose[:, 3:]

        rot_mats = self.rot_vec2mat(rot_vecs)

        # grip points preperation
        kx, ky = meshgrid_fromHW(h, w, dtype=type(inv_depth.data))
        #kx, ky = kx+0.5, ky+0.5
        kxy = torch.stack([kx, ky], dim=-1).repeat(b, 1, 1, 1)
        
        hxy = (kxy - cxy)/fxy
        
        # augment the homogeneous coordinate, add the 1 z dimension
        hxy = torch.cat((hxy, torch.ones(b, h, w, 1).type_as(hxy)), dim=-1)
        
        hxy = V(hxy)
        
        # stack these trans together w.r.t the column
        T = torch.cat((rot_mats, tran_vecs.unsqueeze(-1)), dim=-1)
        T = torch.cat((T, V(self.filler).repeat(b, 1, 1) ), dim=-2)

        # homogeneous space to real space
        rxy = hxy / inv_depth.permute(0,2,3,1)

        # augment the realspace xyz coordinate, add the 1 dimension for xyz transistion
        rxy = torch.cat((rxy, V(torch.ones(b, h, w, 1))), dim=-1)

        # transformation : Output Size NX3XHXW 
        # calculate the transformed extended homogenerous coordinate(in projective space) of the camera screen
        
        # add dummy dimension for broadcasting
        T = T.unsqueeze(1).unsqueeze(2)
        rxy = rxy.unsqueeze(-1)

        t_xyz = torch.matmul(T, rxy).squeeze(-1)
        t_xyz = t_xyz.permute(0,3,1,2)

        t_xy = t_xyz[:, :2] 
        t_z = t_xyz[:, 2:3].clamp(min=EPS)
        
        # projective space to camera space
        cxy = V(cxy.view(b, 2, 1, 1))
        fxy = V(fxy.view(b, 2, 1, 1))        
        t_kxy = ((t_xy * fxy) + cxy) / t_z
        
        # divided the result and return
        t_kx, t_ky = t_kxy[:, 0], t_kxy[:, 1]
        dmask = V((t_xyz[:, 2].data>EPS).type_as(inv_depth.data)) 
        return t_kx, t_ky, dmask

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
    return torch.sum(torch.abs(mask*(x1-x2)))/(torch.sum(mask)+1) 

def compute_img_stats(img):
    # img_pad = torch.nn.ReplicationPad2d(1)(img)
    img_pad = img
    mu = F.avg_pool2d(img_pad, kernel_size=3, stride=1, padding=0)
    sigma = F.avg_pool2d(img_pad**2, kernel_size=3, stride=1, padding=0) - mu**2
    return mu, sigma

def compute_SSIM(img0, img1 ):
    mu0, sigma0= compute_img_stats(img0) 
    mu1, sigma1= compute_img_stats(img1)
    # img0_img1_pad = torch.nn.ReplicationPad2d(1)(img0 * img1)
    img0_img1_pad = img0*img1
    sigma01 = F.avg_pool2d(img0_img1_pad, kernel_size=3, stride=1, padding=0) - mu0*mu1
    # C1 = .01 ** 2
    # C2 = .03 ** 2
    C1 = .001
    C2 = .009

    ssim_n = (2*mu0*mu1 + C1) * (2*sigma01 + C2)
    ssim_d = (mu0**2 + mu1**2 + C1) * (sigma0 + sigma1 + C2)
    ssim = ssim_n / ssim_d
    return ((1-ssim)*.5).clamp(0, 1)

def ssim_loss(img0, img1, mask):
    b, c, h, w = img0.size()
    SSIM = compute_SSIM(img0, img1)
    h = (mask.size(2) - SSIM.size(2))//2
    w = (mask.size(3) - SSIM.size(3))//2
    SSIM = F.pad(SSIM, (w,w,h,h), mode='constant', value=0)
    return torch.mean(
        torch.sum(
            torch.sum((SSIM*mask).view(b, c, -1), dim=-1)/(1+torch.sum(mask.view(b, 1, -1), dim=-1)), 
            dim = -1
        )
    )

class TriAppearanceLoss(nn.Module):
    def __init__(self, scale=0.5):
        super().__init__()
        self.offset = Offset3()
        self.sampler = BilinearProj()

        self.scale = scale        
        #self.imgds = DownSampleLayer(chan=3)
        #self.depthus = nn.Upsample(scale_factor=2, mode='bilinear')
        
    def forward(self, d2s, trans, rotation, x1, x2, x3, camera):
        l1losses = []
        ssimlosses = []
        for i, d2 in enumerate(d2s):            
            if i>0: d2 = F.upsample(input=d2, scale_factor=2**i, mode='bilinear')
            
            cx12, cy12, d_mask12 = self.offset.forward(trans[:, 0], rotation[:, 0], inv_depth = d2, camera = camera)
            #cx32, cy32, d_mask32 = self.offset.forward(trans[:, 1], rotation[:, 1], inv_depth = d2, camera = camera)

            x12, in_mask12 = self.sampler.forward(x1, cx12, cy12)
            #x32, in_mask32 = self.sampler.forward(x3, cx32, cy32)

            mask12 = (d_mask12*in_mask12).unsqueeze(1)
            #mask32 = (d_mask32*in_mask32).unsqueeze(1)
            
            mask12.requires_grad = False
            #mask32.requires_grad = False
            
            # loss on original scale
            l1losses.append(l1_loss(x12, x2, mask12))
            ssimlosses.append(ssim_loss(x12, x2, mask12))       
            
            #l1losses.append( l1_loss(x12, x2, mask12) + l1_loss(x32, x2, mask32) )
            #ssimlosses.append( ssim_loss(x12, x2, mask12) + ssim_loss(x32, x2, mask32) )       
        
        l1loss = torch.mean(torch.cat(l1losses, dim=0))
        ssimloss = torch.mean(torch.cat(ssimlosses, dim=0))

        return ssimloss + self.scale * l1loss, (ssimloss, self.scale * l1loss)
        #return ssimloss + self.scale * l1loss, (ssimloss, self.scale * l1loss)
        

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

class DownSampleLayer(nn.Module):
    def __init__(self, chan):
        super().__init__()
        K = torch.FloatTensor([[0.0751,   0.1238,    0.0751],
                              [0.1238,   0.2042,    0.1238],
                              [0.0751,   0.1238,    0.0751]]).view(1, 1, 3, 3)
        self.register_buffer('smooth_kernel', K)
        if chan>1:
            k = K
            K = torch.zeros(chan, chan, 3, 3)
            for i in range(chan):
                K[i, i, :, :] = k
        self.register_buffer('smooth_kernel_K', K)
        self.avg_pool_func = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.reflection_pad_func = torch.nn.ReflectionPad2d(1)

    def forward(self, input):
        output_dim = input.dim()
        output_size = input.size()
        if output_dim==2:
            K = self.smooth_kernel
            input = input.unsqueeze(0).unsqueeze(0)
        elif output_dim==3:
            K = self.smooth_kernel
            input = input.unsqueeze(1)
        else:
            K = self.smooth_kernel_K

        x = self.reflection_pad_func(input)

        x = F.conv2d(input=x,
                    weight=V(K),
                    stride=1,
                    padding=0)
        # remove here if not work out
        padding = [0, int(np.mod(input.size(-1), 2)), 0, int(np.mod(input.size(-2), 2))]
        x = torch.nn.ReplicationPad2d(padding)(x)
        # -----
        x = self.avg_pool_func(x)

        if output_dim==2:
            x =  x.squeeze(0).squeeze(0)
        elif output_dim==3:
            x =  x.squeeze(1)

        return x
