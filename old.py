
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
    
class DepthFuseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.ConvTranspose2d(1, 1, 2, 2)
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.cont = Conv(1, 1, 3, 1, 1, None)
        self.fuse = Conv(2, 1, 3, 1, 1, nn.Sigmoid())
    def forward(self, up_d, d):    
        up_d = self.cont(self.up(up_d))
        return self.fuse(torch.cat((up_d, d), dim=1))
    