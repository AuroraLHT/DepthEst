import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fastai.column_data import *
from fastai.conv_learner import *
from fastai.dataset import *

class UnsupFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        super().__init__(fnames, transform, path)
    def get_y(self, i): return None
    def get_c(self): return 0
    
class UnShuffleImageData(ImageData):
    def __init__(self, path, datasets, bs, num_workers, classes):
        trn_ds,val_ds,fix_ds,aug_ds,test_ds,test_aug_ds = datasets
        self.path,self.bs,self.num_workers,self.classes = path,bs,num_workers,classes
        self.trn_dl,self.val_dl,self.fix_dl,self.aug_dl,self.test_dl,self.test_aug_dl = [
            self.get_dl(ds,shuf) for ds,shuf in [
                (trn_ds,False),(val_ds,False),(fix_ds,False),(aug_ds,False),
                (test_ds,False),(test_aug_ds,False)
            ]
        ]

class UnScaleTransforms():
    def __init__(self, sz, tfms, normalizer, denorm,
                 tfm_y=TfmType.NO, sz_y=None):
        if sz_y is None: sz_y = sz
        self.sz,self.denorm,self.norm,self.sz_y = sz,denorm,normalizer,sz_y
        self.tfms = tfms
        if normalizer is not None: self.tfms.append(normalizer)
        self.tfms.append(ChannelOrder(tfm_y))

    def __call__(self, im, y=None): return compose(im, y, self.tfms)
    def __repr__(self): return str(self.tfms)
    
# just a little modification which remove the multipler at the batch size of the val DL
class UnDBSColumnarModelData(ModelData):
    def __init__(self, path, trn_ds, val_ds, bs, test_ds=None, shuffle=True):
        test_dl = DataLoader(test_ds, bs, shuffle=False, num_workers=1) if test_ds is not None else None
        super().__init__(path, DataLoader(trn_ds, bs, shuffle=shuffle, num_workers=1),
            DataLoader(val_ds, bs, shuffle=False, num_workers=1), test_dl)

    @classmethod
    def from_arrays(cls, path, val_idxs, xs, y, is_reg=True, is_multi=False, bs=64, test_xs=None, shuffle=True):
        ((val_xs, trn_xs), (val_y, trn_y)) = split_by_idx(val_idxs, xs, y)
        test_ds = PassthruDataset(*(test_xs.T), [0] * len(test_xs), is_reg=is_reg, is_multi=is_multi) if test_xs is not None else None
        return cls(path, PassthruDataset(*(trn_xs.T), trn_y, is_reg=is_reg, is_multi=is_multi),
                   PassthruDataset(*(val_xs.T), val_y, is_reg=is_reg, is_multi=is_multi),
                   bs=bs, shuffle=shuffle, test_ds=test_ds)

    @classmethod
    def from_data_frames(cls, path, trn_df, val_df, trn_y, val_y, cat_flds, bs=64, is_reg=True, is_multi=False, test_df=None, shuffle=True):
        trn_ds  = ColumnarDataset.from_data_frame(trn_df,  cat_flds, trn_y, is_reg, is_multi)
        val_ds  = ColumnarDataset.from_data_frame(val_df,  cat_flds, val_y, is_reg, is_multi)
        test_ds = ColumnarDataset.from_data_frame(test_df, cat_flds, None,  is_reg, is_multi) if test_df is not None else None
        return cls(path, trn_ds, val_ds, bs, test_ds=test_ds, shuffle=shuffle)

    @classmethod
    def from_data_frame(cls, path, val_idxs, df, y, cat_flds, bs=64, is_reg=True, is_multi=False, test_df=None, shuffle=True):
        ((val_df, trn_df), (val_y, trn_y)) = split_by_idx(val_idxs, df, y)
        return cls.from_data_frames(path, trn_df, val_df, trn_y, val_y, cat_flds, bs, is_reg, is_multi, test_df=test_df, shuffle=shuffle)

    def get_learner(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops,
                    y_range=None, use_bn=False, **kwargs):
        model = MixedInputModel(emb_szs, n_cont, emb_drop, out_sz, szs, drops, y_range, use_bn, self.is_reg, self.is_multi)
        return StructuredLearner(self, StructuredModel(to_gpu(model)), opt_fn=optim.Adam, **kwargs)

def get_MD(trn, val, tfms, bs, datapath, path):
    #tfms = (trn_tfms, val_tfms)
    
    datasets = ImageData.get_ds(
        UnsupFilesDataset,
        (trn, None),
        (val, None),
        tfms, path=datapath)
    
    md = UnShuffleImageData(
        path,
        datasets,
        bs, num_workers=16,
        classes=None)
    
    return md

def get_cam(trn_cam, val_cam, bs):
    return UnDBSColumnarModelData.from_data_frames(
        path= None,
        trn_df=trn_cam, val_df=val_cam,
        trn_y=None, val_y=None,
        cat_flds=[],
        bs=bs, shuffle=False)


def denormer(mean, std):
    def denorm(imgs, **kargs):
        return V(imgs.cpu().data * std + mean, **kargs)
    return denorm

"""
def toMD(train, val, bs, stats):
    trn0, trn1, trn2, trn_camera = train.t0, train.t1,train.t2, train[['fx', 'fy', 'cx', 'cy']]
    val0, val1, val2, val_camera  = val.t0, val.t1, val.t2, val[['fx', 'fy', 'cx', 'cy']]
    
    stats = imagenet_stats
    tfm_norm = Normalize(*stats, tfm_y=TfmType.NO) 
    tfm_denorm = Denormalize(*stats)
    
    trn_tfms = UnScaleTransforms(sz=None, tfms=[], normalizer=tfm_norm, denorm=tfm_denorm, tfm_y=TfmType.NO, sz_y=sz_y)
    val_tfms = UnScaleTransforms(sz=None, tfms=[], normalizer=tfm_norm, denorm=tfm_denorm, tfm_y=TfmType.NO, sz_y=sz_y)
    
    MD1 = get_MD(trn0, val0)
    MD2 = get_MD(trn1, val1)
    MD3 = get_MD(trn2, val2)
    MDcam = get_cam(trn_camera, val_camera)
    
    return MD1, MD2, MD3, MDcam
"""
def shuffle(dataframe):
    for i in range(4):
        dataframe = dataframe.sample(frac=1)
    return dataframe.reset_index(drop=True)

def plot_img(recon, index, figsize=(12,4)):
    # input image: R [0,1]
    recon = recon[index].cpu().permute(1,2,0).data.numpy()
    print(recon.min(), recon.max())
    plt.figure(figsize=figsize)
    plt.imshow(recon)
    plt.axis('off')    

def plot_depth(depths, index=0, figsize=(12,4), scale=50, inv=False, cb=True, cmap='plasma'):
    inv_depth = depths.cpu().data[index].numpy()
    inv_depth = np.clip(inv_depth, a_min=0.01, a_max=None)
    if inv:
        depth = 1/inv_depth[0]
    else:
        depth = inv_depth[0]
        
    m, std, mx = depth.mean(axis=(0, 1)), depth.std(axis=(0, 1)), depth.max(axis=(0, 1))
    plt.figure(figsize=figsize)
    plt.imshow(
        depth*scale,
        cmap=cmap,
    )
    if cb: plt.colorbar()
    plt.axis('off')
    
def plot_mask(masks, index=0, figsize=(12,4)):
    plt.style.use('grayscale')
    mask = masks[index]
    plt.figure(figsize=figsize)
    plt.imshow(mask.cpu().data.numpy(),cmap="gray", vmin=0, vmax=1)
    plt.colorbar()
    plt.axis('off')

def plot_losses(losses):
    plt.plot(np.arange(len(losses)), np.array(losses))    
    
def tonp(tensor):
    return tensor.cpu().data.numpy()
    
def save_res(img, recon, depth, path, figsize=(12,12)):
    fig, axs = plt.subplots(3,1, figsize=figsize)
    for i in range(3): axs[i].set_axis_off()
    axs[0].imshow(depth, cmap="plasma") #,  vmin=1, vmax=10)
    axs[1].imshow(img)
    axs[2].imshow(recon)
    fig.savefig(path)
    plt.close(fig)
    
    
class LossCollect:
    def __init__(self, nslot):
        self.collections = [list() for i in range(nslot)]
        
    def collect(self, *args):
        for i, arg in enumerate(args):
            self.collections[i].append(arg)
            
    def show(self, n=5):
        return [np.mean(self.collections[i][-n:]) for i in range(len(self.collections))]