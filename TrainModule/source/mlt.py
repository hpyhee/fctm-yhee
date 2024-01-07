from .ipca import IPCA
import torch
import numpy as np
import os
path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

QINFO = { ##  trained with tvd dataset and common basis
    'TVD-01' : {
        'L' : [-56.81757354736328, 134.69051361084],
        },
    'TVD-02' : {
        'L' : [-58.38869857788086, 151.741985321045],
        },
    'TVD-03' : {
        'L' : [-61.231201171875, 145.275161743164],
        }
}
class MLT:
    def __init__(self) -> None:
        # self.ipca = torch.load(os.path.join(path, "new2_mlt7168x7168.pt"))
        # self.ipca = torch.load(os.path.join(path, "trained_oi3000_DN53/new_mlt1792x7168.pt"))
        # self.ipca = torch.load(os.path.join(path, "trained_oi3000_DN53/new_mlt7168x7168.pt"))
        # self.ipca = torch.load(os.path.join(path, "trained_hieve_data_DN53/new_mlt7168x7168_hieve_DN53.pt"))

        self.d2 = torch.nn.PixelUnshuffle(2)
        self.d4 = torch.nn.PixelUnshuffle(4)
        self.u2 = torch.nn.PixelShuffle(2)
        self.u4 = torch.nn.PixelShuffle(4)
        self.n_components = int(7168 * 0.125)
        self.w  = [1.0, 2.0, 4.0]
    def transform(self, x):
        x = [ d * w for d, w in zip(x,self.w)]
        X = torch.concat([self.d4(x[0]), self.d2(x[1]), x[2]], dim=1)
        A = X.reshape(( X.shape[1], X.shape[2]*X.shape[3]))
        A = A.transpose(0, 1)
        B = self.ipca.transform(A, self.n_components)
        return B, X.shape[2:]
    
    def invtransform(self, x, shape):
        x = x.float()
        C = self.ipca.invtransform(x, self.n_components)
        
        r0 = self.u4(C.T[         :4096     ].reshape((1, -1, shape[0], shape[1])))
        r1 = self.u2(C.T[4096     :4096+2048].reshape((1, -1, shape[0], shape[1])))
        r2 =         C.T[4096+2048:         ].reshape((1, -1, shape[0], shape[1]))
        x = [r0, r1, r2]
        x = [ d / w for d, w in zip(x,self.w)]
        return x
        
        
    def uni_Q(self, layer, test_seq, lev):
        
        layer = (layer - QINFO[test_seq][lev][0]) * 1023 / QINFO[test_seq][lev][1]
        return torch.clip(layer,0,1023)
    
    def uni_DQ(self, layer, test_seq, lev):
        return (layer / 1023 * QINFO[test_seq][lev][1]) + QINFO[test_seq][lev][0]
    
    def get_proportion_shape(self, number):
        center = int(np.sqrt(number))
        while number % center != 0:
            center += 1
        return [center, number // center]
    
    def packed_feature_image(self, feature, num_tiles_x, num_tiles_y):
        feature = feature.clone().detach().cpu().squeeze(0)
        
        tile_h               = feature.shape[1]
        tile_w               = feature.shape[2]
        img     = torch.zeros((1, 1, int(tile_h*num_tiles_y), int(tile_w*num_tiles_x)))

        for tile_y_idx in range(num_tiles_y):
            for tile_x_idx in range(num_tiles_x):
                tile_index = tile_y_idx * num_tiles_x + tile_x_idx
                tile_x = tile_x_idx * tile_w
                tile_y = tile_y_idx * tile_h
                img[0,0,tile_y:tile_y+tile_h,tile_x:tile_x+tile_w] = feature[tile_index,:,:]
        return img
    def WriteImage(self, X, fp):
        X0 = X.numpy().astype(np.uint16)
        fp.writelines(X0)
        
    def Unpacking(self, img, num_tile_x, num_tile_y):
        tile_h    = img.shape[2]//num_tile_y
        tile_w    = img.shape[3]//num_tile_x

        feature = torch.zeros((1, 512, tile_h, tile_w), device="cuda")

        for tile_y_idx in range(num_tile_y):
            for tile_x_idx in range(num_tile_x):
                tile_index = tile_y_idx * num_tile_x + tile_x_idx
                
                tile_x = tile_x_idx * tile_w
                tile_y = tile_y_idx * tile_h
                
                feature[0, tile_index, :, :] = img[0, 0, tile_y:tile_y+tile_h, tile_x:tile_x+tile_w] 
        return feature
    
    def ReadImage(self, fp):
        raw = fp.read(544*608*2)
        yuv = np.frombuffer(raw, dtype=np.uint16)
        yuv = yuv.reshape((1,1,608, 544))
        return torch.from_numpy(yuv.astype(np.float64)).to(device="cuda")