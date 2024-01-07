import os, time
path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DEVICE = "cpu"

import torch
# torch.set_num_threads(1)

import numpy as np
import platform
import glob

from .ipca import IPCA
from source.det2 import *
from source.jde import *
from source.rom import *


class Dummy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

class MLT:
    def __init__(self, args="", mode="eval", device="cuda" ) -> None:
        return

    def load_model(self):
        # Detection & Segmentation
        if self.args.task  in ["det", "seg"]:
            
            model_cfg_name = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" if self.args.task == "det" else  \
                            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(model_cfg_name))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg_name)
            cfg.MODEL.DEVICE = DEVICE
            
            NNtask = DET2(cfg, self.args.task, self.args.eval_format)
            
        # Trakcing
        else:
            NNtask = JDE(frame_rate=30, split_points=self.sp, start_point=self.st)
            cfg = None
        return NNtask, cfg
    
    def get_image_loader(self, NNtask, path=None, img_list=None):
        # Detection & Segmentation
        dataloader = det2_dataloader(NNtask, self.args.inputdir if path == None else path, img_list=img_list)

        return dataloader
    
    def get_bitstream_loader(self):
        bin_list = glob.glob(f"{self.args.inputdir}/*.q{self.args.qp}.vvc")
        bin_list = [os.path.basename(bin.split(".q")[0]) for bin in bin_list]
        return bin_list
    
    def get_packed_ori_img_loader(self):
        if os.path.isdir(self.args.inputdir):
            packed_ori_img_list = glob.glob(f"{self.args.inputdir}/*.yuv")
        else:
            packed_ori_img_list = [self.args.inputdir]

        packed_ori_img_list = [item for item in packed_ori_img_list if not ".q" in item ]

        img_list = [os.path.basename(img).split(".")[0] for img in packed_ori_img_list]
        return img_list
    
    def get_packed_rec_img_loader(self):

        packed_rec_img_list = glob.glob(f"{self.args.inputdir}/*.q{self.args.qp}.yuv")

        img_list = [os.path.basename(img).split(".")[0] for img in packed_rec_img_list]
        return img_list
    
    def fit(self, x, need_sort):
        with torch.no_grad():
            if need_sort:
                x = sorted(x, key=lambda x: x.shape[-1], reverse=True)

            # feature scaling
            x = [ d * w for d, w in zip(x,self.w)]

            # feature reshaping & merge
            x = torch.cat([ down(f) for f, down in zip(x, self.down)], dim=1)

            x = x.reshape(( x.shape[1], x.shape[2]*x.shape[3]))
            x = x.transpose(0, 1)

            # Incremental PCA fitting
            self.ipca.fit(x)

    def save_basis(self):
        torch.save(self.ipca, self.trained_path)

    # def set_output_file_name(self, path):
    #     if self.args.task_unit == "vid" and self.bin_path != None:
    #         return
        
    #     fname = os.path.basename(path) if self.args.task_unit == "img" else os.path.dirname(self.args.inputdir)
    #     basename = fname.split(".")[0]
    #     self.yuv_path    = os.path.join(self.args.savedir, basename) + ".yuv" 
    #     self.header_path = os.path.join(self.args.savedir, basename) + ".hrd" 

    #     if self.args.qp > 0:
    #         self.bin_path    = os.path.join(self.args.savedir, basename) + f".q{self.args.qp}.vvc"
    #         self.rec_path    = os.path.join(self.args.savedir, basename) + f".q{self.args.qp}.yuv"
    #         self.res_path    = os.path.join(self.args.savedir, basename) + f".q{self.args.qp}.res"
    
    def set_vcm_enc_path(self, path, idx):
        if self.args.task_unit == "vid":
            fname = os.path.basename(os.path.dirname(path))
        else:
            fname = os.path.basename(path)
        
        if fname == "img1":
            fname = os.path.basename(os.path.dirname(os.path.dirname(path)))

        basename = fname.split(".")[0]

        self.yuv_path    = os.path.join(self.args.savedir, basename) + ".yuv" 
        self.header_path = os.path.join(self.args.savedir, basename) + ".hrd" 

        if self.args.task_unit == "img" or \
           self.args.task_unit == "vid" and idx == 0:
            if os.path.isfile(self.yuv_path):
                os.remove(self.yuv_path)
            if os.path.isfile(self.header_path):
                os.remove(self.header_path)

    def encode(self, x, img_size):
        start_proc_time,start_perf_time = time.process_time(), time.perf_counter()
        
        self.img_size = img_size
        
        if self.need_sort:
            x = sorted(x, key=lambda x: x.shape[-1], reverse=True)
        
        self.tile_size = x[-1].shape[2:]

        # feature scaling
        # x = [ d * w for d, w in zip(x,self.w)]
        def get_ecs_params(f):
            mean_v = f.mean()
            max_v  = f.max()
            min_v  = f.min()
            return max_v-mean_v #max(max_v-mean_v, mean_v-min_v)
        
        ecs_params = [ get_ecs_params(f) for f in x]

        # feature reshaping & merge
        x = torch.cat([ down(f) for f, down in zip(x, self.down)], dim=1)
        x = x.reshape(( x.shape[1], x.shape[2]*x.shape[3]))
        x = x.transpose(0, 1)

        # PCA forward transform 
        f_low = self.ipca.transform(x)

        # quantization
        if self.args.task_unit == "img":
            scale  = (self.bitdepth_maxval / (f_low.max()-f_low.min()))
            offset = f_low.min()
        else:
            scale  = self.scale 
            offset = self.offset

        if self.min > f_low.min():
            self.min = f_low.min()
        if self.max < f_low.max():
            self.max = f_low.max()

        f_q  = (f_low - offset) * scale
        f_q  = torch.round(f_q)
        f_q  = torch.clip(f_q, 0.0, self.bitdepth_maxval)

        # packing
        self.packed_size = (self.tile_size[0] * self.num_tiles[0], self.tile_size[1] * self.num_tiles[1])
        f_p        = torch.nn.functional.fold(f_q.unsqueeze(0), self.packed_size, self.tile_size, stride=self.tile_size)
        packed_img = f_p.cpu().numpy().astype(np.uint16)
        
        if self.args.task_unit == "img" or \
           self.args.task_unit == "vid" and not os.path.isfile(self.header_path):
            with open(self.header_path, "wb") as fp:
                fp.writelines(torch.tensor(self.tile_size).numpy().astype(np.uint16)) # 4 byte
                fp.writelines(torch.tensor(self.img_size ).numpy().astype(np.uint16)) # 4 byte
                fp.writelines(torch.tensor(self.num_tiles).numpy().astype(np.uint16)) # 4 byte
                fp.write( scale.cpu().numpy().astype(np.float32))                     # 4 byte
                fp.write(offset.cpu().numpy().astype(np.float32))                     # 4 byte
        
        with open(self.header_path, "ab") as fp:
            fp.writelines(torch.tensor(ecs_params).numpy().astype(np.float32)) # 4 byte x layers
    
        self.enc_proc_time += time.process_time()-start_proc_time
        self.enc_perf_time += time.perf_counter()-start_perf_time
        
        return packed_img
    
    def set_vtm_enc_path(self, path):
        fname = os.path.basename(path) 
        basename = fname.split(".")[0]

        if os.path.isdir(self.args.inputdir):
            self.header_path = os.path.join(self.args.inputdir, basename) + ".hrd" 
            self.yuv_path    = os.path.join(self.args.inputdir, basename) + ".yuv" 
        else:
            self.header_path = os.path.join(os.path.dirname(self.args.inputdir), f"{basename}.hrd")
            self.yuv_path    = self.args.inputdir

        if self.args.task_unit == "vid":
            self.bin_path    = os.path.join(self.args.savedir , basename) + f".q{self.args.qp}.vvc"
            self.bin_path    = os.path.join(self.args.savedir , basename) + f"/{basename}.q{self.args.qp}.vvc"
        else:
            self.bin_path    = os.path.join(self.args.savedir , basename) + f".q{self.args.qp}.vvc"
        self.log_path    = f"{self.bin_path}.enc"

        return basename 
    
    def enc_vtm(self, executor, idx, path):
        basename = self.set_vtm_enc_path(path)

        tile_h , tile_w          = np.fromfile(self.header_path, count=2, offset= 0, dtype='uint16').astype(np.int32)
        num_tiles_y, num_tiles_x = np.fromfile(self.header_path, count=2, offset= 8, dtype='uint16').astype(np.int32)
        packed_size = [tile_h * num_tiles_y, tile_w * num_tiles_x ]
        
        if os.path.isfile(self.log_path):
            with open(self.log_path, "rt") as fp:
                lines = fp.readlines()
                lines = [l for l in lines if 'Total Time:' in l]
            if len(lines) > 0:
                print(f"[{idx}]-{self.log_path}")
                return
            
        if self.args.task_unit == "vid":
            IP = GLOBAL_PARAMS[basename][2]
            FR = GLOBAL_PARAMS[basename][3]
            F  = GLOBAL_PARAMS[basename][4]

            name = self.bin_path.split(".vvc")[0]
            for id, FS in enumerate(range(0, F-1, IP)):
                f = IP + 1 if FS + IP + 1 < F else F - FS
                self.bin_path = f"{name}.{id}.vvc"
                self.log_path = f"{self.bin_path}.enc"
                cmd = f'{self.vtm_enc} -c {self.vtm_cfg} -q {self.args.qp} -i {self.yuv_path} -b {self.bin_path} -o "" --IntraPeriod={IP} --FrameRate={FR} --FrameSkip={FS} --FramesToBeEncoded={f} --SourceWidth={packed_size[1]} --SourceHeight={packed_size[0]} --InputBitDepth=10 --Level=2.1 --PrintHexPSNR -v 6 -dph 1 --InputChromaFormat=400 --ConformanceMode=1 > {self.log_path} && echo [{idx}, {id}]-{self.log_path}'
                print(cmd)
                executor.submit(os.system, cmd)
        else:
            cmd = f'{self.vtm_enc} -c {self.vtm_cfg} -q {self.args.qp} -i {self.yuv_path} -b {self.bin_path} -o "" --IntraPeriod={self.args.ip} --FrameRate={self.args.fr} --FramesToBeEncoded={self.args.f} --SourceWidth={packed_size[1]} --SourceHeight={packed_size[0]} --InputBitDepth=10 --Level=2.1 --PrintHexPSNR -v 6 -dph 1 --InputChromaFormat=400 --ConformanceMode=1 > {self.log_path} && echo [{idx}]-{self.log_path}'
            print(cmd)
            executor.submit(os.system, cmd)
    
    def make_vtm_cmd(self, idx, path):
        basename = self.set_vtm_enc_path(path)

        tile_h , tile_w          = np.fromfile(self.header_path, count=2, offset= 0, dtype='uint16').astype(np.int32)
        num_tiles_y, num_tiles_x = np.fromfile(self.header_path, count=2, offset= 8, dtype='uint16').astype(np.int32)
        packed_size = [tile_h * num_tiles_y, tile_w * num_tiles_x ]
        
            
        if self.args.task_unit == "vid":
            IP = GLOBAL_PARAMS[basename][2]
            FR = GLOBAL_PARAMS[basename][3]
            F  = GLOBAL_PARAMS[basename][4]

            name = self.bin_path.split(".vvc")[0]
            for id, FS in enumerate(range(0, F-1, IP)):
                f = IP + 1 if FS + IP + 1 < F else F - FS
                self.bin_path = f"{name}.{id}.vvc"
                self.log_path = f"{self.bin_path}.enc"

                if os.path.isfile(self.log_path):
                    with open(self.log_path, "rt") as fp:
                        lines = fp.readlines()
                        lines = [l for l in lines if 'Total Time:' in l]
                        
                    if len(lines) > 0:
                        print(f"[DONE] << [{idx}]-{self.log_path}")
                        continue

                cmd = f'{self.vtm_enc} -c {self.vtm_cfg} -q {self.args.qp} -i {self.yuv_path} -b {self.bin_path} -o "" --IntraPeriod={IP} --FrameRate={FR} --FrameSkip={FS} --FramesToBeEncoded={f} --SourceWidth={packed_size[1]} --SourceHeight={packed_size[0]} --InputBitDepth=10 --Level=2.1 --PrintHexPSNR -v 6 -dph 1 --InputChromaFormat=400 --ConformanceMode=1 > {self.log_path}'
                print(cmd)
                with open("../scripts/vtm_enc_script.sh", "at") as fp:
                    fp.write(cmd+"\n")
        else:
            if os.path.isfile(self.log_path):
                with open(self.log_path, "rt") as fp:
                    lines = fp.readlines()
                    lines = [l for l in lines if 'Total Time:' in l]
                if len(lines) > 0:
                    print(f"[DONE] << [{idx}]-{self.log_path}")
                    return

            cmd = f'{self.vtm_enc} -c {self.vtm_cfg} -q {self.args.qp} -i {self.yuv_path} -b {self.bin_path} -o "" --IntraPeriod={self.args.ip} --FrameRate={self.args.fr} --FramesToBeEncoded={self.args.f} --SourceWidth={packed_size[1]} --SourceHeight={packed_size[0]} --InputBitDepth=10 --Level=2.1 --PrintHexPSNR -v 6 -dph 1 --InputChromaFormat=400 --ConformanceMode=1 > {self.log_path}'
            print(cmd)
            with open("../scripts/vtm_enc_script.sh", "at") as fp:
                fp.write(cmd+"\n")

    def parcat(self, path):
        if self.args.task_unit != "vid":
             return
         
        basename = self.set_vtm_enc_path(path)
        IP = GLOBAL_PARAMS[basename][2]
        FR = GLOBAL_PARAMS[basename][3]
        F  = GLOBAL_PARAMS[basename][4]

        name = self.bin_path.split(".vvc")[0]
        cmd = f'{self.vtm_parcat}'
        for id, FS in enumerate(range(0, F-1, IP)):
            cmd += f" {name}.{id}.vvc"
        cmd += f" {self.bin_path}"
        os.system(cmd)
    
    def mux_bitstream(self, path):
        start_proc_time = time.process_time()
        start_perf_time = time.perf_counter()
        
        self.set_vtm_enc_path(path)

        hrd_size = np.array(os.path.getsize(self.header_path), dtype=np.uint16).tobytes()
        with open(self.header_path, "rb") as fp:
            hrd = fp.read()

        vvc_size = np.array(os.path.getsize(self.bin_path), dtype=np.uint16).tobytes()
        with open(self.bin_path, "rb") as fp:
            vvc = fp.read()
        
        fcvcm = hrd_size + hrd + vvc_size + vvc
        
        fcvcm_bin = f"{self.bin_path.split('.vvc')[0]}.fcvcm"
        with open(fcvcm_bin, "wb") as fp:
            fp.write(fcvcm)
            
        self.enc_proc_time += time.process_time()-start_proc_time
        self.enc_perf_time += time.perf_counter()-start_perf_time

    def get_vtm_enc_time(self, path):
        basename = self.set_vtm_enc_path(path)
        if self.args.task_unit == "vid":
            basepath = self.bin_path.split(f".q{self.args.qp}")[0]
            log_list = glob.glob(f"{basepath}.q{self.args.qp}.*.vvc.enc")
            
            assert(len(log_list)>0)
            proc, perf = 0,0
            for log in log_list:
                with open(log, "rt") as fp:
                    lines = fp.readlines()
                    runtime_lines = [l for l in lines if 'Total Time:' in l]
                    proc += float(runtime_lines[0].split()[2])
                    perf += float(runtime_lines[0].split()[5])
        else:
            with open(self.log_path, "rt") as fp:
                lines = fp.readlines()
                runtime_lines = [l for l in lines if 'Total Time:' in l]
                proc = float(runtime_lines[0].split()[2])
                perf = float(runtime_lines[0].split()[5])
        return proc, perf

    def set_vtm_dec_path(self, path):
        fname = os.path.basename(path) 
        basename = fname.split(".")[0]

        self.bin_path    = f"{os.path.join(self.args.inputdir, basename)}.q{self.args.qp}.vvc"
        self.rec_path    = f"{os.path.join(self.args.savedir , basename)}.q{self.args.qp}.yuv"
        self.log_path    = f"{self.rec_path}.dec"
            
    def dec_vtm_par(self, executor, idx, path):
        self.set_vtm_dec_path(path)
        cmd = f'{self.vtm_dec} -b {self.bin_path} -o {self.rec_path}>{self.log_path} && echo [{idx}]-{self.log_path}'
        executor.submit(os.system, cmd)
    
    def dec_vtm(self, idx):
        cmd = f'{self.vtm_dec} -b {self.bin_path} -o {self.rec_path}>{self.log_path} && echo [{idx}]-{self.log_path}'
        os.system(cmd)
    

    def get_vtm_dec_time(self, path):
        self.set_vtm_dec_path(path)
        with open(self.log_path, "rt") as fp:
            lines = fp.readlines()
            runtime_lines = [l for l in lines if 'Total Time:' in l]
            proc = float(runtime_lines[0].split()[2])

        self.vtm_dec_proc_time += proc
        return proc
    
    def set_vcm_dec_path(self, path):
        fname = os.path.basename(path) 
        basename = fname.split(".")[0]
          
        self.bin_path    = f"{os.path.join(self.args.inputdir, basename)}.q{self.args.qp}.vvc"
        self.header_path = f"{os.path.join(self.args.savedir , basename)}.hrd" 
        self.rec_path    = f"{os.path.join(self.args.savedir , basename)}.q{self.args.qp}.yuv" 
        self.res_path    = f"{os.path.join(self.args.savedir , basename)}.q{self.args.qp}"
        self.res_path   += ".coco" if self.args.eval_format == "voc" else ".json"
        self.log_path    = f"{self.rec_path}.dec"

        return basename

    def decode_header(self):
        start_proc_time,start_perf_time = time.process_time(), time.perf_counter()

        tile_h , tile_w, img_h, img_w = np.fromfile(self.header_path, count=4, offset= 0, dtype='uint16').astype(np.int32)
        num_tiles_y, num_tiles_x      = np.fromfile(self.header_path, count=2, offset= 8, dtype='uint16').astype(np.int32)
        self.scale, self.offset       = np.fromfile(self.header_path, count=2, offset=12, dtype='float32')
        
        self.max_val = (self.scale * self.bitdepth_maxval) + self.offset
        self.min_val = self.offset

        # print(f"[MLT-HRD] TILE_SIZE: ({tile_h}, {tile_w})")
        # print(f"[MLT-HRD] OIMG_SIZE: ({img_h} , {img_w}) ")
        # print(f"[MLT-HRD] NUM_TILES: ({num_tiles_y}, {num_tiles_x})")
        # print(f"[MLT-HRD] SCALE    : {self.scale} ")
        # print(f"[MLT-HRD] OFFSET   : {self.offset}")

        self.img_size     = (img_h, img_w)
        self.packed_size  = torch.tensor([tile_h * num_tiles_y, tile_w * num_tiles_x])
        self.tile_size    = (tile_h, tile_w)
        
        self.dec_proc_time += time.process_time()-start_proc_time
        self.dec_perf_time += time.perf_counter()-start_perf_time

    def read_frame(self, fp):
        return np.fromfile(fp, dtype='uint16', count=self.packed_size[0]*self.packed_size[1])
    
    def write_frame(self, packed_img):
        # write yuv file
        with open(self.yuv_path, 'ab' if self.args.task_unit=="vid" else "wb" ) as fp:
            fp.writelines(packed_img)

    def decode(self, f_packed):
        start_proc_time,start_perf_time = time.process_time(), time.perf_counter()

        f_p_rec = torch.from_numpy(np.reshape(f_packed, (1, 1, self.packed_size[0], self.packed_size[1])).astype(np.float32)).to(device=self.device)
        
        # unpacking
        f_up  = torch.nn.functional.unfold(f_p_rec, self.tile_size, stride=self.tile_size)

        # #inverse quantization
        f_dq = (f_up  / self.scale  ) + self.offset  
        f_dq = torch.clip(f_dq, self.min_val, self.max_val)

        # inverse transform
        f_rec = self.ipca.invtransform(f_dq)

        x = []
        ch_st = 0 
        for i in range(self.num_layers):
            ch_ed = ch_st + int(self.num_chs[i] * (2**((self.num_layers - 1 - i )*2)))
            r = self.up[i](f_rec.T[ch_st :ch_ed].reshape((1, -1, self.tile_size[0], self.tile_size[1])))
            ch_st = ch_ed
            x.append(r)
      
        x = [ d / w for d, w in zip(x,self.w)]
        
        if self.need_sort:
            x = sorted(x, key=lambda x: x.shape[-1], reverse=False)
        
        self.dec_proc_time += time.process_time()-start_proc_time
        self.dec_perf_time += time.perf_counter()-start_perf_time
        
        return x, self.img_size 
    