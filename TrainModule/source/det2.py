# %%
import cv2, os, sys, time
sys.path.append("../detectron2")

import itertools
import json
import torch
import  detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model, detector_postprocess
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

from torch.utils.data import Dataset
import numpy as np

class det2_dataloader(Dataset):
    def __init__(self, nntask, input_dir, img_list=None) -> None:
        super().__init__()
        if img_list == None:
            img_list = sorted(os.listdir(input_dir))
            
        img_list = [os.path.join(input_dir, img) for img in img_list if img.endswith(".jpg") or img.endswith(".png")]
        self.img_list = img_list
        self.len    = len(self.img_list)

        self.nntask = nntask
        self.count  = 0

    def __iter__(self):
        self.count = -1
        return self
    
    def __len__(self):
        return self.len
    
    def __next__(self):
        self.count += 1
        if self.count == self.len:
            raise StopIteration
        
        with torch.no_grad():
            img_path = self.img_list[self.count]
            original_image = cv2.imread(img_path)

            # # Apply pre-processing to image.
            if self.nntask.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.nntask.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            
            batched_inputs = [inputs]
            images = self.nntask.model.preprocess_image(batched_inputs)
            return img_path, images.tensor, [height, width] #[height, width] #batched_inputs  
    
    def __getitem__(self, index):
        with torch.no_grad():
            img_path = self.img_list[index]
            original_image = cv2.imread(img_path)

            # # Apply pre-processing to image.
            if self.nntask.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.nntask.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            images = self.nntask.model.preprocess_image([inputs])
        return img_path, images.tensor, [height, width]
    
def PART1(func):
    def wrapper(*args, **kwargs):
        start_proc_time = time.process_time()
        start_perf_time = time.perf_counter()

        res = func(*args, **kwargs)

        args[0].part1_proc_time += time.process_time()-start_proc_time
        args[0].part1_perf_time += time.perf_counter()-start_perf_time

        return res
    
    return wrapper

def PART2(func):
    def wrapper(*args, **kwargs):
        start_proc_time = time.process_time()
        start_perf_time = time.perf_counter()

        res = func(*args, **kwargs)

        args[0].part2_proc_time += time.process_time()-start_proc_time
        args[0].part2_perf_time += time.perf_counter()-start_perf_time

        return res
    
    return wrapper

class DET2:
    def __init__(self, cfg, task, eval_format="voc"):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.eval_format = eval_format

        self.model = build_model(self.cfg)
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.model.eval()

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.task = task
        
        if self.eval_format == "voc":
            with open(f"{os.path.dirname(__file__)}/../data/coco_classes.txt", 'r') as f:
                self.coco_classes = f.read().splitlines()  

        self.input_format = cfg.INPUT.FORMAT
        
        assert self.input_format in ["RGB", "BGR"], self.input_format
        
        self.part1_proc_time = 0
        self.part1_perf_time = 0

        self.part2_proc_time = 0
        self.part2_perf_time = 0

    @PART1    
    def part1(self, images):
        with torch.no_grad():  
            features = self.model.backbone(images)  
            return [features[level] for level in ["p2", "p3", "p4", "p5"] ]

    @PART2
    def part2(self, features, image_size):
        # image_size : tuple (int, int)
        with torch.no_grad():  
            dummy_oimg = np.zeros((image_size[0], image_size[1], 3))
            height, width = dummy_oimg.shape[:2]
            image = self.aug.get_transform(dummy_oimg).apply_image(dummy_oimg)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            batched_inputs = [inputs]
            images = self.model.preprocess_image(batched_inputs)

            features = { level:features[idx] for idx, level in enumerate(["p2", "p3", "p4", "p5"]) }
            if self.model.backbone.top_block is not None:
                features["p6"] = self.model.backbone.top_block(features["p5"])[0]
                    
            if self.model.proposal_generator is not None:
                proposals, _ = self.model.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.model.device) for x in batched_inputs]

            results, _ = self.model.roi_heads(images, features, proposals, None)
            pred = self.model._postprocess(results, batched_inputs, images.image_sizes)[0]
                
        return pred
    
    @PART2
    def init_results(self):
        if self.eval_format == "coco":
            return []
        elif self.eval_format == "voc":
            if self.task == "det":
                return ['ImageID,LabelName,Score,XMin,XMax,YMin,YMax']
            else:
                return ['ImageID,LabelName,Score,XMin,XMax,YMin,YMax,ImageWidth,ImageHeight,Mask']

    @PART2
    def summary_results(self, pred, poc):
        if self.eval_format == "coco":
            return self.summary_coco_results(pred, poc)
        elif self.eval_format == "voc":
            return self.summary_voc_results(pred, poc)

    @PART2
    def write_results(self, results, res_path):
        if self.eval_format == "coco":
            return self.write_coco_results(results, res_path)
        elif self.eval_format == "voc":
            return self.write_voc_results(results, res_path)

    def summary_voc_results(self, res, poc):
        with torch.no_grad():  
            classes = res['instances'].pred_classes.to('cpu').numpy()
            scores  = res['instances'].scores.to('cpu').numpy()
            bboxes  = res['instances'].pred_boxes.tensor.to('cpu').numpy()
            H, W    = res['instances'].image_size

            # convert bboxes to 0-1
            # detectron: x1, y1, x2, y2 in pixels
            bboxes = bboxes / [W, H, W, H]

            # OpenImage output x1, x2, y1, y2 in percentage
            bboxes = bboxes[:, [0, 2, 1, 3]]

            if self.task == 'seg':
                masks = res['instances'].pred_masks.to('cpu').numpy()

            results = []
            for ii in range(len(classes)):
                coco_cnt_id = classes[ii]
                class_name = self.coco_classes[coco_cnt_id]

                if self.task == 'seg':
                    assert (masks[ii].shape[1]==W) and (masks[ii].shape[0]==H), \
                    print('Detected result does not match the input image size')
                
                rslt = [poc, class_name, scores[ii]] + \
                    bboxes[ii].tolist()

                if self.task == 'seg':
                    rslt += \
                    [masks[ii].shape[1], masks[ii].shape[0], \
                    oid_mask_encoding.encode_binary_mask(masks[ii]).decode('ascii')]

                o_line = ','.join(map(str,rslt))
                results.append(o_line)
        return results
    
    def write_voc_results(self, results, res_path):
        with open(res_path, "at") as fp:
            for res in results:
                fp.write(f"{res}\n")

    def summary_coco_results(self, pred, poc=None):
        with torch.no_grad():  
            if "instances" in pred:
                instances = pred["instances"].to("cpu")
                res = instances_to_coco_json(instances, poc)
        return [res]
    
    def write_coco_results(self, results, res_path):
        coco_results = list(itertools.chain(*[x for x in results]))
        with open(res_path, "w") as f:
            f.write(json.dumps(coco_results, indent=4))
            f.flush()
    