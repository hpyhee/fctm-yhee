import torch
import sys, os, time

JDEPATH = "../Towards-Realtime-MOT"
DEVICE = "cpu"

sys.path.append(JDEPATH)


import warnings
warnings.filterwarnings('ignore')

from utils import datasets
from tracker.multitracker import *
from track import write_results

from numba import jit
from collections import deque
import torch
from utils.kalman_filter import KalmanFilter
from utils.log import logger
from models import *
from tracker import matching
from tracker.basetrack import BaseTrack, TrackState

def yolov3_part1(yolo, x, split_points):
    with torch.no_grad():
        x = x.to(DEVICE)
        layer_outputs = []
        output = []
        ye=[80,81,82,83,84,85,95,96,97,98,99,100]#yolo_embedding
        for i, (module_def, module) in enumerate(zip(yolo.module_defs, yolo.module_list)):
            if i in ye:
                layer_outputs.append(None)
                continue
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                x = module[0](x, yolo.img_size)
            layer_outputs.append(x)
            if i in split_points:
                output.append(x)
                if i == split_points[-1]:
                    break
    return output

def yolov3_part2(yolo, recon_feature, split_points, start_point):
    layer_outputs = []
    output = []
    
    with torch.no_grad():
        rec_idx = 0
        for i, (module_def, module) in enumerate(zip(yolo.module_defs, yolo.module_list)):
            if i in split_points:
                layer_outputs.append(recon_feature[rec_idx])
                rec_idx += 1
                continue

            elif i < start_point:
                layer_outputs.append(None)
                continue
            
            x = layer_outputs[i-1]
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                x = module[0](x, yolo.img_size)
                output.append(x)

                for s in split_points:
                    if s > start_point:
                        start_point = s
                        break

            layer_outputs.append(x)

    return torch.cat(output, 1)

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

class JDE:
    def __init__(self, frame_rate, split_points, start_point) -> None:
        self.model = Darknet(f"{JDEPATH}/cfg/yolov3_1088x608.cfg", nID=14455)
        self.model.load_state_dict(torch.load(f"{JDEPATH}/weights/jde.1088x608.uncertainty.pt", map_location='cpu')['model'], strict=False)
        self.model = self.model.to(DEVICE)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks    = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = 0.5
        self.track_buffer = 30
        self.min_box_area = 200
        self.conf_thres   = 0.5
        self.data_type    = 'mot'
        self.img_size     = [1088, 608]
        self.nms_thres = 0.4
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

        self.split_points = split_points
        self.start_point  = start_point

        self.part1_proc_time = 0
        self.part1_perf_time = 0

        self.part2_proc_time = 0
        self.part2_perf_time = 0
    
    @PART1
    def part1(self, im_blob):
        start_proc_time = time.process_time()
        start_perf_time = time.perf_counter()
        
        mlf = yolov3_part1(self.model, im_blob, self.split_points)
        
        self.part1_proc_time += time.process_time()-start_proc_time
        self.part1_perf_time += time.perf_counter()-start_perf_time

        return mlf
    
    @PART2
    def part2(self, recon_feature, oimg_size):
        """
        Processes the image frame and finds bounding box(detections).

        Associates the detection with corresponding tracklets and also handles lost, removed, refound and active tracklets

        Parameters
        ----------
        im_blob : torch.float32
                  Tensor of shape depending upon the size of image. By default, shape of this tensor is [1, 3, 608, 1088]

        img0 : ndarray
               ndarray of shape depending on the input image sequence. By default, shape is [608, 1080, 3]

        Returns
        -------
        output_stracks : list of Strack(instances)
                         The list contains information regarding the online_tracklets for the recieved image tensor.

        """
        self.frame_id += 1
        activated_starcks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            pred = yolov3_part2(self.model, recon_feature, self.split_points, self.start_point)

        # pred is tensor of all the proposals (default number of proposals: 54264). Proposals have information associated with the bounding box and embeddings
        pred = pred[pred[:, :, 4] > self.conf_thres]
        # pred now has lesser number of proposals. Proposals rejected on basis of object confidence score
        if len(pred) > 0:
            dets = non_max_suppression(pred.unsqueeze(0), self.conf_thres, self.nms_thres)[0].cpu()
            # Final proposals are obtained in dets. Information of bounding box and embeddings also included
            # Next step changes the detection scales
            scale_coords(self.img_size, dets[:, :4], oimg_size).round()
            '''Detections is list of (x1, y1, x2, y2, object_conf, class_score, class_pred)'''
            # class_pred is the embeddings.

            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f.numpy(), 30) for
                          (tlbrs, f) in zip(dets[:, :5], dets[:, 6:])]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                unconfirmed.append(track)
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.kalman_filter)

        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # The dists is the list of distances of the detection with the tracks in strack_pool
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # The matches is the array for corresponding matches of the detection with the corresponding strack_pool

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # If the track is active, add the detection to the track
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # We have obtained a detection from a track which is not active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # None of the steps below happen if there are no undetected tracks.
        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections
        r_tracked_stracks = [] # This is container for stracks which were tracked till the
        # previous frame but no detection was found for it in the current frame
        for i in u_track:
            if strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        # matches is the list of detections which matched with corresponding tracks by IOU distance method
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Same process done for some unmatched detections, but now considering IOU_distance as measure

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # If no detections are obtained for tracks (u_track), the tracks are added to lost_tracks list and are marked lost

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # after all these confirmation steps, if a new detection is found, it is initialized for a new track
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks
    
    @PART2
    def init_results(self):
        return []

    @PART2
    def summary_results(self, pred, poc=None):
        online_tlwhs = []
        online_ids = []
        for t in pred:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)

        return [(self.frame_id, online_tlwhs, online_ids)]
    
    @PART2
    def write_results(self, results, res_path):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        with open(res_path, 'w') as f:
            for frame_id, tlwhs, track_ids in results:
                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                    f.write(line)
