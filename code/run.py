import numpy as np
import cv2
from time import time
from utils.common import VideoManager, viz_text_bg
from utils.models import YOLOV8TRT, OpticalFlow, SlowonlyTrt
from utils.bytetrack.byte_tracker import BYTETracker

class FightDetection():
    def __init__(self):
        self.frames = []
        self.bboxs = []
        self.model_dt = SlowonlyTrt()

    def remove_bboxs(self, bboxs, _w=50, _h=50):
        w = bboxs[:, 2] - bboxs[:, 0]
        h = bboxs[:, 3] - bboxs[:, 1]
        mask = (w > _w) & (h > _h)
        bboxs = bboxs[mask]
        return bboxs
    
    def __call__(self, frame, bboxs):
        bboxs = self.remove_bboxs(bboxs)

        if len(self.frames) <= 16:
            self.frames.append(frame)
            self.bboxs.append(bboxs)
            return [], []
        
        scores = self.model_dt(self.frames, self.bboxs[8])
        self.frames = self.frames[8:]
        self.bboxs = self.bboxs[8:]
        return scores, self.bboxs[0]

class ABNORMAL:
    min_person = 1.4
    min_grad = 1
    live = 5
    min_ab_frames = 3

    def __init__(self):
        self.h_tracks = {}
        self.c_fs = []
    
    def save(self, tid, speed):
        if tid not in self.h_tracks:
            self.h_tracks[tid] = {"value": [], "end":time()}
        self.h_tracks[tid]["value"].append(speed)
        self.h_tracks[tid]["end"] = time()

    def remove_unuse_id(self):
        keys = [k for k in self.h_tracks]
        for k in keys:
            if time() - self.h_tracks[k]["end"] >= self.live:
                del self.h_tracks[k]

    def is_abnormal(self):
        c_f = []
        self.remove_unuse_id()
        for k in self.h_tracks:
            val = self.h_tracks[k]["value"]
            # chưa thu thập đủ min_ab_frames
            if len(val) <= self.min_ab_frames:
                continue
            i_get = -self.min_ab_frames if len(val) <= self.min_ab_frames*2 else (-self.min_ab_frames*2)
            grad = np.gradient(val[i_get:])
            grad = np.absolute(grad)
            c_f.append(grad.mean())
        num_person = sum([p >= self.min_grad for p in c_f])
        self.c_fs = self.c_fs[1:] if len(self.c_fs) == self.min_ab_frames else self.c_fs
        self.c_fs.append(num_person)
        return np.mean(self.c_fs) >= self.min_person

model = YOLOV8TRT("../weights/yolov8n_crowd.trt",
                    conf=0.5, nms=0.7)

l = [
    # "D01_20230720102254_0.mp4",
    "D01_20230724141749_0.mp4",
    # "D02_20230720102327_0.mp4",
    # "002.mp4",
    ]

for name in l:
    vid = VideoManager(f"../inp/{name}")
    tracker = BYTETracker()
    otp = OpticalFlow(320)
    ab = ABNORMAL()
    fight = FightDetection()
    for i, frame in vid.iter(skip=2):
        
        bboxs = model(frame)
        tracks = tracker.update(bboxs.copy())
        
        flow = otp.get(frame)
        for track in tracks:
            # update ABNORMAL
            tid = int(track[4])
            x1, y1, x2, y2 = otp.resize_bbox(track[:4], frame.shape[:2])
            speed = flow[y1:y2, x1:x2]
            speed = speed[speed>1]
            speed = 0 if len(speed) <= 0 else speed.mean()
            ab.save(tid, speed)

        if ab.is_abnormal():
            frame = viz_text_bg(frame, f"BAT THUONG", (50, 50, 50, 50), (0, 0, 255), 2)

        scores, bboxs = fight(frame, bboxs[:,:4])
        for score, bbox in zip(scores, bboxs):
            if score[64].item() >= 0.35:
                x1, y1, x2, y2 = np.int32(bbox)
                frame = viz_text_bg(frame, "", (x1, y1, x2, y2))

        # frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))
        # vid.write(frame, f"../inp/det/infer_{name}")