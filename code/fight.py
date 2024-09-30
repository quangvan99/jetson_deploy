# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import numpy as np
from utils.common import VideoManager, viz_text_bg
from utils.models import YOLOV8TRT, SlowonlyTrt

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
    
if __name__ == '__main__':
    model_dt = FightDetection()

    vid = VideoManager(f"../inp/HCM_danh.mp4")
    det = YOLOV8TRT("../weights/yolov8n_crowd.trt",
                    conf=0.5, nms=0.7)
    
    for i, frame in vid.iter(skip=2):
        bboxs = det(frame)
        bboxs = bboxs[:, :4]

        scores, bboxs = model_dt(frame, bboxs)
        for score, bbox in zip(scores, bboxs):
            if score[64].item() >= 0.35:
                x1, y1, x2, y2 = np.int32(bbox)
                frame = viz_text_bg(frame, "", (x1, y1, x2, y2))

        vid.write(frame, f"../inp/det/HCM_danh_infer.mp4")