import numpy as np
import cv2
from tqdm import tqdm
from time import time
import warnings
warnings.filterwarnings("ignore")

def viz_text_bg(frame, text, pos, color=(255, 0, 0), font_scale = 1, thickness=1):
    x1, y1, x2, y2 = pos
    txt_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    frame[y1:y1+txt_size[1], x1:x1+txt_size[0]] = color
    cv2.putText(frame, text, (x1, y1 + txt_size[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness=thickness)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame

#================ VIDEO MANAGER ==================
class VideoReader:
    def __init__(self, p):
        self.p = p
        self.cap = cv2.VideoCapture(p)
        self.width  = int(self.cap.get(3))  # float `width`
        self.height = int(self.cap.get(4))  # float `height`
        self.fps = int(self.cap.get(5))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def destroy(self):
        self.cap.release()
        
    def __getitem__(self, index):
        count = 0
        while True:
            ret, frame = self.cap.read()
            if ret == True and count < index:
                count += 1
            else:
                break

        self.destroy()
        self.cap = cv2.VideoCapture(self.p)
        return frame

    def __len__(self):
        return self.n

    def play(self):
        while(True):
            
            _, frame = self.cap.read()
            
            if frame is None:
                break

            # Display the resulting frame
            cv2.imshow('frame', frame)
            

        self.cap.release()
        cv2.destroyAllWindows()


class VideoWriter:
    def __init__(self, p, fps, w, h):
        self.out = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (w, h))

    def destroy(self):
        self.out.release()

    def write(self, frame):
        self.out.write(frame)

class VideoManager:
    def __init__(self, p):
        self.read = VideoReader(p)
        self.exports = {}
        self.n_vid = len(self.read)

    def __len__(self):
        return self.n_vid
    
    def iter(self, skip=1):
        self.n_vid = len(self)//skip
        self.pbar = tqdm(range(len(self)), bar_format='{percentage:3.0f}%|{bar:10}|[{elapsed}<{remaining}{postfix}]')
        for i in self.pbar:
            start = time()

            self.postfix_str = "" 

            for _ in range(skip):
                _, frame = self.read.cap.read()

            if frame is None:
                break

            yield i, frame

            end = time()
            self.postfix_str += f"frame<{i+1}:{len(self.pbar)}"
            self.postfix_str += f", fps={skip/(end-start):.2f}"
            self.pbar.set_postfix_str(self.postfix_str)
        
        self.read.destroy()
        for k in self.exports:
            self.exports[k].destroy()
        # cv2.destroyAllWindows()
    
    def write(self, frame, p_out):
        name = p_out.split('/')[-1].split(".")[0]
        if name not in self.exports:
            self.exports[name] = VideoWriter(p_out, self.read.fps, frame.shape[1], frame.shape[0])
        
        if len(frame.shape) < 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        self.exports[name].write(frame)

    def show(self, frame, name="frame"):
        cv2.imshow(name, frame)