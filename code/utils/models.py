
from time import time
import torch
import torchvision
import cv2
import numpy as np
import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda
import threading


def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def non_max_suppression(outputs, conf_threshold, iou_threshold):
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]  # batch size
    nc = outputs.shape[1] - 4  # number of classes
    xc = outputs[:, 4:4 + nc].amax(1) > conf_threshold  # candidates

    start = time()
    limit = 0.5 + 0.05 * bs  # seconds to quit after

    output = [torch.zeros((0, 6), device=outputs.device)] * bs
    for index, x in enumerate(outputs):  # image index, image inference
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        box, cls = x.split((4, nc), 1)
        box = wh2xy(box)  # (cx, cy, w, h) to (x1, y1, x2, y2)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]

        if not x.shape[0]:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        i = i[:max_det]  # limit detections

        output[index] = x[i]
        if (time() - start) > limit:
            break  # time limit exceeded

    return output

def letterbox(img, new_shape = (640, 640), color = (114, 114, 114), 
              auto = False, scale_fill = False, scaleup = False, stride = 32):
    
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    height, width = img.shape[:2]
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, dw, dh, width, height

class BaseEngine(threading.Thread):
    def __init__(self, engine_path):
        threading.Thread.__init__(self)
        logger = trt.Logger()
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})


    def infer(self, *imgs):
        for i in range(len(imgs)):
            img = imgs[i].astype(np.float32)
            self.inputs[i]["host"] = np.ravel(img)
        # transfer data to the gpu
        try:
            for i in range(len(imgs)):
                cuda.memcpy_htod_async(self.inputs[i]['device'], self.inputs[i]['host'], self.stream)
        except:
            pass
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data


class BaseOnnx(threading.Thread):
    def __init__(self, path):
        threading.Thread.__init__(self)
        import onnxruntime
        self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])

    def pre(self, input_feed):
        for key in input_feed:
            input_feed[key] = input_feed[key].astype(np.float32)
        return input_feed

    def pos(self, out):
        return out[0]
    
    def infer(self, input_feed):
        out = self.session.run(None, input_feed=input_feed)
        return out
    
    def __call__(self, input_feed):

        input_feed = self.pre(input_feed)
        out = self.infer(input_feed)
        out = self.pos(out)
        return out

class SlowonlyTrt(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.mean = np.float64([123.675, 116.28, 103.53])
        self.norm = np.float64([58.395, 57.12, 57.375])
        self.frame_inds = [0,  2,  4,  6,  8, 10, 12, 15]
        self.backbone = BaseEngine("../weights/slowonly/slowonly_backbone.trt")
        # self.bbox_roi_extractor = BaseEngine("../weights/slowonly/slowonly_bbox_roi_extractor.trt")
        # self.bbox_head = BaseEngine("../weights/slowonly/slowonly_bbox_head.trt")
        self.bbox_roi_extractor = BaseOnnx("../weights/slowonly/slowonly_bbox_roi_extractor.onnx")
        self.bbox_head = BaseOnnx("../weights/slowonly/slowonly_bbox_head.onnx")

        self.size = 256

    def infer(self, input_tensor, rois):
        feat = self.backbone.infer(input_tensor)[0]
        feat = feat.reshape(1, 2048, 8, 16, 16)
        bbox_feats = self.bbox_roi_extractor({"feat":feat, "rois":rois})
        bbox_feats = bbox_feats.reshape(-1, 4096, 1, 8, 8)
        cls_score = self.bbox_head({"bbox_feats": bbox_feats})
        cls_score = cls_score.reshape(-1, 81)
        return cls_score

    def normalize(self, x):
        x = x.transpose(-1, 0, 1)
        x = (x - self.mean[:, None, None]) / self.norm[:, None, None]
        return x
    
    def resize_bbox(self, bbox, original_size, target_size):
    
        x1, y1, x2, y2 = bbox
        orig_width, orig_height = original_size
        target_width, target_height = target_size
        
        # Calculate scaling factors for width and height
        width_scale = target_width / orig_width
        height_scale = target_height / orig_height
        
        # Resize bounding box coordinates
        resized_x1 = int(x1 * width_scale)
        resized_y1 = int(y1 * height_scale)
        resized_x2 = int(x2 * width_scale)
        resized_y2 = int(y2 * height_scale)
        
        return resized_x1, resized_y1, resized_x2, resized_y2

    def pre(self, frames_data, bboxs):
        boxs = [self.resize_bbox(box, (frames_data[0].shape[1],frames_data[0].shape[0]), (self.size,self.size)) for box in bboxs]
        imgs = [cv2.resize(frames_data[ind], (self.size, self.size)).astype(np.float32) for ind in self.frame_inds]
        imgs = [self.normalize(img) for img in imgs]
        input_array = np.stack(imgs).transpose((1, 0, 2, 3))[np.newaxis]
        rois = np.insert(boxs, 0, 0, axis=1)
        return input_array, rois


    def pos(self, scores):
        scores = 1 / (1 + np.exp(-scores))
        return scores
    
    def __call__(self, frames_data, bboxs):
        if len(bboxs) <= 0:
            return []

        input_tensor, rois = self.pre(frames_data, bboxs)
        scores = self.infer(input_tensor, rois)
        scores = self.pos(scores)
        return scores

class YOLOV8TRT:
    def __init__(self, p_trt, conf=0.25, nms=0.45, target_size=640):
        self.model = BaseEngine(p_trt)
        self.conf = conf
        self.nms = nms
    
    def pre(self, img0):
        img0, w, h, width, height = letterbox(img0)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img0 = img0 / 255.0
        img0 = img0.transpose(2, 0, 1)
        return img0, w, h, width, height

    def post(self, outputs, w, h, height, width, shape):
        outputs = non_max_suppression(outputs, self.conf, self.nms)[0]
        outputs[:, [0, 2]] -= w  # x padding
        outputs[:, [1, 3]] -= h  # y padding
        outputs[:, :4] /= min(height / shape[0], width / shape[1])

        outputs[:, 0].clamp_(0, shape[1])  # x1
        outputs[:, 1].clamp_(0, shape[0])  # y1
        outputs[:, 2].clamp_(0, shape[1])  # x2
        outputs[:, 3].clamp_(0, shape[0])  # y2
        return outputs

    def __call__(self, img):
        shape = img.shape[:2] 
        x, w, h, width, height = self.pre(img)
        outputs = self.model.infer(x)[0]
        outputs = outputs.reshape(-1,8400)

        outputs = torch.from_numpy(outputs)[None, ...]
        outputs = self.post(outputs, w, h, height, width, shape)
        return outputs.numpy()

# import ncnn
# class YOLOV8_NCNN:
#     def __init__(self, p_param, p_bin, conf=0.25, nms=0.45,
#                  target_size=640, num_threads=4, use_gpu=True):
        
#         self.num_threads=num_threads
#         self.use_gpu=use_gpu
#         self.target_size=target_size
#         self.conf=conf
#         self.nms=nms
#         self.mean_vals = []
#         self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
#         self.net = ncnn.Net()
#         self.net.opt.use_vulkan_compute = self.use_gpu
#         self.net.opt.num_threads = self.num_threads
#         self.net.load_param(p_param)
#         self.net.load_model(p_bin)
    
#     def pre(self, img0):
#         img0, w, h, width, height = letterbox(img0)
#         mat_in = ncnn.Mat.from_pixels(img0,  ncnn.Mat.PixelType.PIXEL_BGR2RGB, self.target_size, self.target_size)
#         mat_in.substract_mean_normalize([], self.norm_vals)
#         return mat_in, w, h, width, height

#     def post(self, outputs, w, h, height, width, shape):
#         outputs = non_max_suppression(outputs, self.conf, self.nms)[0]
#         outputs[:, [0, 2]] -= w  # x padding
#         outputs[:, [1, 3]] -= h  # y padding
#         outputs[:, :4] /= min(height / shape[0], width / shape[1])

#         outputs[:, 0].clamp_(0, shape[1])  # x1
#         outputs[:, 1].clamp_(0, shape[0])  # y1
#         outputs[:, 2].clamp_(0, shape[1])  # x2
#         outputs[:, 3].clamp_(0, shape[0])  # y2
#         return outputs

#     def infer(self, mat_in_pad):
#         ex = self.net.create_extractor()
#         ex.input("in0", mat_in_pad)
#         _, out = ex.extract("out0")  # stride 8
#         return out

#     def __call__(self, img):
#         shape = img.shape[:2] 
#         x, w, h, width, height = self.pre(img)
#         outputs = self.infer(x)
#         outputs = torch.from_numpy(outputs.numpy())[None, ...]
#         outputs = self.post(outputs, w, h, height, width, shape)
#         return outputs.numpy()
        # return outputs



class OpticalFlow():
    def __init__(self, s_opt=320):
        self.prev = None
        self.s_opt = s_opt
    

    def resize_bbox(self, box, d_size):
        x1, y1, x2, y2 = box
        x_scale = self.s_opt / d_size[1]
        y_scale = self.s_opt / d_size[0]
        x1 = int(np.round(x1 * x_scale))
        y1 = int(np.round(y1 * y_scale))
        x2 = int(np.round(x2 * x_scale))
        y2 = int(np.round(y2 * y_scale))
        return x1, y1, x2, y2 
        
    def get(self, frame):

        frame = cv2.resize(frame, (self.s_opt, self.s_opt))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev is None:
            self.prev = frame

        flow = cv2.calcOpticalFlowFarneback(self.prev, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev = frame
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        return mag




