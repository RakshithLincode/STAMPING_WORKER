# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
	$ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
																  img.jpg                         # image
																  vid.mp4                         # video
																  screen                          # screenshot
																  path/                           # directory
																  list.txt                        # list of images
																  list.streams                    # list of streams
																  'path/*.jpg'                    # glob
																  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
																  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
	$ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
										  yolov5s-seg.torchscript        # TorchScript
										  yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
										  yolov5s-seg_openvino_model     # OpenVINO
										  yolov5s-seg.engine             # TensorRT
										  yolov5s-seg.mlmodel            # CoreML (macOS-only)
										  yolov5s-seg_saved_model        # TensorFlow SavedModel
										  yolov5s-seg.pb                 # TensorFlow GraphDef
										  yolov5s-seg.tflite             # TensorFlow Lite
										  yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
										  yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import sys
sys.path.insert(1, 'D:/Segmentatin_yolo/yolov5/')
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
	# Resize and pad image while meeting stride-multiple constraints
	shape = im.shape[:2]  # current shape [height, width]
	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)

	# Scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	if not scaleup:  # only scale down, do not scale up (for better val mAP)
		r = min(r, 1.0)

	# Compute padding
	ratio = r, r  # width, height ratios
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
	if auto:  # minimum rectangle
		dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
	elif scaleFill:  # stretch
		dw, dh = 0.0, 0.0
		new_unpad = (new_shape[1], new_shape[0])
		ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

	dw /= 2  # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad:  # resize
		im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
	return im, ratio, (dw, dh)

def image_preprocess(image ,img_size=640, stride=32):
		# Padded resize
	img = letterbox(image, img_size , stride=stride)[0]

	# Convert
	img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
	# img = img.transpose()
	img = np.ascontiguousarray(img)
	return img

def load_detector(weights,half,device,imgsz):
	dnn=False
	data=ROOT / 'data/coco128.yaml' 
	bs = 6    
	half = False
	device = select_device(device)
	model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
	stride, names, pt = model.stride, model.names, model.pt
	imgsz = check_img_size(imgsz, s=stride)  # check image size
	imz = (1280, 1280)
	for i in range(6):
		model.warmup(imgsz=(1 if pt else bs, 3, *imz))
		seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
	print("Model loaded! and WarmUp is done!!")
	return model , stride , names

def detector_get_inference1(opt ,im0, names,img_size  ,stride, model, device ,half ):
	print("inside inference!!")
	predictions = []
	cord = []

	im = image_preprocess(im0 , img_size = img_size ,stride = stride)
	im = torch.from_numpy(im).to(model.device)
	im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
	im /= 255  # 0 - 255 to 0.0 - 1.0
	if len(im.shape) == 3:
		im = im[None]  # expand for batch dim

	pred = model(im, augment=False, visualize=False)
	agnostic_nms = False
	# Dataloader
	bs = 1  # batch_size
			
	pred, proto = model(im, augment=False, visualize=False)[:2]
	pred = non_max_suppression(pred, conf_thres = opt.crop_conf, iou_thres = opt.crop_iou, classes = None, agnostic = False, max_det=opt.max_det, nm=32)
	mask_results = []
	for i, det in enumerate(pred):  # per image
		# seen += 1
		retina_masks=True
		save_masks=True
		gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
		imc = im0.copy() 
		annotator = Annotator(im0, line_width=opt.line_thickness, example=str(names))
		if len(det):
			if retina_masks:
				# scale bbox first the crop masks
				det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
				masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
				masks_o = masks.detach().cpu().numpy()
				masks_o = numpy.around(numpy.array(masks_o))
				masks_o = masks_o.astype(int)
				c,img_w, img_h = masks_o.shape
				unified_masks = numpy.zeros((img_w, img_h))
				for mask in masks_o:
					unified_masks += mask
				mask_results.append(unified_masks)
			else:
				masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
				det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
				masks_o = masks.detach().cpu().numpy()
				masks_o = numpy.around(numpy.array(masks_o))
				masks_o = masks_o.astype(int)
				c,img_w, img_h = masks_o.shape
				unified_masks = numpy.zeros((img_w, img_h))
				for mask in masks_o:
					unified_masks += mask
				mask_results.append(unified_masks)
			return mask_results







