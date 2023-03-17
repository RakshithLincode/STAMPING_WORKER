import sys

import numpy as np
sys.path.append("D:\deployment")

import os
import glob
import cv2
from mask_predict import *
# from classifier_module import *
# from common_utils import *
import torch
from datetime import  datetime
import uuid
import os
import sys

from utils.torch_utils import select_device

class opt_config():
    def __init__(self):
        self.base_path = ""
        self.detector_weights_path = "./mask_label.pt" # working
        self.separate_crop_model = False
        self.classifier_weights = ""
        self.segmentor_weights = ""
        self.ocr_weights = ""
        # self.batch_size_images = 6
        self.detector_input_image_size = 640
        self.common_conf_thres = 0.1
        self.iou_thres = 0.2
        self.max_det = 1000
        self.device = ""
        self.line_thickness = 2
        self.hide_labels = False
        self.hide_conf = True
        self.half = False
        self.crop = False
        self.cord = []
        self.crop_class = ""
        self.min_crop_size = None
        self.max_crop_size = None
        self.crop_conf = 0.25
        self.crop_iou = 0.25
        self.padding  = 50
        # self.crop_resize = (640,640)
        self.crop_hide_labels = True
        self.crop_hide_conf = True
        self.classes = None
        self.defects = ["rivert-missing","rivet-damage","pad-damage","pad-rivet-missing","pin-damage","pin-missing","wrong-position-assembly"]
        self.feature = ["rivert","pin","pad-rivet","A3","S"]
        self.features_extra = ["bi_barcode","BagID","des_barcode","A3","S"]
        self.visualize = False
        self.individual_thres = {"rivert-missing":0.1,"rivet-damage":0.1,"pad-damage":0.1,"pad-rivet-missing":0.1,"pin-damage":0.1,"pin-missing":0.1,"wrong-position-assembly":0.1}#best_22.pt
        self.rename_labels = {} # {'person':'manju'}
        self.avoid_labels_cords = [{'xmin':0,'ymin':0,'xmax':1280,'ymax':720},{'xmin':0,'ymin':6,'xmax':569,'ymax':548}]
        self.avoid_required_labels = ['person'] # ['person','cell phone']
        self.detector_predictions = None # This will update from the predictions
    

#@singleton
class Inference:
    def __init__(self):
        self.opt = opt_config()
        self.device = select_device(self.opt.device)
        self.half = self.opt.half
        self.crop = self.opt.crop
        self.cropped_frame = None
        self.input_frame = None
        self.predicted_frame = None
        self.detector_predictions = None
        self.detector , self.detector_stride , self.detector_names  = load_detector(weights = self.opt.detector_weights_path,
                                                        half = self.half,device = self.device ,
                                                        imgsz = self.opt.detector_input_image_size )

        

    def dummy(self):
        self.predicted_frame = detector_get_inference1(opt = self.opt,
                                                        im0 = self.input_frame , names = self.detector_names,
                                                        img_size = self.opt.detector_input_image_size ,
                                                        stride = self.detector_stride ,
                                                        model= self.detector , device = self.device,
                                                        half = self.half)
        return self.predicted_frame #, self.classifier_predictions
        # return self.predicted_frame