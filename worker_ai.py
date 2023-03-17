import torch
import pickle
import numpy as np
import cv2 as cv
import cv2 
import matplotlib.pyplot as plt
import glob
import os
import redis
import argparse
import io
from PIL import Image
import cv2
import torch
import requests
from flask import Flask, request
import base64
from io import BytesIO
import time
import json
from flask import jsonify
from flask import Response
import numpy as np
# from mongohelper import MongoHelper
from base64 import decodestring
import pandas as pd
import os
from PIL import Image
import datetime
from json import dumps
import copy
import bson
import cv2 as cv
import cv2 
import math
import numpy as np
from scipy.spatial import distance as dist
import numpy as np
from pymongo import MongoClient
from ai_settings import *

def singleton(cls):
	instances = {}
	def getinstance():
		if cls not in instances:
			instances[cls] = cls()
		return instances[cls]
	return getinstance

@singleton
class CacheHelper():
	def __init__(self):
		self.redis_cache = redis.StrictRedis(host='localhost', port=6379, db=0, socket_timeout=1)
		print("REDIS CACHE UP!")

	def get_redis_pipeline(self):
		return self.redis_cache.pipeline()
	
	def set_json(self, dict_obj):
		try:
			k, v = list(dict_obj.items())[0]
			v = pickle.dumps(v)
			return self.redis_cache.set(k, v)
		except redis.ConnectionError:
			return None

	def get_json(self, key):
		try:
			temp = self.redis_cache.get(key)
			#print(temp)\
			if temp:
				temp= pickle.loads(temp)
			return temp
		except redis.ConnectionError:
			return None
		return None

	def execute_pipe_commands(self, commands):
		#TBD to increase efficiency can chain commands for getting cache in one go
		return None

@singleton
class MongoHelper:
	try:
		client = None
		def __init__(self):
			if not self.client:
				self.client = MongoClient(host=MONGO_SERVER_HOST, port=MONGO_SERVER_PORT)
			self.db = self.client[MONGO_DB]

		def getDatabase(self):
			return self.db

		def getCollection(self, cname, create=False, codec_options=None):
			_DB = MONGO_DB
			DB = self.client[_DB]
			if cname in MONGO_COLLECTIONS:
				if codec_options:
					return DB.get_collection(MONGO_COLLECTIONS[cname], codec_options=codec_options)
				return DB[MONGO_COLLECTIONS[cname]]
			else:
				return DB[cname]
	except:
		pass      

rch = CacheHelper()

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denominator = ((y4 - y3) * (x2 - x1)) - ((x4 - x3) * (y2 - y1))
    if denominator == 0:
        return None
    numerator1 = ((x4 - x3) * (y1 - y3)) - ((y4 - y3) * (x1 - x3))
    numerator2 = ((x2 - x1) * (y1 - y3)) - ((y2 - y1) * (x1 - x3))
    t1 = numerator1 / denominator
    t2 = numerator2 / denominator
    intersection_x = x1 + (t1 * (x2 - x1))
    intersection_y = y1 + (t1 * (y2 - y1))
    return intersection_x, intersection_y

def find_nearest_cord(res,cord):
    A = np.array(res)
    leftbottom = np.array(cord)
    distances = np.linalg.norm(A-leftbottom, axis=1)
    min_index = np.argmin(distances)
    res_value = res[min_index]
    return res_value

def draw_find_circleand_find_centroids(frame):
        frameb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_g = cv2.cvtColor(frameb, cv2.COLOR_BGR2GRAY)
        dst = cv2.Canny(frame_g, 50, 200)
        lines= cv2.HoughLines(dst, 1, math.pi/180.0, 100, np.array([]), 0, 0)
        try:
            a,b,c = lines.shape
            line_1 = []
            line_2 = []
            radius = []
            for i in range(a):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0, y0 = a*rho, b*rho
                y2 = int(y0-1000*(a))
                y1 = int(y0+1000*(a))
                x2 = int(x0-1000*(-b))
                x1 = int(x0+1000*(-b))
                angle = np.rad2deg(np.arctan2(y2-y1, x2-x1))
                if abs(angle) > 20 and abs(angle) < 100:
                    # cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    start = (x1, y1), (x2, y2)
                    line_2.append(start)
            gray_blurred = cv2.blur(frame_g, (3, 3))
            detected_circles = cv2.HoughCircles(gray_blurred,cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,param2 = 30, minRadius = 1, maxRadius = 40)
            if detected_circles is not None:
                detected_circles = np.uint16(np.around(detected_circles))
                test_list = []
                for pt in detected_circles[0, :]:
                    a, b, r = pt[0], pt[1], pt[2]
                    test_list.append([a,b])
                    radius.append(r)
                    cv2.circle(frame, (a, b), r, (0, 255, 0), 2)
                    width, height, channels = frame.shape
                    line_length = int(max(width, height) / 2)
                    cv2.circle(frame, (a, b), 1, (0, 0, 255), 10)
                a = test_list[0][0]
                b = test_list[0][1]
                c = test_list[1][0]
                d = test_list[1][1]
                line_length = int(max(width, height) / 2)
                print(line_length)
                temp = test_list[:]
                temp.sort()
                print(test_list)
                if not (temp == test_list):
                    radius = [radius[1],radius[0]]
                else:
                    radius = radius[:]    
                temp =[[int(temp[0][0]),int(temp[0][1])] , [int(temp[1][0]),int(temp[1][1])]]
                point1 = temp[0]
                point2 = temp[1]
                print(point1)
                m = (point2[1]-point1[1])/(point2[0]-point1[0])
                b = point1[1] - m*point1[0]
                endpoint1 = (0, int(b))
                endpoint2 = (1920, int(m*1920+b))
                end = endpoint1,endpoint2
                line_1.append(end)
                print(endpoint2,endpoint1)
        except:
            pass    
        return temp , frame , line_1 ,line_2 ,radius

    
def acceptedlogic(radius):
    measurment_values = MongoHelper().getCollection('measurment_values')
    measurment_values = measurment_values.find_one()
    isAccpeted= []
    color_value = []
    position_value = []
    value_mis = []
    if(min(radius)>=measurment_values['angle_value']):
        isAccpeted.append('Accepted')
        color_value.append((0,255,0))
        position_value.append('None')  
        value_mis.append('None')   
    else:
        index = radius.index(min(radius))
        if index == 0:
            position_value.append('Left')
        else:
            position_value.append('Right')
        isAccpeted.append('Rejected')
        color_value.append((0,0,255))
        value_mis.append(min(radius))   
    return isAccpeted,position_value,color_value,value_mis 


def predict():
    while 1:
        mp = MongoHelper().getCollection("current_inspection")
        data = mp.find_one()
        try:
            current_inspection_id = data.get('current_inspection_id')
            print(current_inspection_id)

            if current_inspection_id is None:
                continue
        except:
            pass
        vid = cv2.VideoCapture(r"D:\Stamping tool\Bad 1.mp4")
        while(True):
            ret, frame = vid.read()
            if not ret:
                break
            if frame is None:
                continue
            rch.set_json({'input_frame':frame})
            ocr_frame = copy.copy(frame)
            centroids , canvas , line_1 , line_2 ,radius= draw_find_circleand_find_centroids(frame)
            value = []
            try:
                for i in line_2:
                    line1 = (line_1[0][0][0],line_1[0][0][1],line_1[0][1][0],line_1[0][1][1])
                    line2 = (i[0][0],i[0][1],i[1][0],i[1][1])
                    intersection_x, intersection_y = intersection(line1, line2)
                    intersection_line = int(intersection_x),int(intersection_y)
                    value.append(intersection_line)
                res = [*set(value)]
                res.sort()
                res_value = []
                for cord in centroids:
                    index = find_nearest_cord(res,cord)
                    res_value.append(index)
                for l1,l2 in zip(centroids,res_value): 
                    cv2.line(canvas, l1, l2, (0, 0, 255), 2)
                print(res_value)
                print(centroids)  
                width = 1920
                A = dist.euclidean(centroids[0], res_value[0]) / width
                B = dist.euclidean(centroids[1],res_value[1]) / width
                A = A * 25.4
                B = B * 25.4
                A = "{:.1f}mm".format(A) 
                B = "{:.1f}mm".format(B)
                print(A,B)
                status,position,color,value_mis = acceptedlogic(radius)
                cv2.putText(frame, str(A), (res_value[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,250), 2)
                cv2.putText(frame, str(B), (res_value[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,250), 2)
                # cv2.putText(frame, "Radius of Left Circle : "+str(radius[0]), (100,50),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                # cv2.putText(frame,  "Radius of Right Circle : "+str(radius[1]), (100,100),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                # cv2.putText(frame, "length in mm from Left Centroid to Left Part Edge : "+str(A), (100,150),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                # cv2.putText(frame,  "length in mm from Right Centroid to Right Part Edge : "+str(B), (100,200),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                # cv2.putText(frame,  "Status : "+str(status[0]), (100,250),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, color[0], 2)
                canvas = cv2.resize(canvas,(1280,720))    
            except:
                # cv2.putText(frame, "Radius of Left Circle : None", (100,50),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                # cv2.putText(frame,  "Radius of Right Circle : None", (100,100),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                # cv2.putText(frame, "length in mm from Left Centroid to Left Part Edge : None", (100,150),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                # cv2.putText(frame,  "length in mm from Right Centroid to Right Part Edge : None", (100,200),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                # cv2.putText(frame,  "Status : None", (100,250),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                canvas = cv2.resize(frame,(1280,720))      
            trigger = CacheHelper().get_json('inspection_trigger')
            print(trigger)
            if trigger == True:
                worker_start = time.time()
                select_model = CacheHelper().get_json('current_part_name')
                print(select_model,'select_model..................................................................')
                part_name = select_model
                is_accepted = status[0]
                x = bson.ObjectId()
                cv2.imwrite(datadrive_path+str(x)+'_ip.jpg',ocr_frame)
                # cv2.imwrite(datadrive_path+str(x)+'_mask.jpg',mask_frame)
                # cv2.imwrite(datadrive_path+str(x)+'_measure.jpg',measure_frame)
                cv2.imwrite(datadrive_path+str(x)+'_pf.jpg',canvas)
                input_frame_path = 'http://localhost:3306/'+str(x)+'_ip.jpg'
                # mask_frame_path = 'http://localhost:3306/'+str(x)+'_mask.jpg'
                # measure_frame_path = 'http://localhost:3306/'+str(x)+'_measure.jpg'
                predicted_frame_path = 'http://localhost:3306/'+str(x)+'_pf.jpg'
                print(input_frame_path)
                measurment_values = MongoHelper().getCollection('measurment_values')
                measurment_values = measurment_values.find_one()
                rch.set_json({"input_frame_path":input_frame_path})
                rch.set_json({"right_length":str(B)})
                rch.set_json({"left_length":str(A)})
                rch.set_json({"inference_frame":predicted_frame_path})
                rch.set_json({"status":is_accepted})
                rch.set_json({"feature_mismatch":'features'})
                rch.set_json({"defects":position[0]})
                rch.set_json({"label_angle":str(measurment_values['angle_value'])})
                rch.set_json({"left_radius":str(radius[0])})
                rch.set_json({"right_radius":str(radius[1])})
                rch.set_json({"radius_defect_value":value_mis[0]})
                data = {'current_inspection_id':str(current_inspection_id)}#,'raw_frame':input_frame_path,'inference_frame':inference_frame_path,'status':is_accepted,'defect_list':conf.defects,'feature_list':conf.feature,'features':[],'defects':defect_list}
                requests.post(url = 'http://localhost:8000/livis/v1/inspection/save_inspection_details/', data = data)
                CacheHelper().set_json({'inspection_trigger':False})
                print("Worker_Time_Taken",time.time() - worker_start)
                if cv2.waitKey(1) == 27: 
                    break  # esc to quit
            cv2.destroyAllWindows()
        
if __name__ == "__main__":
    start = time.time()
    # parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    # parser.add_argument("--port", default=5000, type=int, help="port number")
    # args = parser.parse_args()
    datadrive_path = 'D:/Stamping tool/DEMO/datadrive/'
    print("load architecture",time.time() - start)
    predict()

    
