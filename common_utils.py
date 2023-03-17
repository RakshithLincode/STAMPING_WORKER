import torch
import os.path as osp
import pickle
import numpy as np
import cv2 as cv
import cv2 
import matplotlib.pyplot as plt
import glob
import os
from paddleocr import PaddleOCR,draw_ocr 
import PIL
from urllib.request import urlopen
import ast
import zxingcpp
import string
import ast
import keyboard
import math
from pyzbar import  pyzbar
from pymongo import MongoClient
import time
from scipy.spatial import distance as dist
import redis 
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



# frame = stream()
# print(frame)
# rch.set_json({'frame':frame})

class Hemlock_Process():
	def __init__(self):
		self.model_dir = './LINCODE_AI'
		self.weights_path = "./AI_WEIGHTS/hemlock_version_1.pt"
		self.image_size = 640
		self.common_confidence = 0.1
		self.common_iou = 0.45
		self.line_thickness = None
		
		## If your renaming labels then defects names should be renamed labels , for ex. your label is 'cell phone' if you want to rename that to 'Mobile Phone' then defecet should be 'Mobile Phone'
		self.ind_thresh = {'bi_barcode':0.34,"BagID":0.5,"des_barcode":0.5,"A3":0.65,"S":0.3}
		self.rename_labels =  {'ma':'main_logo'} # {'person':'manju'}
		## avoid labels with in the given co-ordinates
		self.avoid_labels_cords = [{'xmin':184,'ymin':188,'xmax':379,'ymax':385}]
		self.avoid_required_labels = ['burr','operation_missing'] # ['person','cell phone']

		self.HSC140_features=["sigma","A3","S","main_logo"]
		self.HSC440_features=["A3"]
		self.HSC760_features=[]
		self.POLY199_features=["C"]

		self.HSC140_defects=["Improper_logo"]
		self.HSC440_defecst=["S","sigma"]
		self.HSC760_defects=[]
		self.POLY199_defects=["S","sigma","A3"]

		self.defects=["improper_logo"]
		self.ocr_lables=['Net_Wt', 'Bag_ID', 'Batch_ID', 'Description', 'Material_ID']
		self.barcode_labels=["bi_barcode","mi_barcode","nw_barcode","bagid_barcode"]
		self.detector_predictions=None
		self.unique_logo=["main","DE"]

	def load_model(self):
		model = torch.hub.load(self.model_dir,'custom',path=self.weights_path,source = 'local',force_reload=True,autoshape=True)
		model.conf = self.common_confidence
		model.iou = self.common_iou
		return model

	def run_inference_hub(self,model, image):
		results = model(image,size=self.image_size)
		labels = results.pandas().xyxy[0]
		labels = list(labels['name'])
		result_dict = results.pandas().xyxy[0].to_dict()
		labels_ = []
		coordinates = []
		for i in range(len(labels)):
			xmin = list(result_dict.get('xmin').values())[i]
			ymin = list(result_dict.get('ymin').values())[i]
			xmax = list(result_dict.get('xmax').values())[i]
			ymax = list(result_dict.get('ymax').values())[i]
			c = list(result_dict.get('class').values())[i]
			name = list(result_dict.get('name').values())[i]
			confidence = list(result_dict.get('confidence').values())[i]
	
			## avoid labels with the given co ordinates
			skip = None
			if self.avoid_labels_cords:
				if bool(self.avoid_required_labels):
					for label in self.avoid_required_labels:
						if label == name:
							for crd in self.avoid_labels_cords:
								if round(xmin) >= crd['xmin'] and round(ymin) >= crd['ymin'] and round(xmax) <= crd['xmax'] and round(ymax) <= crd['ymax']:
									skip = True
				else:
					for crd in self.avoid_labels_cords:
						if round(xmin) >= crd['xmin'] and round(ymin) >= crd['ymin'] and round(xmax) <= crd['xmax'] and round(ymax) <= crd['ymax']:
							skip = True
			if skip :
				continue

			## line width
			line_width = self.line_thickness or max(round(sum(image.shape) / 2 * 0.003), 2)

			## Checking individual threshold for wach label 
			if name in self.ind_thresh:
				try:
					if self.ind_thresh.get(name) <= confidence:

						p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
						
						if name:
							namer = self.rename_labels.get(name)
							if namer is None:
								name = name
							else:
								name = namer			
							
							## Bounding color   
							if name in self.defects:
								color = (0,0,255) # Red color bounding box 
							else:
								color = (0,128,0) # Green color bounding box 


							cv2.rectangle(image, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)
							
							tf = max(line_width - 1, 1)  # font thickness
							
							w, h = cv2.getTextSize(name, 0, fontScale=line_width / 3, thickness=tf)[0]  # text width, height
							outside = p1[1] - h - 3 >= 0  # label fits outside box
							p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
							cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
							coordinates.append({name:[int(xmin),int(ymin),int(xmax),int(ymax)]})
							cv2.putText(image, name, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, line_width / 3, (255,255,255),
										thickness=tf, lineType=cv2.LINE_AA)
								
							labels_.append(name)     
				except:
					pass
			## If not individual threshold
			else:
				# line_width or max(round(sum(im.shape) / 2 * 0.003), 2)
				p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))	
				if name:
					namer = self.rename_labels.get(name)
					if namer is None:
						name = name
					else:
						name = namer
					## Bounding color   
					if name in self.defects:
						color = (0,0,255) # Red color bounding box 
					else:
						color = (0,128,0) # Green color bounding box
					cv2.rectangle(image, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)

					tf = max(line_width - 1, 1)  # font thickness
					w, h = cv2.getTextSize(name, 0, fontScale=line_width / 3, thickness=tf)[0]  # text width, height
					outside = p1[1] - h - 3 >= 0  # label fits outside box
					p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
					cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
					coordinates.append({name:[int(xmin),int(ymin),int(xmax),int(ymax)]})
					cv2.putText(image, name, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, line_width / 3, (255,255,255),
								thickness=tf, lineType=cv2.LINE_AA)
					labels_.append(name)			

		self.detector_predictions = labels_
		for img in results.ims:
			return img, labels_, coordinates


	def get_fetaure_list(self,detector_predections,select_model):
		feature_list = []
		mp = MongoHelper().getCollection("parts")
		for x in mp.find():
			if x['select_model'] == select_model:
				features = x.get('features')
				for feature in features:
					if not feature in detector_predections:
						feature_list.append(feature) 
		return feature_list

	def get_defect_list(self,detector_predections,select_model):
		defect_list = []
		mp = MongoHelper().getCollection("parts")
		for x in mp.find():
			if x['select_model'] == select_model:
				defects = x.get('defeats')
				for defect in defects:
					if defect in detector_predections:
						defect_list.append(defect) 
		return defect_list

	def get_ocr_labels(self,detector_predections):
		ocr_labels_list=[] #  Label list from features using detection
		for ocr_labels in self.ocr_lables:
			if  ocr_labels in detector_predections:
				ocr_labels_list.append(ocr_labels)
		return ocr_labels_list

	def get_unique_logo(self,image,cords={}):
		unique_logo_results=None
		if bool(cords):
			crop_image = image[cords.get('ymin'):cords.get('ymax'), cords.get('xmin'):cords.get('xmax')]    
		else:
			crop_image = crop_image		
		ocr = PaddleOCR(use_angle_cls=True)
		result1 = ocr.ocr(crop_image)
		try:
			for line in result1:
				txts = [i[1][0] for i in line]
				for i in txts:
					unique_logo_results=i
					print(unique_logo_results,"*unique logo values from paddle ocr  ")
		except IndexError:
			print("need to adjust values")
		# print("unique_logo_results",unique_logo_results)
		
		return unique_logo_results

		# unique_logo_list=[],
		# for unique_logo in self.unique_logo:
		# 	if not unique_logo  in detector_predections:
		# 		unique_logo_list.append(unique_logo)
		# return unique_logo_list


	def get_barcode_labels(self,detector_predections):
		barcode_labels_list=[]
		for barcode_lables in detector_predections:
			if barcode_lables in self.barcode_labels:
				barcode_labels_list.append(barcode_lables)
				
		print(barcode_labels_list,"get_barcode_labels")
		return barcode_labels_list
	

	def get_ocr_results(self,image, cords={}):
		'''
		cords = {'xmin':1,'ymin':20,'xmax':22,'ymax':33}
		'''
		ocr_results=None
		if bool(cords):
			crop_image = image[cords.get('ymin'):cords.get('ymax'), cords.get('xmin'):cords.get('xmax')]
		else:
			crop_image = image
		ocr = PaddleOCR(use_angle_cls=True,use_gpu=True)
		result1 = ocr.ocr(crop_image)
		try:
			for line in result1:
				txts = [i[1][0] for i in line]
				for i in txts:
					if "kg" in i or "BatchID:"  in i:
						pass
					else:
						ocr_results=i
					print(ocr_results,"*ocr_results numbers from infrence ")
		except IndexError:
			print("need to adjust values")
		return ocr_results

	def get_barcode_results(self,crop_image):
		barcode_data=None
		crop_image=cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)
		ret, crop_image = cv2.threshold(crop_image,100,200 ,cv2.THRESH_BINARY)
		try:
			net_barc = pyzbar.decode(crop_image)
			barcode_data = net_barc[0][0].decode()
			print(barcode_data,"barcode code from inferenec***************")
		except:
				print("no data found $#################################")
		return barcode_data


	def check_barcode(self,ocr_dict,barcode_dict):
		'''
		ocr_dict = {'bag_id':'value','material_id':'value','net_weight':'value','batch_id':'value'}
		barcode_dict = {'bag_id':'value','material_id':'value','net_weight':'value','batch_id':'value'}

		'''
		compare_list=[] #  Compared Mismatch values 
		print(ocr_dict,"ocrrrrr dictt in chcvk barcode")
		print(barcode_dict,"barcode_dict dictt in chcvk barcode")
		try:
			barcode_dict["BI"] = barcode_dict.pop("bi_barcode")
			barcode_dict["MI"] = barcode_dict.pop("mi_barcode")
			barcode_dict["NW"] = barcode_dict.pop("nw_barcode")
			barcode_dict["BagID"] = barcode_dict.pop("bagid_barcode")
		except KeyError:
			print("hellooo")
		for i in ocr_dict:
			print(ocr_dict[i],"*******")
			print(barcode_dict[i],"###################")
			for sub in barcode_dict:
				if 'Description' in sub and len(barcode_dict[sub]) > 0:
					barcode_dict['Description']  = ocr_dict['Description']
			if not (ocr_dict[i] == barcode_dict[i]):
				compare_list.append(i)
		print("compare_list",compare_list)
		return compare_list

	def get_measurement(self,angle):
		angle_value = [] # Angle Mismatch List Values
		if angle_value == None:
			status = 'True'
		else:
			for angle in angle:
				measurment_values = MongoHelper().getCollection('measurment_values')
				measurment_values = measurment_values.find_one()
				if int(angle) < measurment_values['angle_value'] :
					status = 'False'
					angle_value.append('None')
				else : 
					status = 'True'
					angle_value.append(angle)
				return status,angle_value  

	def get_length(self,measure):
		length_value = [] # Length Mismatch List Values
		if measure == 'None':
			status = 'True'
			pass
		else:
			for i in measure:
				print(int(float(i)))
				measurment_values = MongoHelper().getCollection('measurment_values')
				measurment_values = measurment_values.find_one()
				if int(float(i)) >= measurment_values['label_to_sealer_value_1'] and int(float(i)) <= measurment_values['label_to_sealer_value_2']:
					status = 'False'
					length_value.append(None)
				else : 
					status = 'True'
					length_value.append(measure)
				return status,length_value   

	def check_kanban(self,detector_predections,ocr_dict,barcode_dict,angle,measure,select_model):
		responce = {'features':self.get_fetaure_list(detector_predections,select_model),'defects':self.get_defect_list(detector_predections,select_model),'ocr_barcode_mismatch':self.check_barcode(ocr_dict,barcode_dict),'label_angle':self.get_measurement(angle)[1],'label_to_sealent_measurment':self.get_length(measure)[1],'status':None}
		if bool(self.get_fetaure_list(detector_predections,select_model)) or bool(self.get_defect_list(detector_predections,select_model)) or bool(self.check_barcode(ocr_dict,barcode_dict)) or "True" in  self.get_measurement(angle) or "True" in  self.get_length(measure) :
			responce['status'] = 'Rejected'
		else:
			responce['status'] = 'Accepted'
		return responce                 

	def barcode_decoder(self,img):
		list_str = []
		result_list = []
		results = zxingcpp.read_barcodes(img)
		for result in results:
			value = format(result.text)
			print(value)
			value = str(value).replace(' ', ',')
			value = value.split(",")  
			value = [i.split('|', 9) for i in value]
			for i in value:
				value = str(i)[1:-1]
				result = ast.literal_eval(value)
				list_str.append(result)
		for i in list_str:
			if type(i) is tuple:
				for k in i:
					result_list.append(k)
			else:
				result_list.append(i) 
		result_list = [*set(result_list)]
		if len(results) == 0:
			print("Could not find any barcode.")
		return result_list 

	def convert_list_order(self,value):
		count = 0
		barcode_dict = {} # Barcode Dict Values
		list_value = []
		list_check = ['Net_Wt', 'Bag_ID', 'Description', 'Material_ID', 'Batch_ID']
		for i in value:
			if len(i) == 8:
				barcode_dict['Bag_ID'] = i 
				list_value.append('Bag_ID')
			if len(i) == 7:
				barcode_dict['Material_ID'] = i
				list_value.append('Material_ID')
			if i == '5.0':
				barcode_dict['Net_Wt'] = i
				list_value.append('Net_Wt')
			if i == '10.0':
				barcode_dict['Net_Wt'] = i 
				list_value.append('Net_Wt')
			if len(i) == 10 and i[0] == '0':
				barcode_dict['Batch_ID'] = i 
				list_value.append('Batch_ID')
			if len(i) == 6:
				barcode_dict['Description'] = i 
				list_value.append('Description') 
		for item in list_check:
			if item not in list_value:
				barcode_dict[item] = 'None'
			else:
				pass                            
		return barcode_dict

	def cord_decoder(self,img):
		list_str = []
		result_list = []
		results = zxingcpp.read_barcodes(img)
		for result in results:
			value = format(result.position)
			print(value)
			value = str(value).replace(' ', ',')
			value = value.split(",")  
			value = [i.split('|', 9) for i in value]
			for i in value:
				value = str(i)[1:-1]
				result = ast.literal_eval(value)
				list_str.append(result)
		for i in list_str:
			if type(i) is tuple:
				for k in i:
					result_list.append(k)
			else:
				result_list.append(i) 
		result_list = [*set(result_list)]
		if len(results) == 0:
			print("Could not find any barcode.")
		return result_list  
	
	def barcode_value_decoder(self,img):
		value = self.barcode_decoder(img)
		barcode_dict = self.convert_list_order(value)
		return barcode_dict

	def gradient(self,pt1,pt2):
		return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])

	def measurment(self,mask,img):
		measurment_list = []
		a1 = []
		a2 = []
		try:
			gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
			edges = cv2.Canny(gray, 50, 200, apertureSize=3)
			contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			for cnt in contours:
				(x,y,w,h) = cv2.boundingRect(cnt)
				print(x+w,y+h)
				print((x,y), (x+w,y))
				pt1 = [x,y]
				pt2 = [x+w,y]
				rect = cv2.minAreaRect(cnt)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				list_1 = box.tolist()
				list_1 = sorted(list_1 , key=lambda k: [k[1], k[0]])    
				pt3 = list_1[0]
				pt4 = list_1[1]
				p = int(pt1[1]) - int(pt3[1])
				p1 = int(pt2[1]) - int(pt4[1])
				if p1 < 0:
					cv2.line(img, tuple(pt1), tuple(pt2), (0,250,0), 2)
					cv2.line(img, tuple(pt1), tuple(pt4), (0,0,250), 2)
					cv2.line(mask, tuple(pt1), tuple(pt2), (0,250,0), 2)
					cv2.line(mask, tuple(pt1), tuple(pt4), (0,0,250), 2)
					m1 = self.gradient(pt1,pt2)
					a1.append(m1)
					m2 = self.gradient(pt3,pt4)
					a2.append(m2)
				elif p < 0:
					cv2.line(img, tuple(pt2), tuple(pt1), (0,250,0), 2)
					cv2.line(img, tuple(pt2), tuple(pt3), (0,0,250), 2)
					cv2.line(mask, tuple(pt2), tuple(pt1), (0,250,0), 2)
					cv2.line(mask, tuple(pt2), tuple(pt3), (0,0,250), 2)
					m1 = self.gradient(pt3,pt4)
					a2.append(m1)
					m2 = self.gradient(pt1,pt2)
					a1.append(m2)
				for m1,m2 in zip(a1,a2):	
					angR = math.atan((m2-m1)/(1+(m2*m1)))
					angD = round(math.degrees(angR))
					measurment_list.append(angD)
					cv2.putText(img,str(angD)+ "_degree",(pt1[0]-40,pt1[1]-20),cv2.FONT_HERSHEY_COMPLEX,
								1,(0,0,255),2) 
					cv2.putText(mask,str(angD)+ "_degree",(pt1[0]-40,pt1[1]-20),cv2.FONT_HERSHEY_COMPLEX,
								1,(0,0,255),2)   
		except ZeroDivisionError:
			measurment_list.append('No_Detection')
			print('No_Detection')   
		return mask ,img, measurment_list , list_1

	def midpoint(self,ptA, ptB):
		return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

	def centroid_to_label(self,mask,img,tl ,tr):
		(tltrX, tltrY) = self.midpoint(tl[0], tl[1])
		(tltrz, tltrh) = self.midpoint(tr[0], tr[1])
		dA = dist.euclidean((tltrX, tltrY), (tltrz, tltrh))
		refObj = (dA / 1920)
		refObj = refObj + 0.9
		refObj = refObj*2.45
		measure = "{}".format(refObj)
		cv2.circle(mask, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		cv2.circle(mask, (int(tltrz), int(tltrh)), 5, (255, 0, 0), -1)
		cv2.line(mask, (int(tltrX), int(tltrY)), (int(tltrz), int(tltrh)),
		(255, 0, 255), 2)
		cv2.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		cv2.circle(img, (int(tltrz), int(tltrh)), 5, (255, 0, 0), -1)
		cv2.line(img, (int(tltrX), int(tltrY)), (int(tltrz), int(tltrh)),
		(255, 0, 255), 2)
		(mX, mY) = self.midpoint((tltrX, tltrY), (tltrz, tltrh))
		cv2.putText(mask, "{}_cm".format(refObj), (int(mX), int(mY - 10)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 159), 2)
		cv2.putText(img, "{}_cm".format(refObj), (int(mX), int(mY - 10)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 159), 2)
		if(len(measure) == 0):
			measure = 'None'
		else:
			pass    
		return mask, img , measure 

	# def crop_image(self,mask,img):
	#     gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	#     edges = cv2.Canny(gray, 50, 200, apertureSize=3)
	#     contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#     for cnt in contours:
	#         (x,y,w,h) = cv2.boundingRect(cnt)
	#         img = img[y:y+h, x:x+w]
	#         return img

	# def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
	#     h_min = min(im.shape[0] for im in im_list)
	#     im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
	#                     for im in im_list]
	#     return cv2.hconcat(im_list_resize)

	   