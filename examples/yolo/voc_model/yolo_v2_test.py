import caffe
caffe.set_mode_cpu()
GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)
import sys, getopt
import os
from datetime import datetime
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from operator import itemgetter
import re

def sigmoid(x):
  y = np.exp(x)
  return y/(1+y)

def interpret_output(output):
  bias = [0.08307, 0.09153, 0.26307, 0.33923, 0.51, 0.87538461538, 0.72461538461, 0.39307692307, 1, 0.80923076923]

  OBJECT_NUM_ = 5
  CLASSES_ = 20
  SIDE_ = 13
  GRIDS_ = SIDE_ * SIDE_ 
  BOX_SIZE_ = (5+CLASSES_)*GRIDS_
  CONFIDENCE_THD = 0.4
  IOU_THD = 0

  if not (len(output) == OBJECT_NUM_ * (CLASSES_ + 5) * GRIDS_):
    print ('output size: {} does not match expected value.'.format(len(output)))
    return None

  results = []
  #check result for each grid
  for y_idx in range(SIDE_):
    for x_idx in range(SIDE_):
      records = []

      for o_idx in range(OBJECT_NUM_):
        b_idx = y_idx*SIDE_ + x_idx + o_idx*BOX_SIZE_
        #x
        x = (sigmoid(output[b_idx + 0*GRIDS_]) + x_idx)/SIDE_
        #y
        y = (sigmoid(output[b_idx + 1*GRIDS_]) + y_idx)/SIDE_
        #w
        w = bias[2*o_idx + 0]*np.exp(output[b_idx + 2*GRIDS_])
        #h
        h = bias[2*o_idx + 1]*np.exp(output[b_idx + 3*GRIDS_])
        #confidence
        c = output[b_idx + 4*GRIDS_]

        pred_class = 0
        best_score = 0
        for c_idx in range(CLASSES_):
          if output[b_idx+(5+c_idx)*GRIDS_] > best_score:
            best_score = output[b_idx+(5+c_idx)*GRIDS_]
            pred_class = c_idx
        l = pred_class
        records.append({"t": 1, "l" : l, "c" : c, "x" : x, "y" : y, "w" : w, "h" : h});

      sorted_records = sorted(records, key=itemgetter('c'), reverse=True)
      #print('\nSorted grid record:')
      #for record in sorted_records:
        #print '{', record["l"], record["c"], record["x"], record["y"], record["w"], record["h"], '}';

      flag = np.zeros(OBJECT_NUM_)
      for i in range(OBJECT_NUM_):
        if (sorted_records[i]['c'] < CONFIDENCE_THD):
          break
        if (flag[i] == 1):
          continue

        #compare this one with others to eliminate duplicates if overlap
        for j in range(i+1,OBJECT_NUM_):
          if (flag[j]):
            continue
          if (sorted_records[j]['c'] < CONFIDENCE_THD):
            break
          iou = getTwoBoxArea(sorted_records[i], sorted_records[j])
          if (iou > IOU_THD):
            flag[j] = True
            continue

        #add to final results
        results.append(sorted_records[i])

  #for record in results:
    #print '{', record["l"], record["c"], record["x"], record["y"], record["w"], record["h"], '}';

  return results
                     
def overlap(x1, w1, x2, w2):
  left = max(x1-w1/2, x2-w2/2)
  right = min(x1+w1/2, x2+w2/2)
  return (right-left)

def getTwoBoxArea(box1, box2):
  width = overlap(box1['x'], box1['w'], box2['x'], box2['w'])
  height = overlap(box1['y'], box1['h'], box2['y'], box2['h'])
  if (width <= 0) or (height <=0):
    return 0
  else:
    print 'Boxes overlapped'
    area = width*height
    area1 = box1['w']*box1['h']
    area2 = box2['w']*box2['h']
    return (area/min(area1, area2))

#analyze the final results to eliminate redundent boxes
#
# rule: 1) boxes identify the same class of objects
#       2) boxes are in neighboring grid and center distance less than 1/2 grid
#       3) the jacard overlap is sufficiently large
# result:
#       select only the box with highest confidence
#
def box_NMS(results):
  SIDE_ = 13
  obj_cnt = len(results)
  for i in range(obj_cnt):
    if results[i]['t'] == 0:
      continue
    for j in range(i+1, obj_cnt):
      if results[j]['t'] == 0:
        continue
      if not (int(results[i]['l']) == int(results[j]['l'])):
        continue
      if abs(results[i]['x'] - results[j]['x']) > (416/SIDE_/2):
        continue
      if abs(results[i]['y'] - results[j]['y']) > (416/SIDE_/2):
        continue
      iou = getTwoBoxArea(results[i], results[j])
      if (iou > 0.5):
        if (results[i]['c'] > results[j]['c']):
          results[j]['t'] = 0
        else:
          results[i]['t'] = 0

  return results
   
def show_results(img, results, img_width, img_height, save_img, img_filename):
  classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
  img_cp = img.copy()
  imshow = True
  
  print ('Final detected objects:')
  for record in results:
    if record['t'] == 0:
      continue
    c = record['c'] 
    l = int(record['l']) 
    x = int(record['x']*img_width)
    y = int(record['y']*img_height) 
    w = int(record['w']*img_width) 
    h = int(record['h']*img_height)
 
    xmin = x - w/2
    if xmin < 0:
       xmin = 0
    xmax = x + w/2
    if xmax > img_width:
       xmax = img_width

    ymin = y - h/2
    if ymin < 0:
       ymin = 0
    ymax = y + h/2
    if ymax > img_height:
       ymax = img_height

    name = classes[l]  
    print ('Detected object: {}, {}, {}, {}, {} @ {}'.format(name, xmin, ymin, xmax, ymax, c))

    if imshow:
      cv2.rectangle(img_cp,(xmin,ymin),(xmax,ymax),(0,255,0),2)
      if (ymin - 20) > 0:
        cv2.rectangle(img_cp,(xmin,ymin-20),(xmin+160,ymin),(125,125,125),-1)
        cv2.putText(img_cp, name + ' : %.2f' % c,(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)  
      else:
        cv2.rectangle(img_cp,(xmin,ymin),(xmin+160,ymin+20),(125,125,125),-1)
        cv2.putText(img_cp, name + ' : %.2f' % c,(xmin+5,ymin+13),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)  

  #save result image to file
  if save_img:
    directory = './result_imgs/'
    if not os.path.exists(directory):
      os.makedirs(directory)

    out_filename = os.path.basename(img_filename)
    out_filename = directory + 'res_' + out_filename
    print out_filename
    cv2.imwrite(out_filename, img_cp)

  #display result image on screen
  if imshow :
    cv2.imshow('YOLO V2 Detection',img_cp)
    cv2.waitKey(1000)

def show_groundtruth(img, label_filename):
  img_cp = img.copy()
  imshow = True

  #parse label file to extract fields
  print ('Extract ground truth from: {}'.format(label_filename))
  tree = ET.parse(label_filename)
  root = tree.getroot()
  for object in root.findall('object'):
    name = object.find('name').text
    bndbox = object.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    print ('Ground truth object: {}, {}, {}, {}, {}'.format(name, xmin, ymin, xmax, ymax))
    cv2.rectangle(img_cp, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    if (ymin - 20) > 0:
      cv2.rectangle(img_cp,(xmin,ymin-20),(xmin+160,ymin),(125,125,125),-1)
      cv2.putText(img_cp, name, (xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)  
    else:
      cv2.rectangle(img_cp,(xmin,ymin),(xmin+160,ymin+20),(125,125,125),-1)
      cv2.putText(img_cp, name, (xmin+5,ymin+13),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)  

  if imshow :
    cv2.imshow('YOLO V2 Ground Truth',img_cp)
    cv2.waitKey(1000)        

def main(argv):
  model_filename = ''
  weight_filename = ''
  img_filename = ''
  mean_filename = ''
  label_filename = ''
  list_filename = ''
  use_mean = 0
  save_img = 0
  try:
    opts, args = getopt.getopt(argv, "hsp:w:m:i:t:l:")
    print opts
  except getopt.GetoptError:
    print 'yolo_main.py [-s] -p <model_file> -w <weight_file> [-m <mean_file>] -i <img_file> [-t <label_file>] [-l <list_file>]'
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print 'yolo_main.py [-s] -p <model_file> -w <weight_file> [-m <mean_file>] -i <img_file> [-t <label_file>] [-l <list_file>]'
      sys.exit()
    elif opt == "-s":
      save_img = 1
    elif opt == "-p":
      model_filename = arg
    elif opt == "-w":
      weight_filename = arg
    elif opt == "-m":
      mean_filename = arg
      use_mean = 1 
    elif opt == "-i":
      img_filename = arg
    elif opt == "-t":
      label_filename = arg
    elif opt == "-l":
      list_filename = arg

  print 'model file is: ', model_filename
  print 'weight file is: ', weight_filename
  print 'mean file is: ', mean_filename
  print 'list file is: ', list_filename
  
  #if use image list file, then processing all images listed
  #the list file contains <img_filename> or <img_filename label_filename> pair on each line per VOC format
  #parse to extract into a list
  if not list_filename == '':
    processing_list = []
    f = open(list_filename, "r")
    for line in f:
      if re.match('^#', line) or re.match('\s*\n', line):
        continue
      line = line.strip("\n")
      token = line.split(" ")
      if len(token) == 2:
        img_filename = token[0]
        label_filename = token[1]
      else:
        img_filename = token[0]
        label_filename = ''
      processing_list.append([img_filename, label_filename]) 
    f.close()
  else:
    processing_list = [[img_filename, label_filename]]

  for files in processing_list:
    img_filename = files[0]
    label_filename = files[1]
    print ('image file is: {}'.format(img_filename))
    print ('label file is: {}'.format(label_filename))

    #img = caffe.io.load_image(img_filename) # load the image using caffe io
    img_cv = cv2.imread(img_filename)
    img_width = img_cv.shape[1]
    img_height = img_cv.shape[0]
    print ('Input image width: {}; Input image height: {}'.format(img_width, img_height))
    if not label_filename == '':
      show_groundtruth(img_cv, label_filename)
  
    img = img_cv
    #NOTE: to match netforward.cpp, the image is kept at RGB
    #img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (416, 416), interpolation = cv2.INTER_CUBIC)
    print ('Resize image width: {}; Resize image height: {}'.format(img.shape[1], img.shape[0]))

    #subtract image mean if meanfile is provided
    if use_mean == 1:
      blob = caffe.proto.caffe_pb2.BlobProto()
      data = open(mean_filename, 'rb' ).read()
      blob.ParseFromString(data)
      img_mean = np.array( caffe.io.blobproto_to_array(blob) )
      img_mean = np.swapaxes(np.swapaxes(img_mean, 0, 3), 1, 2)
      img_mean = np.squeeze(img_mean, 3)
      img = img.astype(np.float32)
      img = img - img_mean
  
    print 'Start YOLO V2 network inference ...'
    inputs = img
    net = caffe.Net(model_filename, weight_filename, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    start = datetime.now()
    out = net.forward_all(data=np.asarray([transformer.preprocess('data', inputs)]))
    end = datetime.now()
    elapsedTime = end-start
    print 'total time is " milliseconds', elapsedTime.total_seconds()*1000
  
    results = interpret_output(out['regression'][0])
  
    if len(results) > 0:
      results = box_NMS(results)
      show_results(img_cv, results, img_width, img_height, save_img, img_filename)
      cv2.waitKey(10000)

  print 'All Done'

if __name__=='__main__':  
  main(sys.argv[1:])
