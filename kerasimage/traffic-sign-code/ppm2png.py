## translate ppm image to png image

import cv2
import os

ORIGINAL_TRAIN_PATH = '../data/Training'
ORIGINAL_TEST_PATH = '../data/Testing'

DST_TRAIN_PATH = '../data/traffic-sign/train/'
DST_TEST_PATH = '../data/traffic-sign/test/'

if not os.path.isdir(DST_TRAIN_PATH):
    os.mkdir(DST_TRAIN_PATH)

for train_class in os.listdir(ORIGINAL_TRAIN_PATH):
    if train_class == '.DS_Store':
        continue;
    if not os.path.isdir( DST_TRAIN_PATH + train_class):
        os.mkdir(DST_TRAIN_PATH+train_class)
    for pic in os.listdir(ORIGINAL_TRAIN_PATH + '/'+ train_class):
        if not (pic.split('.')[1] == 'ppm'):
            continue
        im = cv2.imread(ORIGINAL_TRAIN_PATH + '/' +train_class+'/'+ pic) 
        name = pic.split('.')[0]
        new_name = name+'.png'
        print (new_name)
        cv2.imwrite(DST_TRAIN_PATH + train_class + '/'+ new_name,im)
        
if not os.path.isdir(DST_TEST_PATH):
    os.mkdir(DST_TEST_PATH)

for test_class in os.listdir(ORIGINAL_TEST_PATH):
    if test_class == '.DS_Store':
        continue;
    if not os.path.isdir(DST_TEST_PATH+test_class):
        os.mkdir(DST_TEST_PATH+test_class)
    for pic in os.listdir(ORIGINAL_TEST_PATH + '/'+ test_class):
        if not (pic.split('.')[1] == 'ppm'):
            continue
        im = cv2.imread(ORIGINAL_TEST_PATH + '/' +test_class+'/'+ pic) 
        name = pic.split('.')[0]
        new_name = name+'.png'
        print (new_name)
        cv2.imwrite(DST_TEST_PATH + test_class + '/'+ new_name,im)
