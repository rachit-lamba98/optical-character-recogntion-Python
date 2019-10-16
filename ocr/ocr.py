#!/usr/bin/env python
# coding: utf-8

from PIL import Image, ImageEnhance
import pytesseract as pyTes
import cv2
import numpy as np
import sys

img_cv = cv2.imread(sys.argv[1])

img_height ,img_width, img_depth = img_cv.shape
flag = 1 if img_height < 300 and img_width < 300 else 2

#converting to greyscale
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

# applying binary threshold
ret,thresh = cv2.threshold(gray,102,255,cv2.THRESH_BINARY_INV) 

#dilating the  image
kernel = np.ones((50,50), np.uint8) 
img_dilation = cv2.dilate(thresh, kernel, iterations=1) 

#finding contours
ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

ROI = []
for i, ctr in enumerate(sorted_ctrs): 
    # Get bounding box 
    x, y, w, h = cv2.boundingRect(ctr) 
    
    # Getting ROI 
    roi = img_cv[y:y+h, x:x+w] 
    # show ROI 
#     cv2.imshow('segment no:'+str(i),roi) 
    #cv2.waitKey(0) 
    if (w > 100 and w < img_cv.shape[1]) and (h > 100 and h < img_cv.shape[0]): 
        cv2.rectangle(img_cv,(x,y),( x + w, y + h ),(0,255,0),1) 
        ROI.append(roi)

if not ROI:
    crop_img_pil = Image.fromarray(img_cv)
else:
    crop_img_pil = Image.fromarray(ROI[0])
basewidth = 500
wpercent = (basewidth/float(crop_img_pil.size[0]))
hsize = int((float(crop_img_pil.size[1])*float(wpercent)))
crop_img_pil = crop_img_pil.resize((basewidth,hsize), Image.ANTIALIAS)
display(crop_img_pil)
crop_img = np.array(crop_img_pil)
crop_img = crop_img[:, :, ::-1].copy()

crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

ret,thresh_img = cv2.threshold(crop_img_gray,102,255,cv2.THRESH_BINARY_INV) 

kernel = np.ones((5,10), np.uint8) 
img_dilation = cv2.dilate(thresh_img, kernel, iterations=1) 

ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
ROI2 = []
for i, ctr in enumerate(sorted_ctrs): 
    x, y, w, h = cv2.boundingRect(ctr) 
    roi = crop_img[y:y+h, x:x+w] 
#     cv2.imshow('segment no:'+str(i),roi) 
#     cv2.rectangle(crop_img,(x,y),( x + w, y + h ),(0,255,0),1) 
    #cv2.waitKey(0) 
    if w > 10 and h > 5: 
#          cv2.imwrite('/home/rachit/Desktop/ocr/{}.png'.format(i), roi)
        ROI2.append(roi)

text = []
for roi in ROI2:
    crop = Image.fromarray(roi)
    basewidth = 400
    wpercent = (basewidth/float(crop.size[0]))
    hsize = int((float(crop.size[1])*float(wpercent)))
    crop = crop.resize((basewidth,hsize), Image.ANTIALIAS)
    
    enhancer = ImageEnhance.Sharpness(crop)
    crop = enhancer.enhance(5.0 * flag)
#     crop = crop.convert('LA')
    display(crop)
    text.append(pyTes.image_to_string(crop))




text = list(filter(None, text))

print(text)



