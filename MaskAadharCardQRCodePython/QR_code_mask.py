#!/usr/bin/env python
# coding: utf-8



import cv2
import os
import numpy as np
import sys



def QR_code_mask(input_path, input_file_name, output_path, output_path_for_exceptions, output_file_name):
    
    #image reading and resizing
    image=cv2.imread(input_path + "/" + input_file_name)
                        # if resizing is not necessary comment the following 3 lines
    dim=image.shape
    new_dim= (round(dim[0]*0.5), round(dim[1]*0.5))  #resize by 50%
    image= cv2.resize(image, new_dim)
    
    half_width = 2*np.int(image.shape[1]/3)    #for right half of the document since QRcode is mostly at that side
    
    #filter, threshold and extract contour
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   #converting to grayscale
    kernel = np.ones((7,7),np.float32)/49    #(7x7) kernal for filtering
    gray_img = cv2.filter2D(gray_img,-1,kernel)     # filter out small lines 
    ret,thresh = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY_INV)    # threshold the image
    dilated_img = cv2.dilate(thresh,kernel,iterations = 1)     #dilate for more pronounced shapes
    contours,_ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # extracting the contours
    
    # drawing only those rectangles which are atleast half the area of maximum area rectangle,
    # because we see that the biggest possible rectangle is the QR code. 
    #req_area is the minimum threshold area for a rectangle to be considered
    mask_count=0
    rectangles=[]  #contains all the rectangle approximations of the contours
    
    for each_contour in contours:  #identifying rectangles
        polygon_approx= cv2.approxPolyDP(each_contour,0.09*cv2.arcLength(each_contour,True),True)
        if len(polygon_approx)==4: 
            rectangles.append(polygon_approx)
            
    req_area=0.5*cv2.contourArea(max(rectangles, key = cv2.contourArea))
    
    for each_rect in rectangles: #drawing the required rectangles
        if cv2.contourArea(each_rect)>req_area and each_rect[0][0][0]>half_width:
            cv2.drawContours(image,[each_rect],0,0,-1)
            mask_count+=1
    
    #saving the final image
    if mask_count<3 and mask_count>0:
        os.chdir(output_path)
        cv2.imwrite(output_file_name, image)
    else:
        os.chdir(output_path_for_exceptions)
        cv2.imwrite(output_file_name, image)
    


    

if __name__ == "__main__":
    #for x in sys.argv:
     #print("Argument: ", x)
    QR_code_mask(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])


