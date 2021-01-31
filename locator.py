#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 16:13:50 2021

@author: emre
"""
import numpy as np
import cv2
import argparse

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('-query', type=str, default= 'Small_area.png')
parser.add_argument('-train', type=str, default= 'StarMap.png')
args = parser.parse_args()

# Reading inputs
query_img = cv2.imread(args.query)
train_img = cv2.imread(args.train)

# B&W Conversion
query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
train_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

# ORB (Oriented FAST and Rotated BRIEF) 
orb = cv2.ORB_create(12500) 

# KeyPoints & Descriptors
kp1, des1 = orb.detectAndCompute(query_gray, None) 
kp2, des2 = orb.detectAndCompute(train_gray, None)

# BruteForce matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# Matches
matches = bf.match(des1, des2, None)  
# Matches (sorted)
matches = sorted(matches, key = lambda x:x.distance)

# Array initialization for matches' points
points_q = np.zeros((len(matches), 2), dtype=np.float32)
points_t = np.zeros((len(matches), 2), dtype=np.float32)

# Filling arrays of matches' points
for i, match in enumerate(matches):
   points_q[i, :] = kp1[match.queryIdx].pt
   points_t[i, :] = kp2[match.trainIdx].pt    

# Homography matrix and mask
H, mask = cv2.findHomography(points_q, points_t, cv2.RANSAC)

# Detects location of the query image in the train image 
location = cv2.drawMatches(query_img,kp1,train_img,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)   

# Shape of the query image
height, width = query_gray.shape

# Finds corner coordinate points query image and
# transformed corner coordinates on train image 
corners = np.array([[0,0],[width-1,0],[width-1,height],[0,height-1]]).reshape(-1,1,2).astype(np.float32)
transformed_corners = cv2.perspectiveTransform(corners, np.array(H).astype(np.float32)).reshape(-1,2).astype(np.int)

# Draws lines for showing the location of the detected area
cv2.line(location, (transformed_corners[0][0]+width,transformed_corners[0][1]), (transformed_corners[1][0]+width,transformed_corners[1][1]), [255,0,255], 3) 
cv2.line(location, (transformed_corners[1][0]+width,transformed_corners[1][1]), (transformed_corners[2][0]+width,transformed_corners[2][1]), [255,0,255], 3)
cv2.line(location, (transformed_corners[2][0]+width,transformed_corners[2][1]), (transformed_corners[3][0]+width,transformed_corners[3][1]), [255,0,255], 3)
cv2.line(location, (transformed_corners[3][0]+width,transformed_corners[3][1]), (transformed_corners[0][0]+width,transformed_corners[0][1]), [255,0,255], 3)

# Saves output image
cv2.imwrite(f"loc_{args.query}",location)
print(f"\nEstimated corner coodinates of {args.query}: \n", transformed_corners)

# Do not close the image manually instead press any key in order to prevent the terminal freeze 
print(f"\nPress any key to close the image. \n(Image output can also be found in the current directory as {args.query})\n")

# shows image
cv2.imshow(args.query, location)
cv2.waitKey(0)


