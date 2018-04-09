from django.shortcuts import render

# Create your views here.
from django.views.decorators.csrf import csrf_exempt
from scipy.spatial.distance import *
from django.http import JsonResponse
import numpy as np
import urllib.request
import json
import cv2 as cv,cv2
import os
import numpy as np
import math
import pandas as pd
import pickle
import io
import glob
import threading
from darkflow.net.build import TFNet 
import math
import imutils

# define the path to the face detector
'''fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))'''


#global pt
class Obstacles:
    def assign(self,label=None,confidence=None,topleft_x=None,topleft_y=None,bottomright_x=None,bottomright_y=None):
        
        self.label=label
        self.confidence=confidence
        self.topleft_x=topleft_x
        self.topleft_y=topleft_y
        self.bottomright_x=bottomright_x
        self.bottomright_y=bottomright_y
        #self.middle_x=int((bottomright_x+topleft_x)/2)
        self.middle_x=calc_middle(bottomright_x,topleft_x)
        #self.middle_y=int((bottomright_y+topleft_x)/2)
        self.middle_y=calc_middle(bottomright_y,topleft_y)
        self.left=[topleft_x,bottomright_y]
        self.right=[bottomright_x,bottomright_y]


        
        
    
    def assign_dist(self,distance=None):
        self.distance=distance

    '''def assign_offset(self):
        obj_offset=lane_center - self.middle_x
        if(obj_offset<0):
            self.offset="L"'''


    def contents(self):
        print("label",self.label,"\nconfidence",self.confidence,"\ntopleft",self.topleft_x,"topleft_y",self.topleft_y,
            "bottomright_x",self.bottomright_x,"bottomright_y",self.bottomright_y,"middle_x",self.middle_x,"middle_y",self.middle_y,"distance",self.distance)

    
def calc_middle(x,y):
    return int((x+y)/2)




    
def calc_distance(x,y):
    a=np.array([x,y])
    return euclidean(pt,a)



def find_nearest_obstacle(lst):
    try:
        obj=[]
        for l in lst:
            obj.append(l.__dict__)
        from operator import itemgetter
        nearest_obstacle=sorted(obj,key=itemgetter('distance'))
    except Exception as e:
        print(e)
        return None
    return nearest_obstacle[0]

def getOffsetStatus():
    if  offset <-0.5 :
        return "Getting off the lane on Right side "
    elif offset > 0.5 :
        return "Getting off the line on left side"
    else:
        return "Inside the lane "


def findObstaclesInPath(nearest_obstacle):
    print(nearest_obstacle)
    print(nearest_obstacle['middle_x'])
    print(nearest_obstacle['middle_y'])
    print()
    print("nearest_obstacle",nearest_obstacle['middle_x'])
    if(int(nearest_obstacle['middle_x']) in range(int(left_low),int(right_low)+1)) and int(nearest_obstacle['distance'])<=300 :
        return "Obstacle in your path at a distance of " + str(int(nearest_obstacle['distance']))+"   "
    else:
        return "No Nearest Obstacle is in your path"



    






points_pickle = pickle.load( open( "object_and_image_points.pkl", "rb" ) )
chess_points = points_pickle["chesspoints"]
image_points = points_pickle["imagepoints"]
img_size = points_pickle["imagesize"]
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(chess_points, image_points, img_size, None, None)
camera = pickle.load(open( "camera_matrix.pkl", "rb" ))
mtx = camera['mtx']
dist = camera['dist']
camera_img_size = camera['imagesize']


options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}


tfnet = TFNet(options)

#global first_frame
@csrf_exempt

def detect(request):
    s=str(request.method)
    fil=[]
    print("request files",request.FILES)
    for x in request.FILES:
        fil.append(str(request.FILES[x]))
    for y in fil:
        
        
        from time import time
        t1=time()
        im=cv.imread(y)

        #im=imutils.rotate(im,-90)
        #im=im[im.shape[1]//2:,:]
        obj,lst=object_detect(im)
        tobj=obj
        tobj=imutils.rotate(tobj,-90)


        print("time for object detection",time()-t1)
        
        img=lanedetect(tobj)
        #img=imutils.rotate(img,90)

        
        for l in lst:
            l.assign_dist(calc_distance(l.middle_x,l.middle_y))
            #l.assign_offset()

            

        
        
        try:
            nearest_obstacle=find_nearest_obstacle(lst)
            
        except Exception as e:
            print("In calling function")
            print(e)
        
        try:
            cv2.line(obj,(int(pt[0]),int(pt[1])),(nearest_obstacle['middle_x'],nearest_obstacle['middle_y']),5)
            
        except Exception as e:
            print(e)

        






        cv2.imshow("image",obj)
        cv2.waitKey(0)
    data = {"success": False,"hiii":"hello","method":s,"File":fil}
 

    
    data['offset']=getOffsetStatus()
    data['obstacles']=findObstaclesInPath(nearest_obstacle)
    
    
    return JsonResponse(data)



def object_detect(frame=None):
    result = tfnet.return_predict(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lst = [Obstacles() for i in range(len(result))]
    count=0
    for results in result:
        
        lst[count].assign(results['label'],results['confidence'],results['topleft']['x'],results['topleft']['y'],results['bottomright']['x'],results['bottomright']['y'])
        
        cv2.rectangle(frame, (results['topleft']['x'], results['topleft']['y']), (results['bottomright']['x'], results['bottomright']['y']),(255,0,0), 2)
        cv2.putText(frame,str(results['label']),(results['topleft']['x'], results['topleft']['y']), font, 1, (0,0,0), 1, cv2.LINE_AA)
        count=count+1
    return frame,lst



def lanedetect(img):
    global curve_radius
    global offset
    undist = distort_correct(img,mtx,dist,camera_img_size)
    #undist=imutils.rotate(undist,-180)
    # get binary image
    binary_img = binary_pipeline(undist)
    #binary_img=imutils.rotate(binary_img,-90)
    #perspective transform
    birdseye, inverse_perspective_transform = warp_image(binary_img)
    #cv2.imshow("birdseye",birdseye)
    #cv2.waitKey(0)
    left_fit,right_fit = track_lanes_initialize(birdseye)
    
    
    
    global pt
    #draw polygon
    processed_frame , temp= lane_fill_poly(birdseye, undist, left_fit, right_fit,inverse_perspective_transform)
    curve_radius = measure_curve(birdseye,left_fit,right_fit)
    offset,pt = vehicle_offset(undist, left_fit, right_fit)
    
    d=[]
    for t in temp:
        for u in t:
            d=t[0]
            break;

    
    try:
        mini=np.amin(d,axis=0)
        maxi=np.amax(d,axis=0)
    except:
        print("\n error finding min and max")
    cv2.rectangle(processed_frame, (mini[0]-100,mini[1]), (maxi[0]-100,maxi[1]),(255,0,0), 2)
    #for x in d:
     #   cv2.circle(processed_frame,(x[0],x[1]), 10, (255,0,0), -1)

        
    #printing information to frame
    font = cv.FONT_HERSHEY_TRIPLEX
    processed_frame=imutils.rotate(processed_frame,90)
    processed_frame = cv.putText(processed_frame, 'Radius: '+str(curve_radius)+' m', (30, 40), font, 1, (0,255,0), 2)
    processed_frame = cv.putText(processed_frame, 'Offset: '+str(offset)+' m', (30, 80), font, 1, (0,255,0), 2)
    
   
    return processed_frame


def track_lanes_initialize(binary_warped):
    
    global window_search
    
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # we need max for each half of the histogram. the example above shows how
    # things could be complicated if didn't split the image in half 
    # before taking the top 2 maxes
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    # this will throw an error in the height if it doesn't evenly divide the img height
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    global win_left,win_right
    win_left=win_right=[]
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
        win_y_high = int(binary_warped.shape[0] - window*window_height)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3) 
        #cv2.imshow("boxes",out_img)
        #cv2.waitKey(0)
        
        win_left.append([int((win_xleft_low+win_xleft_high)/2),int((win_y_low+win_y_high)/2)])
        cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3) 
        win_right.append([int((win_xright_low+win_xright_high)/2),int((win_y_low+win_y_high)/2)])
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)


    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit,right_fit





def distort_correct(img,mtx,dist,camera_img_size):
    img=cv2.resize(img,(1280,720))
    img_size1 = (img.shape[1],img.shape[0])


    
    assert (img_size1 == camera_img_size),'image size is not compatible'
    undist = cv.undistort(img, mtx, dist, None, mtx)
    return undist

def abs_sobel_thresh(img, orient='x', thresh=(0,255)):
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

def mag_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the xy magnitude 
    mag = np.sqrt(x**2 + y**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale = np.max(mag)/255
    eightbit = (mag/scale).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(eightbit)
    binary_output[(eightbit > thresh[0]) & (eightbit < thresh[1])] =1 
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 1) Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    x = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel))
    y = np.absolute(cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel))
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direction = np.arctan2(y, x)
    binary_output = np.zeros_like(direction)
    binary_output[(direction > thresh[0]) & (direction < thresh[1])] = 1
    return binary_output


def hls_select(img, sthresh=(0, 255),lthresh=()):
    # 1) Convert to HLS color space
    hls_img = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    L = hls_img[:,:,1]
    S = hls_img[:,:,2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(S)
    binary_output[(S >= sthresh[0]) & (S <= sthresh[1])
                 & (L > lthresh[0]) & (L <= lthresh[1])] = 1
    return binary_output

def red_select(img, thresh=(0, 255)):
    # Apply a threshold to the R channel
    R = img[:,:,0]
    # Return a binary image of threshold result
    binary_output = np.zeros_like(R)
    binary_output[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary_output


def binary_pipeline(img):
    
    img_copy = cv.GaussianBlur(img, (3, 3), 0)
    #img_copy = np.copy(img)
    
    # color channels
    s_binary = hls_select(img_copy, sthresh=(140, 255), lthresh=(120, 255))
    #red_binary = red_select(img_copy, thresh=(200,255))
    
    # Sobel x
    x_binary = abs_sobel_thresh(img_copy,thresh=(25, 200))
    y_binary = abs_sobel_thresh(img_copy,thresh=(25, 200), orient='y')
    xy = cv.bitwise_and(x_binary, y_binary)
    
    #magnitude & direction
    mag_binary = mag_threshold(img_copy, sobel_kernel=3, thresh=(30,100))
    dir_binary = dir_threshold(img_copy, sobel_kernel=3, thresh=(0.8, 1.2))
    
    # Stack each channel
    gradient = np.zeros_like(s_binary)
    gradient[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    final_binary = cv.bitwise_or(s_binary, gradient)
    
    return final_binary


def warp_image(img):
    
    image_size = (img.shape[1], img.shape[0])
    x = img.shape[1]
    y = img.shape[0]
    
    #source_points = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]],[x[2], y[2]], [x[3], y[3]]]) / 4)

    #destination_points = np.floor(np.float32([[X[0], Y[0]], [X[1], Y[1]],[X[2], Y[2]], [X[3], Y[3]]]) / 4)

    #the "order" of points in the polygon you are defining does not matter
    #but they need to match the corresponding points in destination_points!
    '''

    source_points = np.float32([
    [0.117 * x, y],
    [(0.5 * x) - (x*0.078), (2/3)*y],
    [(0.5 * x) + (x*0.078), (2/3)*y],
    [x - (0.117 * x), y]
    ])

      #chicago footage
    x = [ 0, 1280,1280,0]
    y = [ 520,520,0,0]
    X = [ int(0.25*1280),int(0.25*1280),1280-int(0.25*1280),1280-int(0.25*1280)]
    Y = [ 720,0,0,720]
    #X = [150, 1100, 0, 1500,]
    #Y = [720, 720, 0, 0]

    source_points = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]],[x[2], y[2]], [x[3], y[3]]]) / 4)

    destination_points = np.floor(np.float32([[X[0], Y[0]], [X[1], Y[1]],[X[2], Y[2]], [X[3], Y[3]]]) / 4)
    '''
    source_points = np.float32([
                 [0+100, y-100],
                 [x-100, y-100],
                 [x-100, 0+100],
                 [0+100, 0+100]
                 ])
    
    destination_points = np.float32([
    [0.25 * x, y],
    [0.25 * x, 0],
    [x - (0.25 * x), 0],
    [x - (0.25 * x), y]
    ])
    '''
 destination_points = np.float32([
                 [200, 720],
                 [200, 200],
                 [1000, 200],
                 [1000, 720]
                 ])
  
    '''
        
    #cv2.line(img,(int(pt[0]),int(pt[1])),(nearest_obstacle['middle_x'],nearest_obstacle['middle_y']),5)
    
    perspectivce_transform = cv.getPerspectiveTransform(source_points, destination_points)
    
    #cv2.imshow("SRC",t)
    #cv2.waitKey(0)
    inverse_perspective_transform = cv.getPerspectiveTransform( destination_points, source_points)
    warped_img = cv.warpPerspective(img,perspectivce_transform, image_size, flags=cv.INTER_LINEAR)

    #cv2.imshow("warpPerspective",warped_img)
    #cv2.waitKey(0)
    return warped_img, inverse_perspective_transform


def track_lanes_update(binary_warped, left_fit,right_fit):

    global window_search
    global frame_count
    
    
    if frame_count % 10 == 0:
        window_search=True
   
        
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    return left_fit,right_fit,leftx,lefty,rightx,righty

def get_val(y,poly_coeff):
    return poly_coeff[0]*y**2+poly_coeff[1]*y+poly_coeff[2]

def lane_fill_poly(binary_warped,undist,left_fit,right_fit,inverse_perspective_transform):
    
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = get_val(ploty,left_fit)
    right_fitx = get_val(ploty,right_fit)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast x and y for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane 
    
    
    y=np.int_([pts])



    cv.fillPoly(color_warp, np.int_([pts]), (0,255,0))
    
    

    # Warp using inverse perspective transform
    newwarp = cv.warpPerspective(color_warp, inverse_perspective_transform, (binary_warped.shape[1], binary_warped.shape[0])) 
    # overlay
    #newwarp = cv.cvtColor(newwarp, cv.COLOR_BGR2RGB)
    result = cv.addWeighted(undist, 1, newwarp, 0.3, 0)
        
    return result,y


def measure_curve(binary_warped,left_fit,right_fit):
        
    # generate y values 
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    # measure radius at the maximum y value, or bottom of the image
    # this is closest to the car 
    y_eval = np.max(ploty)
    
    # coversion rates for pixels to metric
    # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
    # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
   
    # x positions lanes
    leftx = get_val(ploty,left_fit)
    rightx = get_val(ploty,right_fit)

    # fit polynomials in metric 
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # calculate radii in metric from radius of curvature formula
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # averaged radius of curvature of left and right in real world space
    # should represent approximately the center of the road
    curve_rad = round((left_curverad + right_curverad)/2)
    
    return curve_rad


def vehicle_offset(img,left_fit,right_fit):
    
    # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
    # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
    global xm_per_pix
    xm_per_pix = 3.7/700 
    image_center = img.shape[1]/2
    global left_low
    global right_low
    
    ## find where lines hit the bottom of the image, closest to the car
    left_low = get_val(img.shape[0],left_fit)
    right_low = get_val(img.shape[0],right_fit)
    print("Left low",left_low,"\nRight low",right_low)
    print()
    
    
    # pixel coordinate for center of lane
    lane_center = (left_low+right_low)/2.0
    
    ## vehicle offset
    #cv2.imshow("Image center",img)
    #cv2.waitKey(0)
    distance = image_center - lane_center
    
    ## convert to metric
    return (round(distance*xm_per_pix,5)),[image_center,lane_center]

