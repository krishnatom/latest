# import the necessary packages
import requests
import urllib
import cv2
import json

# define the URL to our face detection API
url = "http://localhost:8000/detect/"

# use our face detection API to find faces in images via image URL
'''
cap = cv2.VideoCapture('project_video.mp4')
ret=True
count=0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(type(frame),ret)
    if not ret:
    	break
    cv2.imwrite("frame.jpg",frame)
    files = {'media': open("frame.jpg", 'rb')}
    r = requests.post(url, files=files).json()
    count=count+1
    print(count)
'''
for i in range(1,8):

	files = {'media': open("test%d.jpg"%i, 'rb')}
	r = requests.post(url, files=files).json()




    # Our operations on the frame come here
    

    # Display the resulting frame
    
    


print(str(r))
print("{}".format(r))
'''
# loop over the faces and draw them on the image
for (startX, startY, endX, endY) in r["faces"]:
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# show the output image
cv2.imshow("obama.jpg", image)
cv2.waitKey(0)

# load our image and now use the face detection API to find faces in
# images by uploading an image directly
image = cv2.imread("adrian.jpg")
payload = {"image": open("adrian.jpg", "rb")}
r = requests.post(url, files=payload).json()
print "adrian.jpg: {}".format(r)

# loop over the faces and draw them on the image
for (startX, startY, endX, endY) in r["faces"]:
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# show the output image
cv2.imshow("adrian.jpg", image)
cv2.waitKey(0)'''