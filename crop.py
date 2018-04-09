import cv2
img=cv2.imread("l4.jpg")
print("img.shape",img.shape)
cv2.imshow("ACtual image",img)
cv2.waitKey(0)
img=img[img.shape[1]//2:,:]
cv2.imshow("Croped img",img)
cv2.waitKey(0)