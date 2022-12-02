import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
from glob import glob
import numpy as np
import imutils

def load_images(directory):
    images = []

    w = 500
    h = 300

    for filepath in tqdm(os.listdir(directory)[:4]):
        if filepath[0] != ".":
          img_path = os.path.join(directory,filepath)
          img = cv2.imread(img_path)
          # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          img = cv2.resize(img,(w, h))
          # img = cv2.medianBlur(img,5)
          images.append(img)
    
    images = np.array(images) #.reshape(len(images),w,h,3)
    return images

images = load_images("./../test")

img = images[2]

cv2.imshow('Sample', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 13, 15, 15)

edged = cv2.Canny(gray, 30, 200) 
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in contours:
    
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
 
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print ("No contour detected")
else:
     detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

ret, thresh1 = cv2.threshold(Cropped, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

def bwareaopen(img, min_size, connectivity=8):
        """Remove small objects from binary image (approximation of 
        bwareaopen in Matlab for 2D images).
    
        Args:
            img: a binary image (dtype=uint8) to remove small objects from
            min_size: minimum size (in pixels) for an object to remain in the image
            connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).
    
        Returns:
            the binary image with small objects removed
        """

        img = img.copy()
    
        # Find all connected components (called here "labels")
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            img, connectivity=connectivity)
        
        # check size of all connected components (area in pixels)
        for i in range(num_labels):
            label_size = stats[i, cv2.CC_STAT_AREA]
            
            # remove connected components smaller than min_size
            if label_size < min_size:
                img[labels == i] = 0
                
        return img

if thresh1[0][0] >= 0:
    thresh1 = 255-thresh1

picture = bwareaopen(thresh1, 100)

cv2.imshow('Box', picture)

# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()