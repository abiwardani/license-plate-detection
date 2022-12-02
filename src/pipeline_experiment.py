import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
from glob import glob
import numpy as np

#--- general utils ---#

def load_images(directory):
    images = []

    w = 500
    h = 300

    for filepath in tqdm(os.listdir(directory)):
        if filepath[0] != ".":
            img_path = os.path.join(directory,filepath)
            img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(w, h))
            # img = cv2.medianBlur(img,5)
            images.append(img)
    
    images = np.array(images) #.reshape(len(images),w,h,3)
    return images

def run_length_encoding(arr, n):
    # starts with 0

    rle = []
    i = 0

    zero_count = 0
    one_count = 0

    while i < n:
        if (zero_count > 0):
            rle.append(zero_count)

            zero_count = 0

        while i < n and arr[i] == 0:
            if (one_count > 0):
                if len(rle) == 0:
                    rle = [0]
                
                rle.append(one_count)

                one_count = 0

            zero_count += 1
            i += 1

        if i < n:
            one_count += 1
        i += 1
    
    if (zero_count > 0):
        rle.append(zero_count)
    
    if (one_count > 0):
        rle.append(one_count)

    return rle

def load_characters():
    directory = "../patterns"
    chars = []

    for filepath in tqdm(os.listdir(directory)):
        if filepath[0] != "." and "." in filepath:
            char_path = os.path.join(directory,filepath)
            char = cv2.imread(char_path)
            char = cv2.cvtColor(char, cv2.COLOR_RGB2GRAY)
            trimmed_char = trim_mask(char)
            chars.append(trimmed_char)

    chars = np.array(chars)
    return chars

#--- image processing utils ---#

def bwlabel(img, connectivity=8):
    img = img.copy()

    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=connectivity)

    return num_labels, labels, stats, centroids

def bwareaopen(img, min_size, connectivity=8):
    img = img.copy()

    # Find all connected components
    num_labels, labels, stats, _ = bwlabel(img)
    
    # check size of all connected components
    for i in range(num_labels):
        label_size = stats[i, cv2.CC_STAT_AREA]
        
        # remove connected components smaller than min_size
        if label_size < min_size:
            img[labels == i] = 0
            
    return img

def trim_mask(mask):
    _, _, stats, _ = bwlabel(mask)
    
    top = stats[1, cv2.CC_STAT_TOP]
    bottom = top+stats[1, cv2.CC_STAT_HEIGHT]
    left = stats[1, cv2.CC_STAT_LEFT]
    right = left+stats[1, cv2.CC_STAT_WIDTH]

    trimmed_mask = mask[top:bottom, left:right]

    return trimmed_mask

#--- image processing procedures ---#

def area_thresholding(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.bilateralFilter(img,3,75,75)

    ret, thresh1 = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # thresh1 = 255-thresh1

    area0 = bwareaopen(thresh1, 30)
    area1 = bwareaopen(thresh1, 350)
    area2 = area0-area1

    area = bwareaopen(area2, 75)

    cv2.imshow('Final', area)

    return area

def detect_plate_area(img):
    h, w = img.shape
    sums = [0 for _ in range(h)]
    counts = [0 for _ in range(h)]
    
    for i, row in enumerate(img):
        rle = run_length_encoding(row, w)
        
        counts[i] = len(rle)//2
        sums[i] = sum([val for i, val in enumerate(rle) if i%2 == 1])

    prods = [sum*count**2 for sum, count in zip(sums, counts)]
    prod_avg = sum(prods)/len(prods)
    count_avg = sum(counts)/len([i for i in counts if i != 0])

    filtered = [i for i in range(h) if prods[i] > prod_avg and counts[i] > count_avg]

    new_img = img[min(filtered)-5:max(filtered)+6]

    return new_img

def detect_plate_centroids(img):
    num_labels, labels, stats, centroids = bwlabel(img)

    h_mean = np.average(centroids[:, 0])
    v_mean = np.average(centroids[:, 1])
    h_sd = np.std(centroids[:, 0])
    v_sd = np.std(centroids[:, 1])

    d = 1.5

    filtered = [i for i in range(num_labels) if i == 0 or (abs(centroids[i][0]-h_mean) < d*h_sd and abs(centroids[i][1]-v_mean) < d*v_sd)]

    for i in range(num_labels):
        if i not in filtered:
            img[labels == i] = 0
    
    left = max(min(stats[filtered[1:], cv2.CC_STAT_LEFT])-5, 0)
    right = min(max(stats[filtered[1:], cv2.CC_STAT_LEFT]+stats[filtered[1:], cv2.CC_STAT_WIDTH])+5, len(img[0]))
    top = max(min(stats[filtered[1:], cv2.CC_STAT_TOP])-5, 0)
    bottom = min(max(stats[filtered[1:], cv2.CC_STAT_TOP]+stats[filtered[1:], cv2.CC_STAT_HEIGHT])+5, len(img))

    img = img[top:bottom, left:right]
    
    return img

def plate_to_numbers(img, templates):
    num_labels, labels, stats, centroids = bwlabel(img)

    text = []

    for i in range(1, num_labels):
        top = stats[i, cv2.CC_STAT_TOP]
        bottom = top+stats[i, cv2.CC_STAT_HEIGHT]
        left = stats[i, cv2.CC_STAT_LEFT]
        right = left+stats[i, cv2.CC_STAT_WIDTH]
        
        target = img[top:bottom, left:right]

        cv2.imshow('Target', target)
        
        for template in templates:
            target_height = stats[i, cv2.CC_STAT_HEIGHT]
            target_width = stats[i, cv2.CC_STAT_WIDTH]

            scaled_template = cv2.resize(template, (target_width, target_height))



    return text

# pipeline

images = load_images("./../test")

imuse = images[4]

cv2.imshow('Sample', imuse)

# otsu thresholding + BW area open
bin_img = area_thresholding(imuse)

# heuristic: check white frequency in each row
rough_box = detect_plate_area(bin_img)

# heuristic: connected components with |centroid - avg_centroids| < d*stdev_centroids
filtered_box = detect_plate_centroids(rough_box)

cv2.imshow('Filtered box', filtered_box)

# pattern recognition
templates = load_characters()

extracted_text = plate_to_numbers(filtered_box, templates)

# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()