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

def corr2_coeff(matA, matB):
    A_norm = matA - matA.mean(1)[:, None]
    B_norm = matB - matB.mean(1)[:, None]

    sumA = (A_norm**2).sum(1)
    sumB = (B_norm**2).sum(1)

    return np.dot(A_norm, B_norm.T)/np.sqrt(np.dot(sumA[:, None], sumB[None]))

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

    segments = []
    starts = []
    run = 0
    for i, c in enumerate(counts):
        if c == 0:
            if run > 0:
                segments.append(run)
                starts.append(i-run)
                run = 0
        else:
            run += 1
    
    segments = np.array(segments)
    starts = np.array(starts)

    h_start = starts[np.argmax(segments)]
    h_length = np.max(segments)

    new_img = img[h_start-5:h_start+h_length+6]

    return new_img

def get_plate_bounding_box(img):
    _, _, stats, _ = bwlabel(img)

    left = max(min(stats[1:, cv2.CC_STAT_LEFT])-5, 0)
    right = min(max(stats[1:, cv2.CC_STAT_LEFT]+stats[1:, cv2.CC_STAT_WIDTH])+5, len(img[0]))
    top = max(min(stats[1:, cv2.CC_STAT_TOP])-5, 0)
    bottom = min(max(stats[1:, cv2.CC_STAT_TOP]+stats[1:, cv2.CC_STAT_HEIGHT])+5, len(img))

    img = img[top:bottom, left:right]
    
    return img

def plate_to_numbers(img, templates):
    num_labels, labels, stats, centroids = bwlabel(img)
    character_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    text = []

    for i in range(1, num_labels):
        top = stats[i, cv2.CC_STAT_TOP]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        left = stats[i, cv2.CC_STAT_LEFT]
        width = stats[i, cv2.CC_STAT_WIDTH]
        
        target = img[top:top+height, left:left+width]

        scores = np.zeros(len(templates))
        
        for j, template in enumerate(templates):
            scaled_template = cv2.resize(template, (width, height))

            score = 0

            for m in range(height):
                for n in range(width):
                    if scaled_template[m][n] != 0:
                        if target[m][n] != 0:
                            score += 1
                        else:
                            score -= 1

            scores[j] = score
        
        best_score = np.max(scores)
        best_match = np.argmax(scores)

        if best_score > 0:
            text.append(character_list[best_match])

    return text

# pipeline

images = load_images("./../test")

imuse = images[4]

cv2.imshow('Sample', imuse)

# otsu thresholding + BW area open
bin_img = area_thresholding(imuse)

# heuristic: check white frequency in each row
rough_box = detect_plate_area(bin_img)

# get bounding box from min/max of connected components
filtered_box = get_plate_bounding_box(rough_box)

cv2.imshow('Filtered box', filtered_box)

# pattern recognition
templates = load_characters()

extracted_text = plate_to_numbers(filtered_box, templates)

print("".join(extracted_text))

# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()