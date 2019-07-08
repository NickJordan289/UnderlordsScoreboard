import cv2
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer

img_rgb = cv2.imread('5ftquof7qu831.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('drow.png', 0)
w, h = template.shape[::-1]
# 10 extra pixels added to bottom, this is to grab the stars (10 pixels should actually be a percentage based on the screen grab resolution)
h = h + 10

def run():
    start = timer()
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    #print('Found:', len(loc[0]))    
    boxes = []
    np_images = []
    for pt in zip(*loc[::-1]):
        bb = {}
        bb['x1'] = x = pt[0]
        bb['x2'] = x2 = pt[0] + w
        bb['y1'] = y = pt[1]
        bb['y2'] = y2 = pt[1] + h
        crop = img_rgb[y:y2, x:x2]

        if len(boxes) > 0:
            found_overlap = False
            for box in boxes:
                overlap_percent = get_iou(box, bb)
                #print(overlap_percent)
                if(overlap_percent > 0):
                    found_overlap = True
            if not found_overlap:
                boxes.append(bb)
                np_images.append(crop)
        else:
            boxes.append(bb)
            np_images.append(crop)
        #cv2.rectangle(img_rgb, pt, (x2, y2), (0, 0, 255), 2)

        #im = img_rgb[pt[1]:pt[1]+h, pt[0]:pt[0]+w].copy()
        #cv2.imshow('Img', im)
        #cv2.waitKey(0)

    for img in np_images:
        #print(img)
        pass

    end = timer()
    #print(f'Took: {end-start} seconds')
    #print('Reduced:', len(boxes))
    #cv2.imwrite('res.png', img_rgb)

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


if __name__ == '__main__':
    start = timer()
    for i in range(60):
        run()
    end = timer()
    print(f'Took: {end-start} seconds')