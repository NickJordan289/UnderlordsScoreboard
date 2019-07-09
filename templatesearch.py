import cv2
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer

img_rgb = cv2.imread('sample_game/27.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
star = cv2.imread('star.png', 0)
star_w, star_h = star.shape[::-1]

drow = cv2.imread('chars/drow.png', 0)
pudge = cv2.imread('chars/pudge.png', 0)
slark = cv2.imread('chars/slark.png', 0)
templarassassin = cv2.imread('chars/templarassassin.png', 0)
tidehunter = cv2.imread('chars/tidehunter.png', 0)
venomancer = cv2.imread('chars/venomancer.png', 0)
warlock = cv2.imread('chars/warlock.png', 0)
lycan = cv2.imread('chars/lycan.png', 0)
alchemist = cv2.imread('chars/alchemist.png', 0)
necrophos = cv2.imread('chars/necrophos.png', 0)
ogre = cv2.imread('chars/ogre.png', 0)
crystalmaiden = cv2.imread('chars/crystalmaiden.png', 0)
keepermaybe = cv2.imread('chars/keepermaybe.png', 0)
queenofpain = cv2.imread('chars/queenofpain.png', 0)
tusk = cv2.imread('chars/tusk.png', 0)
sniper = cv2.imread('chars/sniper.png', 0)
axe = cv2.imread('chars/axe.png', 0)
dontknow = cv2.imread('chars/dontknow.png', 0)
dontknow2 = cv2.imread('chars/dontknow2.png', 0)
dontknow4 = cv2.imread('chars/dontknow4.png', 0)
dontknow5 = cv2.imread('chars/dontknow5.png', 0)
dontknow6 = cv2.imread('chars/dontknow6.png', 0)
dontknow9 = cv2.imread('chars/dontknow9.png', 0)
dontknow10 = cv2.imread('chars/dontknow10.png', 0)
dontknow11 = cv2.imread('chars/dontknow11.png', 0)
dontknow12 = cv2.imread('chars/dontknow12.png', 0)
dontknow13 = cv2.imread('chars/dontknow13.png', 0)
dontknow14 = cv2.imread('chars/dontknow14.png', 0)
dontknow15 = cv2.imread('chars/dontknow15.png', 0)
dontknow16 = cv2.imread('chars/dontknow16.png', 0)
dontknow17 = cv2.imread('chars/dontknow17.png', 0)
dontknow18 = cv2.imread('chars/dontknow18.png', 0)
dontknow19 = cv2.imread('chars/dontknow19.png', 0)
dontknow20 = cv2.imread('chars/dontknow20.png', 0)
dontknow21 = cv2.imread('chars/dontknow21.png', 0)
dontknow22 = cv2.imread('chars/dontknow22.png', 0)

def run(template, name):
    w, h = template.shape[::-1]
    w = w + 10
    h = h + 15

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
                if(overlap_percent > 0):
                    found_overlap = True
            if found_overlap:
                continue # proceed to next box, overlaps don't get added

        # this only runs if it is the first box or not overlapping
        boxes.append(bb)
        np_images.append(crop)
        #cv2.rectangle(img_rgb, pt, (x2, y2), (0, 0, 255), 1)

        # crop image
        im = img_rgb[y:y2, x:x2].copy()
        tier1 = [178, 157, 142]
        tier2 = [184, 198, 221]
        tier3 = [255, 251, 17]

        pix1 = im[33,11][::-1]
        dif1 = pix1-tier1
        dif_sum1 = abs(sum(dif1))
        print('Tier1:', dif_sum1)

        pix2 = im[34,18][::-1]
        dif2 = pix2-tier2
        dif_sum2 = abs(sum(dif2))
        print('Tier2:', dif_sum2)

        pix3 = im[34,17][::-1]
        if(pix3[0] > 250):
            dif_sum3 = 0 # its obviously a tier 3
        else:
            dif_sum3 = 255 # its not a tier 3
        #dif3 = pix3-tier3
        #dif_sum3 = abs(sum(dif3))
        print('Tier3:', dif_sum3)

        evaluated_tier = np.argmin([dif_sum1, dif_sum2, dif_sum3])+1
        print('Min: Tier', evaluated_tier)

        # display crop
        #cv2.imshow('Img', im)
        #cv2.waitKey(0)
        cv2.imwrite(f'{evaluated_tier}/{x}{x2}.png', im)


    #for img in np_images:
    #    #print(img)
    #    pass

    end = timer()
    #print(f'Took: {end-start} seconds')
    #print(f'{name}: {len(boxes)}')

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
    run(drow, 'Drow Ranger')
    run(pudge, 'Pudge')
    run(slark, 'Slark')
    run(templarassassin, 'Templar Assassin')
    run(tidehunter, 'tidehunter')
    run(venomancer, 'venomancer')
    run(warlock, 'warlock')
    run(lycan, 'lycan')
    run(alchemist, 'alchemist')
    run(necrophos, 'necrophos')
    run(ogre, 'ogre')
    run(crystalmaiden, 'crystalmaiden')
    run(keepermaybe, 'keepermaybe')
    run(queenofpain, 'queenofpain')
    run(tusk, 'tusk')
    run(sniper, 'sniper')
    run(axe, 'axe')
    run(dontknow, 'dontknow')
    run(dontknow2, 'dontknow2')
    run(dontknow4, 'dontknow4')
    run(dontknow5, 'dontknow5')
    run(dontknow6, 'dontknow6')
    run(dontknow9, 'dontknow9')
    run(dontknow10, 'dontknow10')
    run(dontknow11, 'dontknow11')
    run(dontknow12, 'dontknow12')
    run(dontknow13, 'dontknow13')
    run(dontknow14, 'dontknow14')
    run(dontknow15, 'dontknow15')
    run(dontknow16, 'dontknow16')
    run(dontknow17, 'dontknow17')
    run(dontknow18, 'dontknow18')
    run(dontknow19, 'dontknow19')
    run(dontknow20, 'dontknow20')
    run(dontknow21, 'dontknow21')
    run(dontknow22, 'dontknow22')
    
    #for i in range(60):
    #    run()
    end = timer()
    print(f'Took: {end-start} seconds')

    cv2.imwrite('res.png', img_rgb)
