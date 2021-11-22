import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm

def Undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def mag_thresh(gray, sobel_kernel=3, m_thresh=(50, 255)):

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= m_thresh[0]) & (scaled_sobel < m_thresh[1])] = 1
    return binary_output

def dir_threshold(gray, sobel_kernel=3, d_thresh=(0, 1.1)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobelx = np.abs(sobely)
    abs_sobely = np.abs(sobelx)
    
    direction = np.arctan2(abs_sobelx, abs_sobely)

    binary_output = np.zeros_like(direction)
    binary_output[(direction >= d_thresh[0]) & (direction < d_thresh[1])] = 1
    return binary_output


def region_of_interest(img, vertices=None):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    imshape= img.shape
    if vertices is None:
        vertices = np.array([[(0,imshape[0]),(imshape[1]//2 - 20, 3*imshape[0]//5), (imshape[1]//2 + 20, 3*imshape[0]//5), (imshape[1],imshape[0])]], dtype=np.int32)
        
    #draw_lines(img,[[(0,imshape[0],imshape[1]//2 - 20, 3*imshape[0]//5),(imshape[1]//2 - 20, 3*imshape[0]//5,imshape[1]//2 + 20, 3*imshape[0]//5),(imshape[1]//2 + 20, 3*imshape[0]//5,imshape[1],imshape[0]), (imshape[1],imshape[0],0,imshape[0])]] )
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def scaler2(channel):
    roi = region_of_interest(channel)
    
    
    scaled_channel = channel.copy()
    scale_mean = np.mean(roi)
    
    scaled_channel = scaled_channel - scale_mean
    scaled_channel[scaled_channel<0] = 0
    scale_max = np.max(scaled_channel)
    
    scaled_channel = np.uint8(255*scaled_channel/(scale_max))
    return scaled_channel

def Scale(channel):
    return np.uint8(255*channel/np.max(channel))

def Thresh(channel, thresh=(30,255), condition=False):
    Condition = (channel >= thresh[0]) & (channel < thresh[1])
    if condition:
        return Condition
    else:
        binary = np.zeros_like(channel)
        binary[Condition] = 1
        return binary
    

def ROI(channel):
    t = 1
    scaled_channel = channel.copy()
    scaled_channel = cv2.medianBlur(scaled_channel, 5)
    for x in range(10):
        for y in range(10):
            scale_mean = np.median(scaled_channel[90*x:90*(x+1), 160*y:160*(y+1)])
            scale_max = np.max(scaled_channel[90*x:90*(x+1), 160*y:160*(y+1)])
            
            if (scale_max - scale_mean)/scale_mean > t:
                ...
                #scaled_channel[90*x:90*(x+1), 160*y:160*(y+1)] = (scaled_channel[90*x:90*(x+1), 160*y:160*(y+1)] - t*scale_mean)#(scale_max-t*scale_mean)
            else:
                scaled_channel[90*x:90*(x+1), 160*y:160*(y+1)] = 0
    scaled_channel[scaled_channel<0] = 0
    scaled_channel[scaled_channel>0] = 255
    scaled_channel = np.uint8(scaled_channel)
    return scaled_channel



def Sobel(channel, ksize=11):
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=ksize)
    interiorx = np.abs(sobelx) 
    
    scaled_interiorx = np.uint8(255*interiorx/np.max(interiorx))
    return scaled_interiorx

def Binary(img,  thresh=(30, 255), visualize=False, ksize=25):
    img = np.copy(img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur_g = cv2.blur(scaler2(gray), (5,5))

    sobel = Sobel(blur_g, ksize=ksize)
    roi = ROI(blur_g)
    
    interior_g = cv2.bitwise_and(sobel, roi)
    canny = cv2.Canny(blur_g, 10, 255)
    interior_g = cv2.bitwise_or(interior_g, canny)
    if not visualize:
        return interior_g
    else:
   
        color = np.dstack((bitor, interior_s, interior_g))
    
        return np.uint8(color)



def Warp(img, src, target):
    try:
        y, x, c = img.shape
    except:
        y, x = img.shape

    M = cv2.getPerspectiveTransform(src, target)
    
    warped = cv2.warpPerspective(img, M, (x, y), flags=cv2.INTER_LINEAR)
   
    return warped


def Radius(fit):
    df = 2*fit[0]*900 + fit[1]
    ddf = 2*fit[0]

    R = (1+(df)**2)**1.5/(ddf)

    return R
    
def Write(string, img, line=1):
    img = img.copy()
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    position = (10,40 + 50*line)
    fontScale              = 2
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(img, string, 
        position, 
        font, 
        fontScale,
        fontColor,
        lineType)
    return img

##############################################################################################################################################

src = np.array([[740, 540], # upper left
                [530, 790], # lower left
                [920, 540], # upper right
                [1460, 790]], dtype=np.float32) #lower right

target = np.array([[600, 0],
                [600, 790],
                [1400, 0],
                [1400, 790]], dtype=np.float32)

M = cv2.getPerspectiveTransform(src, target)
    

def fitWindow(binary, x, binary_viz, left_current, right_current, left_delta, right_delta, windowWidth=100, windowHeight=90, thresh=35):
    binary_viz = binary_viz.copy()
    overlapU = windowHeight//2
    overlapD = windowHeight//2
    s = windowWidth//2

    left_win = binary[900 - windowHeight*(x+1) - overlapU :900-windowHeight*x + overlapD, left_current - s: left_current + s]
    right_win = binary[900 - windowHeight*(x+1) - overlapU :900-windowHeight*x + overlapD, right_current - s: right_current + s]
    
    binary_viz[900 - windowHeight*(x+1):900-windowHeight*x , left_current - s: left_current + s, 1:] = 0
    #binary_viz[900 - windowHeight*(x+1) + 40:900-windowHeight*x -40, left_current - 5: left_current + 5, 0] = 255
    
    binary_viz[900 - windowHeight*(x+1):900-windowHeight*x , right_current - s: right_current + s, :-1] = 0
    #binary_viz[900 - windowHeight*(x+1) + 40:900-windowHeight*x -40, right_current - 5: right_current + 5, -1] = 255
    
    binary_viz[900 - windowHeight*(x+1):900-windowHeight*x , : left_current - s] = 0
    binary_viz[900 - windowHeight*(x+1):900-windowHeight*x , left_current + s: right_current - s] = 0
    binary_viz[900 - windowHeight*(x+1):900-windowHeight*x , right_current + s: ] = 0
    
    binary_viz[900 - windowHeight*(x+1):900-windowHeight*x, left_current - s - 5: left_current - s + 5] = np.array([0,255,0])
    binary_viz[900 - windowHeight*(x+1):900-windowHeight*x, left_current + s - 5: left_current + s + 5] = np.array([0,255,0])
    binary_viz[900 - windowHeight*(x) - 10:900-windowHeight*x, left_current-s: left_current + s ] = np.array([0,255,0])
    binary_viz[900 - windowHeight*(x+1) - 10:900-windowHeight*(x+1 )+ 10, left_current-s: left_current + s ] = np.array([0,255,0])
    
    binary_viz[900 - windowHeight*(x+1):900-windowHeight*x, right_current - s - 5: right_current - s + 5] = np.array([0,255,0])
    binary_viz[900 - windowHeight*(x+1):900-windowHeight*x, right_current + s - 5: right_current + s + 5] = np.array([0,255,0])
    binary_viz[900 - windowHeight*(x) - 10:900-windowHeight*x, right_current-s: right_current + s ] = np.array([0,255,0])
    binary_viz[900 - windowHeight*(x+1) :900-windowHeight*(x+1) + 10, right_current-s: right_current + s ] = np.array([0,255,0])

    try:
        if np.max(np.sum(left_win, axis=0)) > thresh:
            left_delta = np.argmax(np.sum(left_win, axis=0)) - s

        elif np.max(np.sum(right_win, axis=0)) > thresh:
            #left_delta = np.argmax(np.sum(right_win, axis=0)) - s
            ...
        

    except:
        ...
    try:
        if np.max(np.sum(right_win, axis=0)) > thresh:
            right_delta = np.argmax(np.sum(right_win, axis=0)) - s

        elif np.max(np.sum(left_win, axis=0)) > thresh:
            ...
            #right_delta = np.argmax(np.sum(left_win, axis=0)) - s
            
    except:
        ...
        
    #print(right_delta)
    left_current += left_delta 
    right_current += right_delta
        
    return binary_viz, left_current, right_current, left_delta, right_delta
    
#########################################################################################################################
def FindLane(path, indexes, show=False):
    ploty = np.arange(0, 900)
    datas = []
    for i in tqdm(indexes):
        img = mpimg.imread(f'{path}/{i}.jpg')
        warped = Warp(img, src, target)
        binary = Binary(warped, thresh=(20,255))
        binary[binary<20] = 0
        
        gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        binary_viz = np.dstack((binary, gray, binary))
        
        if show:
            plt.imshow(gray, cmap='gray')
            plt.show()
        
            plt.imshow(binary, cmap='gray')
            plt.show()
        
        histogram = np.sum(binary[450:,:], axis=0)
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        left_current = leftx_base
        right_current = rightx_base
        left_delta = 0
        right_delta = 0
        
        if show:
            plt.plot(histogram/histogram.max())
            plt.plot([rightx_base, rightx_base], [0, 1])
            plt.plot([leftx_base, leftx_base], [0, 1])
            plt.show()
        
        for x in range(10):
            _, left_current, right_current, left_delta, right_delta = fitWindow(binary, x, binary_viz, left_current, right_current, left_delta, right_delta)
            binary_viz, left_current, right_current, left_delta, right_delta = fitWindow(binary, x, binary_viz, left_current, right_current, left_delta, right_delta)
    
        left_idx = binary_viz[:,:,0].nonzero()
        right_idx = binary_viz[:,:,2].nonzero()
        
        try:
            left_fit = np.polyfit(left_idx[0], left_idx[1], 2)
        except:
            left_fit = np.polyfit([0,900], [400, 400], 2)
            
        try:
            right_fit = np.polyfit(right_idx[0], right_idx[1], 2)
        except:
            left_fit = np.polyfit([0,900], [1200, 1200], 2)
            
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        leftRadius = Radius(left_fit)
        rightRadius = Radius(right_fit)
    
        laneCenter = (left_fitx[-1] + right_fitx[-1])/2
        deviation = laneCenter - 800
    
        stringL = f'Left = {10000/leftRadius:.2f}'
        stringR = f'Right = {10000/rightRadius:.2f}'
    
        ptsL = np.vstack((left_fitx, ploty)).astype(np.int32).T
        cv2.polylines(binary_viz, [ptsL], isClosed=False, color=(255,0,0), thickness=3)
        
        ptsR = np.vstack((right_fitx, ploty)).astype(np.int32).T
        cv2.polylines(binary_viz, [ptsR], isClosed=False, color=(0,0,255), thickness=3)
        
        if show:
            plt.imshow(binary_viz)
            plt.show()
        
        overlay = warped.copy()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, np.int_([pts]), (0,255, 0))
        newwarp = Warp(overlay, target, src)
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        result = Write(f'index = {i}', result, line=1)
        result = Write(stringL, result, line=2)
        result = Write(stringR, result, line=3)
        result = Write(f'{deviation:.2f}', result, line=4)
        
        datas.append([10000/leftRadius, 10000/rightRadius, deviation/100])
        if show:
            plt.imshow(result)
            plt.show()
            
    return np.array(datas)