import cv2
import numpy as np

cutted_img_height = 350 #@param {type: "slider", min: 0, max: 480, step:1}
resize_ratio = 0.35 #@param {type: "slider", min: 0.0, max: 1.0, step:0.01}

img_height = 480
top_crop = img_height - cutted_img_height

img_final_height = int(cutted_img_height * resize_ratio)
img_final_width = int(640 * resize_ratio)

def cropimg(img):
    """
    Crop top of image top_crop px, they are noise most of the time

    :param img: (RGB image as np array) Image to be cropped
    """
    return img[top_crop:,:]

def houghtransform(img):
    """
    Apply Hough Line transform, for theory see:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

    :param img: (RGB image as np array)
    """
    frame_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY, 3)
    edges = cv2.Canny(frame_BGR,50,150,apertureSize = 3)
    #minLineLength = 100
    #maxLineGap = 10
    #lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    #for x1,y1,x2,y2 in lines[0]:
    #    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    imgRGB = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)
    return imgRGB

def resizeimg(img, ratio):
    """
    Resize image
    :param img: (np array)
    :param ratio: (float) 0<ratio<1
    """
    try:
        return cv2.resize(img, (0,0), fx=ratio, fy=ratio)
    except cv2.error as e:
        print(e)
        print("Error: Image too small")
        return img
  
def takeyellow(img):
    """
    Extract yellow lines, for color ranges see:
    https://stackoverflow.com/questions/48109650/how-to-detect-two-different-colors-using-cv2-inrange-in-python-opencv

    :param img: (RGB image as np array)
    """
    frame_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    frame_threshold = cv2.inRange(frame_HSV, (20,100,100), (50, 255, 255))
    imgRGB = cv2.cvtColor(frame_threshold, cv2.COLOR_GRAY2RGB)
    return imgRGB

def takewhiteyellow(img):
    """
    Extract white and yellow lines

    :param img: (RGB image as np array)
    """
    #white
    sensitivity = 100
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])
    frame_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    maskwhite = cv2.inRange(frame_HSV, lower_white, upper_white)
    img[maskwhite > 0] = (255, 0, 0)
    img[maskwhite == 0] = (0,0,0)
    #yellow
    maskyellow = cv2.inRange(frame_HSV, (15,70,70), (50, 255, 255))
    img[maskyellow > 0] = (0, 255, 0)
    return img

def white_balance(img):
    """
    Grayworld assumption:
    https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption/46391574

    :param img: (RGB image as np array)
    """
    result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return result

def birdeye(img, test=False):
    """
    Apply perspective transform to image, for theory see:
    https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html

    Also:
    https://stackoverflow.com/questions/48264861/birds-eye-view-opencv
    """
    black_wd = int(2500*resize_ratio)
    black_img = np.zeros(shape=(img.shape[0], black_wd, 3), dtype=np.uint8)
    x_offset=int((black_wd-img.shape[0])/2)
    black_img[:img.shape[0], x_offset:x_offset+img.shape[1]] = img
    img = black_img
    row, cols, ch = img.shape
    src = np.float32([[1060, 170], [1600, 170], [110, row/resize_ratio], [cols/resize_ratio, row/resize_ratio]])*resize_ratio
    dst = np.float32([[0,0],[img_final_width,0],[0,img_final_height],[img_final_width,img_final_height]])
    M = cv2.getPerspectiveTransform(src, dst)
    img_transformed = cv2.warpPerspective(img, M, (cols, row))
    img_cutted_to_480_640 = img_transformed[:img_final_height, :img_final_width]
    return img_cutted_to_480_640

class SBWrapper:
    transposed_shape = (img_final_height, img_final_width, 3)
    def __init__(self):
        pass
    
    def preprocess(self, obs):
        """
        Preprocess the observation
        """
        # cropped = cropimg(obs)
        balanced = white_balance(obs)
        img = takewhiteyellow(balanced)
        resized = resizeimg(img, resize_ratio)
        bird = birdeye(resized)
        # img = extracted_colors/255
        return bird