# import libraries

from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import os
import sys
import time

TGT_WIDTH = 800                 # set final width for image to be scaled to
TGT_AR = 0.71                   # set desired aspect ratio
DET_WIDTH = 500                 # set width to scaled to for face detection
INFOLDER = r'images/'          # input folder of images to be processed
OUT_PASS = r'output/'           # folder for correctly processed images
OUT_TOO_SMALL = r'too small/'   # folder for original images with small dimensions
OUT_MULTI = r'to check/'        # folder for images with > 1 faces detected
OUT_UNDET = r'face undetected/' # folder for images with 0 face detected
UPSAMPLE = 1                    # HOG number of upsampling

def original_params(f, image, scaledWidth):
    fileName = os.path.splitext(f)[0]
    fileExt = os.path.splitext(f)[1]
    iy, ix, ic = image.shape
    aspect = ix/iy
    scaleFactor = ix/scaledWidth
    return (fileName, fileExt, iy, ix, ic, aspect, scaleFactor)

def detect_faces(image, scaledWidth, detector, upsample = 1):
    image = imutils.resize(image, width = scaledWidth)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, upsample)
    return rects
        
def face_centroid(rect, scaleFactor):
    (x, y, w, h) = rect_to_bb(rect)
    cx = x + w/2
    cx = round(cx * scaleFactor)
    cy = y + h/2
    cy = round(cy * scaleFactor)
    return (cx, cy)
    
def get_crop_params(aspectOriginal, aspectTarget, x_face_centroid, 
                y_face_centroid, x_original, y_original):
    if aspectOriginal > aspectTarget:
        x_len = y_original * aspectTarget
        x_left = x_face_centroid - x_len/2
        x_right = x_face_centroid + x_len/2
        y_top = 0
        y_bottom = y_original
    elif aspectOriginal < aspectTarget:
        y_len = x_original/aspectTarget
        x_left = 0
        x_right = x_original
        y_top = 0
        y_bottom = y_len
    elif aspectOriginal == aspectTarget:
        x_left = 0
        x_right = x_original
        y_top = 0
        y_bottom = y_original
    # handle off-centered faces
    if x_left < 0:
        x_left = 0
        y_bottom = x_right/aspectTarget
    if y_top < 0:
        y_top = 0
    if x_right > x_original:
        x_right = x_original
        y_bottom = x_right/aspectTarget
    if y_bottom > y_original:
        y_bottom = y_original
    return (round(x_left), round(x_right), round(y_top), round(y_bottom))

def image_out(image, path, crop_params, resizeWidth=None):
    x_left = crop_params[0]
    x_right = crop_params[1]
    y_top = crop_params[2]
    y_bottom = crop_params[3]
    image = image[y_top:y_bottom, x_left:x_right]
    if resizeWidth is None:
        cv2.imwrite(path, image)
    else:
        image = imutils.resize(image, width=resizeWidth)
        cv2.imwrite(path, image)
    

# initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# check if input folder exists
if not os.path.exists(INFOLDER):
    raise Exception('Input folder missing, or named wrongly. Folder name should be "{}".'.format(INFOLDER))
    sys.exit(0)

# initialize folders for pipeline
folders = [OUT_PASS, OUT_TOO_SMALL, OUT_MULTI, OUT_UNDET]
for fd in folders:
    if not os.path.exists(fd):
        os.mkdir(fd)

start = time.time()
for f in os.listdir(INFOLDER):
    img = cv2.imread(INFOLDER + f)
    
    # check if loaded file is an image file
    if img is None:
        continue
    
    (fname, ext, iy, ix, ic, ar_orig, scale) = original_params(f, img, DET_WIDTH)
    
    # check if original image is too small
    if ix < DET_WIDTH:
        outfolder = OUT_TOO_SMALL
        cv2.imwrite(outfolder + f, img)
        continue
    
    # detect faces
    rects = detect_faces(img, DET_WIDTH, detector, upsample = UPSAMPLE)
    
    # piping processed images to appropriate folders
    if len(rects) == 0:
        outfolder = OUT_UNDET
        cv2.imwrite(outfolder + f, img)
        print(f + ' - No face')
    elif len(rects) == 1:
        outfolder = OUT_PASS
        outpath = outfolder + f
        (cx, cy) = face_centroid(rects[0], scale)
        crop_params = get_crop_params(ar_orig, TGT_AR, cx, cy, ix, iy)
        image_out(img, outpath, crop_params, resizeWidth=TGT_WIDTH)
        print(f + ' - single face')
    elif len(rects) > 1:
        # more than one face detected index each face and output in 'to check' folder
        outfolder = OUT_MULTI
        i = 1
        for rect in rects:
            outpath = outfolder + fname + '_' + str(i) + ext
            (cx, cy) = face_centroid(rect, scale)
            crop_params = get_crop_params(ar_orig, TGT_AR, cx, cy, ix, iy)
            image_out(img, outpath, crop_params, resizeWidth=TGT_WIDTH)
            i += 1

end = time.time()

print('Total time = {}'.format(end-start))