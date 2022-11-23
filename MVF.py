from PIL import Image
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
import cv2
import glob
import scipy.misc as misc
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.io import imshow, imread
from skimage import morphology
import skimage
import os
import pandas as pd
from sklearn.cluster import DBSCAN
from collections import Counter
import shutil
#from datetime import datetime
from datetime import datetime as dt
import csv 
from pylab import *
from numpy import loadtxt
from scipy.optimize import leastsq
import pathlib
from skimage.filters import threshold_minimum
from skimage.filters import threshold_yen
import sys
import imageio
import random
import re

Image.MAX_IMAGE_PIXELS = 933120000 

###parameters to set
##image size
size = 16000 #set a desired window size, 1000, 2000, 4000, 8000, 12000, 16000
###total repetitions
Count = 5 #number of repeating
###Image Dir
folder = "/home/mr189568/venv6/Volume farction/DP750/"#folder contains SEM ano images (24000*24000)
###GAN model
model = load_model('/home/mr189568/venv6/Volume farction/model_068400.h5')
#### los gehts
section = int(size/250)
main_path = "/home/mr189568/venv6/Volume farction/"
sub_path = "Cropped250"
FINAL = 'FINAL'
natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]
Martensite_V_F = []
# load an image
def load_image(filename, size=(256,256)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels

for R in range(Count):
    ###random coordination
    n = random.randint(500,22500-size)#	depend on SEM Pano Size, for prevent reading out of image
    m = random.randint(500,22500-size)
    for filename in glob.glob(folder + '*.png'): 
        ##Setting the coordinations for cropping image 
        left = n
        top = m 
        right = n + size
        bottom = m + size
        im=Image.open(filename)
        im1 = im.crop((left, top, right, bottom)) 
        im1.save('%s_%s_%s.jpg'%((filename)[:-4],str(size),R))
        new_im=Image.open('%s_%s_%s.jpg'%((filename)[:-4],str(size),R))
        os.chdir(folder)
        os.mkdir((filename)[:-4] + '_Cropped250_' + str(R))
        os.chdir((filename)[:-4] + '_Cropped250_' + str(R))
        os.mkdir(FINAL)
        ##saving each cropped image
        for j in range(section):

            for i in range (section):

                cropped_im = new_im.crop(((i)*250, (j)*250, (i+1)*250, (j+1)*250)) 
                cropped_im.save('Pic_r%s_c%s.jpg'%((j+1),(i+1)))
        
        
        for FILENAME in glob.glob((filename)[:-4] + '_Cropped250_' + str(R) + '/*.jpg'):
            ## load source image
            src_image = load_image(FILENAME)
            ## generate image from source
            gen_image = model.predict(src_image)
            ## scale from [-1,1] to [0,1]
            gen_image = (gen_image + 1) / 2.0
            ## saving GAN result
            imageio.imsave('%s_GAN.jpg'%(FILENAME)[:-4],gen_image[0]) 

            ## Closing
            img =cv2.imread('%s_GAN.jpg'%(FILENAME)[:-4])
            kernel = np.ones((3,3),np.uint8)
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) 

            ## Morphology_Correction
            org_img = closing
            grayscale = skimage.color.rgb2gray(org_img)
            binarized = np.where(grayscale>0.59 , 1, 0)
            processed = morphology.remove_small_objects(binarized.astype(bool), min_size=150, connectivity=1).astype(int)

            ## black out pixels
            mask_x, mask_y = np.where(processed == 0)
            org_img[mask_x, mask_y, 0] = 168
            org_img[mask_x, mask_y, 1] = 38
            org_img[mask_x, mask_y, 2] = 62
            x = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
            name_after_Morphology_Correction =  '%s_GAN_P.jpg'%(FILENAME)[:-4]
            io.imsave(name_after_Morphology_Correction, x)
            shutil.copy(name_after_Morphology_Correction , (filename)[:-4] + '_Cropped250_' + str(R) + '/FINAL/')

        ###Stitching cropped images    
        image_list = []
        image_list_Sort = []
        for IMG in glob.glob((filename)[:-4] + '_Cropped250_' + str(R) + '/FINAL/' + '*.jpg'):
            image_list.append(IMG)

        image_list_Sort = sorted(image_list, key=natsort)
        images = [Image.open(x) for x in image_list_Sort]
        widths, heights = zip(*(i.size for i in images))
        total_width = int(section * 256)
        max_height = int(section * 256)
        NEW_IM = Image.new('RGB', (total_width, max_height))
        
        for c in range(len(images)):
                NEW_IM.paste(images[c], (c%section * 256,int(c/section) * 256))
        ###saving stitched image        
        NEW_IM.save('%s_%s_%s_TOTAL.jpg'%((filename)[:-4],str(size),R))
        
        ###Double Thresholding & Saving
        DB = cv2.imread('%s_%s_%s_TOTAL.jpg'%((filename)[:-4],str(size),R),0)
        ret1, DB1 = cv2.threshold(DB, 100, 255, cv2.THRESH_BINARY)
        ret2, DB2 = cv2.threshold(DB, 200, 255, cv2.THRESH_BINARY)
        black_pixels= np.argwhere(DB1-DB2)

        ###Convert numpy to a list
        my_list=black_pixels.tolist()
        black_list=int(len(my_list))
	
	###MVF calculation
        print('Martensite Volume Fraction is: ', 100 * (black_list/(DB.shape[0]*DB.shape[1])))
        MVF =str(round(100 * (black_list/(DB.shape[0]*DB.shape[1])),2))
        Martensite_V_F.append([(filename)[:-4],str(size),R,MVF])
        os.chdir(folder)

os.chdir(main_path)
print(Martensite_V_F)

###Crating txt file contains MVF
with open("Martensite VF.txt","w") as f:
        for item in Martensite_V_F:
            f.write("%s\n" % item)       
