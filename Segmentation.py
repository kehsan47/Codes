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
import imageio
import xlwt
from xlwt import Workbook 
import re



########################################################

###FOR GAN

########################################################



###Import folder contains images for DBSCAN + GAN
folder = '/work/mr189568/Thesis/DP775/DP775_14_4_q/TOT/Crops_3/'

###Necessary path
main_path = "/home/mr189568/venv6/DB_GAN/"
sub_path = "DBSCAN_GAN_"
Segmentation_folder = folder + '/Segmentation/'


###Current time, avoiding overwriting
now = dt.now()
current_time = now.strftime("%H:%M:%S")

###Make directory to save results
os.chdir(folder)
os.mkdir(sub_path + str(current_time))
os.mkdir('Segmentation')
os.chdir(main_path)

###Exporting images and names and lists
names = []
images = []
c = 0
Damage_Size = []
Aspect_ratio = []
Shadowing = "Shadowing"
Inclusion = 'Inclusion'
#N = "Not Classified"

###Load GAN model, set the weight directory
model_GAN = load_model('/home/mr189568/venv6/DB_GAN/model_068400.h5')

def load_image(filename, size=(256,256)):
	###Load image with the preferred size
	pixels = load_img(filename, target_size=size)
	###Convert to numpy array
	pixels = img_to_array(pixels)
	###Scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	###Reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels

###Find all segmented images
for filename in os.listdir(folder):
    #if not Shadowing in filename:
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                    images.append(img)
                    c+=1
                    names.append(filename)
print('number of images: ' + str(c))

###Sorting image names
natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]
names = sorted(names, key=natsort)

###Reading every images, preprocessing
for i in range(len(images)):
    
    os.chdir(folder)
    print(names[i])
    img = cv2.imread(names[i])
    
    
    ###Erosion & Saving (Keep it)

    #kernel = np.ones((1,1),np.uint8)
    #erosion = cv2.erode(img, kernel, iterations = 1)
    #name_after_erosion = "%s_Eroded.png"%names[i][:-4]
    os.chdir(sub_path + str(current_time))
    os.mkdir(names[i][:-4])
    os.chdir(names[i][:-4])
    #cv2.imwrite(names[i],img)
    #cv2.imwrite(name_after_erosion,erosion)
  
    
    ###Thresholding & Saving
    
    ###Helping for Thresholding
    #img = cv2.imread(name_after_erosion)
    for x in range(img.shape[0]): 
        for y in range(img.shape[1]):
            if int(img[x,y][0]) > 100:
                img[x,y] = [255,255,255]
    
    ###Dynamic Thresholding

    yen = 0.8 * threshold_yen(img)
    black_pixels=np.argwhere(img<yen)
    ret,thresh1 = cv2.threshold(img,yen,255,cv2.THRESH_BINARY)
    name_after_erosion_Thresholding = '%s_Thresholded.png'%names[i][:-4]
    #cv2.imwrite(name_after_erosion_Thresholding,thresh1)

    ###Convert numpy to a list
    my_list=black_pixels.tolist()
    black_list=[]
 
    ###Each pixel reported 3 times, we only need one
    for j in range(len(my_list)):
        if j%3==0:
            black_list.append(my_list[j])
    
 
    ###There are 3 elements in each list we do not need the 3rd one
    for k in range(len(black_list)):
        del black_list[k][2]
        
    ###[y,x]~> sorted by Y    
    with open("black_pixles.txt","w") as f:
        for item in black_list:
            f.write("%s\n" % item)
              
    ###Check if there is a damage or not
    if len(black_list) != 0:

        ###Creating csv file
        with open("black_pixels.csv","w",newline="") as file_writer:
           fields=["X","Y"]
           writer=csv.DictWriter(file_writer,fieldnames=fields)
           writer.writeheader()
           for e in range(len(black_list)):
                writer.writerow({"X":black_list[e][1],"Y":black_list[e][0]})
    
        ###Load the data set
        df = pd.read_csv('black_pixels.csv', encoding='utf-8')
        df.head()
        data = df.iloc[:,:]
        
        ###DBSCAN model
        model = DBSCAN(eps = 1, min_samples = 5).fit(data)
        outliers_df = data[model.labels_ == -1]
        clusters_df = data[model.labels_ != -1]

        colors = model.labels_
        colors_clusters = colors[colors != -1]
        color_outliers = 'balck'
        clusters = Counter (model.labels_)
        

        ###Ploting the result
        plot = np.ones(img.shape[:2], dtype = "uint8")*255
        clusters_X_list = clusters_df['X'].tolist()
        clusters_Y_list = clusters_df['Y'].tolist()
        outliers_X_list = outliers_df['X'].tolist()
        outliers_Y_list = outliers_df['Y'].tolist()
        for z in range(len(clusters_X_list)):
            plot[clusters_Y_list[z]][clusters_X_list[z]] = 0
        finalname = '%s_DBscaned.png'%name_after_erosion_Thresholding[:-4] 
        cv2.imwrite(finalname,plot)

        ###Extracting each Cluster
        DB_TARGET =[]
        for q in range(len(clusters)):
            Clusters = data[model.labels_ == q]
            plot = np.ones(img.shape[:2], dtype = "uint8")*255
            Each_clusters_X_list = Clusters['X'].tolist()
            Each_clusters_Y_list = Clusters['Y'].tolist()
            
            ###Appending each cluster to list
            DB_TARGET.append([])
            for w in range(len(Clusters)):
                plot[Each_clusters_Y_list[w]][Each_clusters_X_list[w]] = 0
                DB_TARGET[q].append([Clusters['X'].tolist()[w],Clusters['Y'].tolist()[w]])
                
        ###Converting to yellow image
        pixels = cv2.imread(finalname) 
        for l in range(pixels.shape[0]): 
            for p in range(pixels.shape[1]):
                if int(pixels[l,p][0]) == 0:
                    pixels[l,p] = [0, 255 ,255]
                    
        yellow_Full = '%s_DB.png'%names[i][:-4]       
        cv2.imwrite(yellow_Full,pixels)
  
    ###If there is no damage
    else:
        pass
    os.chdir(folder)    
    ###Load source image
    src_image = load_image(names[i])
    ###Generate image from source
    gen_image = model_GAN.predict(src_image)
    ###Scale from [-1,1] to [0,1]
    gen_image = (gen_image + 1) / 2.0
    ###Saving GAN result
    #misc.imsave( '%s_GAN.png'%names[i][:-4],gen_image[0])
    imageio.imsave('%s_GAN.png'%names[i][:-4],gen_image[0]) 
    base = cv2.imread('%s_GAN.png'%names[i][:-4])

    ###Resizing the GAN result
    dim = (250, 250)
    base = cv2.resize(base, dim, interpolation = cv2.INTER_AREA)
    #basename = '%s_DB_Resize.png'%names[i][:-4]  
    #cv2.imwrite(basename,base)
    ###Extrating damage coordinations from GAN
    GAN_Yellow = []
    for v in range(len(base)):
        for m in range(len(base)):
            if base[v][m][2] > 100:
                GAN_Yellow.append([m, v])
                
    ###Extracting the common damage coordination from DBscan & GAN            
    plot_target = np.ones(img.shape[:2], dtype = "uint8")*255
    Each_clusters_X_list_target = []
    Each_clusters_Y_list_target = []
    target_cluster = []
    
    for f in range(len(DB_TARGET)):
        
        if len(DB_TARGET[f]) != 0:
        
            xcor = []

            ycor = []

            for i9 in range(len(DB_TARGET[f])):
                xcor.append(DB_TARGET[f][i9][0])
                ycor.append(DB_TARGET[f][i9][1])
            xmin= min(xcor)
            xmax = max(xcor)
            ymin = min(ycor)
            ymax = max(ycor)

            print(xmin,xmax,ymin,ymax)

            GAN_Yellow = []

            for xCOR in range(xmin,xmax):

                for yCOR in range(ymin,ymax):

                    if base[yCOR][xCOR][2] > 120:
                        
                        GAN_Yellow.append([xCOR, yCOR])
            g = 0
            print(GAN_Yellow)
            for h in DB_TARGET[f]:
                if h in GAN_Yellow:
                    g += 1
                    
            if g > 0.5*(len(GAN_Yellow)):
                print(f)
                target_cluster.append(f)
    
    if len(target_cluster) != 0:
        
        ###Ploting the common damage sites from DBscan & GAN
        for num in target_cluster:
            Clusters_target = data[model.labels_ == num]
            Each_clusters_X_list_target = Clusters_target['X'].tolist()
            Each_clusters_Y_list_target = Clusters_target['Y'].tolist()

            for counter in range(len(Clusters_target)):
                plot_target[Each_clusters_Y_list_target[counter]][Each_clusters_X_list_target[counter]] = 0
        target = '%s_target.png'% (finalname[:-4])
        cv2.imwrite(target,plot_target)

        pixels_target = cv2.imread(target)

        ###Converting the target image to yellow
        for N in range(pixels_target.shape[0]): 
            for M in range(pixels_target.shape[1]):
                if int(pixels_target[N,M][0]) == 0:
                    pixels_target[N,M] = [0, 255 ,255]
                    
        ###Resizing the final yello image
        yellow = pixels_target
        yellow = cv2.resize(yellow, dim, interpolation = cv2.INTER_AREA)
        
        Damages = []
        for t in range(yellow.shape[0]): 
            for b in range(yellow.shape[1]):
                if int(yellow[t,b][0]) != 255:
                    Damages.append([t,b])
                    
        ###Combining the DBscan & GAN results
        for a in range(base.shape[0]): 
            for r in range(base.shape[1]):
                    if [a,r] in Damages:
                        base[a,r] = [0, 255 ,255]
        
        
    else:
        ###If there is not common damage site just return GAN result
        yellow = np.ones(img.shape[:2], dtype = "uint8")*255
        yellow = cv2.resize(yellow, dim, interpolation = cv2.INTER_AREA)
    
            
            
    #without_CS_Cl = '%s_without_CS_Cl.png'% (finalname[:-4])
    #cv2.imwrite(without_CS_Cl,base)
	
    img1 = base
    
    ###Closing Postprocessing
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)

    ###Morphology_Correction
    org_img = closing
    grayscale = skimage.color.rgb2gray(org_img)
    binarized = np.where(grayscale>0.59 , 1, 0)
    processed = morphology.remove_small_objects(binarized.astype(bool), min_size=150, connectivity=1).astype(int)

    ###Black out pixels
    mask_x, mask_y = np.where(processed == 0)
    org_img[mask_x, mask_y, 0] = 168
    org_img[mask_x, mask_y, 1] = 38
    org_img[mask_x, mask_y, 2] = 62

    ###Saving Final result
    x = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    name_after_Morphology_Correction =  '%s_CS_CL.png'%names[i][:-4]  
    io.imsave(name_after_Morphology_Correction, x)

    
    size = cv2.imread(name_after_Morphology_Correction)
    shutil.move(name_after_Morphology_Correction , Segmentation_folder)
    

for f in os.listdir(folder):
    if  f.endswith("GAN.png") or f.endswith("Thresholded_DBscaned_target.png"):
        os.remove(os.path.join(folder, f))
        
for d in os.listdir(folder):
    if  d.endswith(current_time):
        shutil.rmtree(d)
        
        
    
os.chdir(main_path)
