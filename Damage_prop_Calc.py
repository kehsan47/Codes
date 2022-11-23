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



###Import folder which contains images for DBSCAN + GAN
folder = '/work/mr189568/Thesis/DP800/DN/DP800_19/NewFrom_3_AfterDBSCANPrvd_Mani/Segmentation/'
check =  80
###Necessary path
main_path = "/home/mr189568/venv6/DB_GAN/"
sub_path = "DBSCAN_GAN_"


###Current time, for avoid overwriting
now = dt.now()
current_time = now.strftime("%H:%M:%S")

###Make directory to save results
os.chdir(folder)
os.mkdir(sub_path + str(current_time))
os.chdir(main_path)

###Exporting images and names and lists
names = []
images = []
c = 0
Damage_Size = []
Aspect_ratio = []
Status = []


###Find all segmented images
for filename in os.listdir(folder):
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
    img = cv2.imread(names[i],0)
    os.chdir(sub_path + str(current_time))
    os.mkdir(names[i][:-4])
    os.chdir(names[i][:-4])
    
    ###Thresholding & Saving
    black_pixels=np.argwhere(img>180)
    #print(black_pixels)

        
    ###[y,x]~> sorted by Y    
    with open("black_pixles.txt","w") as f:
        for item in black_pixels:
            f.write("%s\n" % item)
              
    ###Check if there is a damage or not
    if len(black_pixels) != 0:

        ###Creating csv file
        with open("black_pixels.csv","w",newline="") as file_writer:
           fields=["X","Y"]
           writer=csv.DictWriter(file_writer,fieldnames=fields)
           writer.writeheader()
           for e in range(len(black_pixels)):
                writer.writerow({"X":black_pixels[e][1],"Y":black_pixels[e][0]})
    
        ###Load the data set
        df = pd.read_csv('black_pixels.csv', encoding='utf-8')
        df.head()
        data = df.iloc[:,:]
        
        ###DBSCAN model
        model = DBSCAN(eps = 1, min_samples = 1).fit(data)
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
        finalname = '%s_DBscaned.png'%names[i][:-4]
        cv2.imwrite(finalname,plot)
        #shutil.copy(finalname , folder)

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
            #Finalname = '%s_%s.png'% (names[i][:-4],q)
            #cv2.imwrite(Finalname,plot)    

  
    ###If there is no damage
    else:
        DB_TARGET = []
      
                
    ###Extracting the common damage coordination from DBscan & GAN            
    plot_target = np.ones(img.shape[:2], dtype = "uint8")*255
    Each_clusters_X_list_target = []
    Each_clusters_Y_list_target = []
    target_cluster = []
    
    for f in range(len(DB_TARGET)):
        
        if len(DB_TARGET[f]) != 0:
        
            xcor = []

            ycor = []
            
            g = 0

            for i9 in range(len(DB_TARGET[f])):
                xcor.append(DB_TARGET[f][i9][0])
                ycor.append(DB_TARGET[f][i9][1])
                

            for i10 in range(len(DB_TARGET[f])):
                if  125 - (check/2) <xcor[i10]< 125 + (check/2):
                    if 125 - (check/2)<ycor[i10]<125 + (check/2):
                        g = g + 1
                        
            if g !=0:
                target_cluster.append(f)
                #print(target_cluster)
    
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

        pixels_target = cv2.imread(target,0)
        pixels_target2 = cv2.imread(target)
        ###Extraction Damage size
        points = np.argwhere(pixels_target==0)
        
        if len(points) >= 5:
            Damage_Size.append(len(points))

            ###Extraction of aspect ratio
            ellipse = cv2.fitEllipse(points)
            (xc,yc),(WT,HT),ang = ellipse
            #ellipse2 = (yc,xc),(WT,HT),90-ang
            #target_new = cv2.ellipse(pixels_target2,ellipse2,(0,0,255),2)
            #target_new_name = '%s_target_new.png'% (finalname[:-4])
            #cv2.imwrite(target_new_name,target_new)
            aspect_ratio = float(WT)/HT
            Aspect_ratio.append(aspect_ratio)
            Status.append(len(target_cluster))
            
        else:
            Damage_Size.append('No damage Found')
            Aspect_ratio.append('No damage Found')
            Status.append('No damage Found')
            
        
        
                
    else:
        Damage_Size.append('No damage Found')
        Aspect_ratio.append('No damage Found')
        Status.append('No damage Found')
        

        
    
###Creating a Excel file for damage size & aspect ratio   
WB = Workbook()
Sheet1 = WB.add_sheet ('Damage_Size')
style = xlwt.XFStyle ()   # create a style Object initialization pattern 
Al = xlwt.Alignment () 
Al.horz = 0x02       # arranged horizontally centered 
style.alignment = Al 


Sheet1.write (0,0,'Image',style)
Sheet1.write (0,1,'Damage Size',style)
Sheet1.write (0,2,'Damage Aspect Ratio',style)
Sheet1.write (0,3,'Number of Clusters',style)
 

for i in range(c):
    Sheet1.write (i+1,0,names[i],style)
    Sheet1.write (i+1,1,Damage_Size[i],style)
    Sheet1.write (i+1,2,Aspect_ratio[i],style)
    Sheet1.write (i+1,3,Status[i],style)


Coordinates_Excel = folder + "Damage_size.xls"
WB.save (Coordinates_Excel)

os.chdir(folder)

for d in os.listdir(folder):
    if  d.endswith(current_time):
        shutil.rmtree(d)
        
os.chdir(main_path)
