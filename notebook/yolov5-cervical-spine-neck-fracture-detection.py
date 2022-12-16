#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


import cv2
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from torch.utils.data import Dataset, DataLoader

import tqdm.notebook as tq


import gc


import albumentations as albu
from albumentations import Compose


from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, jaccard_score
import itertools


from PIL import Image

from numpy import asarray

from skimage.transform import resize


import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')



print(torch.__version__)
print(torchvision.__version__)


# In[2]:


import random

seed_val = 101

os.environ['PYTHONHASHSEED'] = str(seed_val)
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True


# In[8]:


os.listdir('/home/nvr5386/Desktop/RSNA/project_data/RSNA')


# In[15]:


base_path = '/home/nvr5386/Desktop/RSNA/project_data/RSNA/'

prep_data_path = '/home/nvr5386/Desktop/RSNA/project_data/'


# ## Config

# In[6]:


NUM_EPOCHS = 80
BATCH_SIZE = 32
IMAGE_SIZE = 512 # Yolo will automatically resize the input images to this size.

CHOSEN_FOLD = 0

NUM_FOLDS = 5

NUM_CORES = os.cpu_count()
NUM_CORES


# In[13]:


get_ipython().system('ls')


# ## Load the train data

# In[16]:


path = prep_data_path + 'df_data.csv'

df_data = pd.read_csv(path)

print(df_data.shape)

df_data.head()


# In[17]:


df_data['label'].value_counts()


# ## Process the train data

# In[18]:


df_data['target'] = list(df_data['label'])



# In[19]:


df_data['target'].value_counts()


# ## Create a column for the bbox info

# In[20]:


path = base_path + 'train_bounding_boxes.csv'
df_bbox = pd.read_csv(path)

print(df_bbox.shape)

df_bbox.head()


# In[21]:


def create_study_slice(row):
    
    study_id = str(row['StudyInstanceUID'])
    slice_num = str(row['slice_number'])
    
    study_slice = study_id + '_' + slice_num
    
    return study_slice

df_bbox['study_slice'] = df_bbox.apply(create_study_slice, axis=1)

df_bbox = df_bbox.set_index('study_slice')

print(df_bbox.shape)

df_bbox.head()


# In[22]:


study_slice_list = list(df_data['study_slice'])

bbox_list = []

for i in range(0, len(df_data)):
    
    target = df_data.loc[i, 'target']
    study_slice = df_data.loc[i, 'study_slice']
    
    if target == 1:
    
        x = df_bbox.loc[study_slice, 'x']
        y = df_bbox.loc[study_slice, 'y']
        width = df_bbox.loc[study_slice, 'width']
        height = df_bbox.loc[study_slice, 'height']

        bbox_dict ={
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }

        bbox_list.append(bbox_dict)
        
    else:
        bbox_list.append('none')
        
        

df_data['boxes'] = bbox_list


# In[23]:


df_data.head(2)


# In[24]:


df_data.loc[0, 'boxes']


# ## Helper functions

# In[25]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                         text_size=12):
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=text_size)
    plt.yticks(tick_marks, classes, fontsize=text_size)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=text_size)

    plt.ylabel('True label', fontsize=text_size)
    plt.xlabel('Predicted label', fontsize=text_size)
    plt.tight_layout()


# In[26]:


from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)

    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data




def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im


# ## Create the folds

# In[27]:


from sklearn.model_selection import KFold, StratifiedKFold

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=101)

for fold, ( _, val_) in enumerate(skf.split(X=df_data, y=df_data.target)):
      df_data.loc[val_ , "fold"] = fold
        
df_data['fold'].value_counts()


# In[28]:


for fold_index in range(0, NUM_FOLDS):
    
    df_train = df_data[df_data['fold'] != fold_index]
    df_val = df_data[df_data['fold'] == fold_index]

    print(f'\nFold {fold_index}')
    print('.........')
    print()
    print('Train shape:',df_train.shape)
    print('Val shape:',df_val.shape)
    print()
    print('Train target distribution')
    print(df_train['target'].value_counts())
    print()
    print('Val target distribution')
    print(df_val['target'].value_counts())


# In[39]:


os.chdir('/home/nvr5386/Desktop/RSNA/project_data/yolov5')


base_dir = 'base_dir'

images = os.path.join(base_dir, 'images')

# labels
labels = os.path.join(base_dir, 'labels')




train = os.path.join(images, 'train')
validation = os.path.join(images, 'validation')


train = os.path.join(labels, 'train')
validation = os.path.join(labels, 'validation')


# In[40]:


os.chdir('/home/nvr5386/Desktop/RSNA/project_data/')


# In[41]:


fold_index = CHOSEN_FOLD

df_train = df_data[df_data['fold'] != fold_index]
df_val = df_data[df_data['fold'] == fold_index]

print(df_train['target'].value_counts())
print(df_val['target'].value_counts())


# In[42]:


def process_data_for_yolo(df, data_type='train'):

    for _, row in tq.tqdm(df.iterrows(), total=len(df)):
        
        target = row['target']
        
        study_slice = row['study_slice']
        fname = study_slice + '.png'
        
        
        if target == 1:
            bbox_dict = row['boxes']
            
            bbox_list = [bbox_dict]

            image_width = row['w']
            image_height = row['h']
            
            yolo_data = []

            for coord_dict in bbox_list:

                xmin = int(coord_dict['x'])
                ymin = int(coord_dict['y'])
                bbox_w = int(coord_dict['width'])
                bbox_h = int(coord_dict['height'])

                class_id = target

                x_center = xmin + (bbox_w/2)
                y_center = ymin + (bbox_h/2)



                x_center = x_center/image_width
                y_center = y_center/image_height
                bbox_w = bbox_w/image_width
                bbox_h = bbox_h/image_height

                yolo_list = [class_id, x_center, y_center, bbox_w, bbox_h]

                yolo_data.append(yolo_list)

            yolo_data = np.array(yolo_data)

            np.savetxt(os.path.join('yolov5/base_dir', 
                        f"labels/{data_type}/{study_slice}.txt"),
                        yolo_data, 
                        fmt=["%d", "%f", "%f", "%f", "%f"]
                        ) 




        shutil.copyfile(
            f"{prep_data_path}/images_dir/{fname}",
            os.path.join('yolov5/base_dir', f"images/{data_type}/{fname}")
        )
        
        

process_data_for_yolo(df_train, data_type='train')
process_data_for_yolo(df_val, data_type='validation')


# In[43]:


get_ipython().system('ls')


# In[44]:


print(len(os.listdir('yolov5/base_dir/images/train')))
print(len(os.listdir('yolov5/base_dir/images/validation')))

print(len(os.listdir('yolov5/base_dir/labels/train')))
print(len(os.listdir('yolov5/base_dir/labels/validation')))


# In[45]:


text_file_list = os.listdir('yolov5/base_dir/labels/train')

text_file = text_file_list[0]

text_file


# In[46]:


get_ipython().system(" cat 'yolov5/base_dir/labels/train/1.2.826.0.1.3680043.26979_172.txt'")


# In[47]:


yaml_dict = {'train': 'base_dir/images/train',  
            'val': 'base_dir/images/validation',
            'nc': 2,                             
            'names': ['0', '1']}               




import yaml

with open(r'yolov5/my_data.yaml', 'w') as file:
    documents = yaml.dump(yaml_dict, file)


# In[48]:


os.listdir('yolov5')


# In[49]:


get_ipython().system(" cat 'yolov5/my_data.yaml'")


# ## Create a custom hyperameter/augmentation yaml file

# In[50]:


yaml_dict = {
    
'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
'lrf': 0.032,  # final OneCycleLR learning rate (lr0 * lrf)
'momentum': 0.937,  # SGD momentum/Adam beta1
'weight_decay': 0.0005,  # optimizer weight decay 5e-4
'warmup_epochs': 3.0,  # warmup epochs (fractions ok)
'warmup_momentum': 0.8,  # warmup initial momentum
'warmup_bias_lr': 0.1,  # warmup initial bias lr
'box': 0.1,  # box loss gain
'cls': 1.0,  # cls loss gain
'cls_pw': 0.5,  # cls BCELoss positive_weight
'obj': 2.0,  # obj loss gain (scale with pixels)
'obj_pw': 0.5,  # obj BCELoss positive_weight
'iou_t': 0.20,  # IoU training threshold
'anchor_t': 4.0,  # anchor-multiple threshold
'anchors': 0,  # anchors per output layer (0 to ignore)
'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
'hsv_h': 0,  # image HSV-Hue augmentation (fraction)
'hsv_s': 0,  # image HSV-Saturation augmentation (fraction)
'hsv_v': 0,  # image HSV-Value augmentation (fraction)
'degrees': 30.0,  # image rotation (+/- deg)
'translate': 0.2,  # image translation (+/- fraction)
'scale': 0.3,  # image scale (+/- gain)
'shear': 0.0,  # image shear (+/- deg)
'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
'flipud': 0.2,  # image flip up-down (probability)
'fliplr': 0.5,  # image flip left-right (probability)
'mosaic': 0.8,  # image mosaic (probability)
'mixup': 0.0  # image mixup (probability)
    
}


import yaml

with open(r'yolov5/my_hyp.yaml', 'w') as file:
    documents = yaml.dump(yaml_dict, file)


# In[51]:


os.listdir('yolov5')


# In[52]:


get_ipython().system(" cat 'yolov5/my_hyp.yaml'")


# ## Train the model

# In[54]:


os.chdir('/home/nvr5386/Desktop/RSNA/project_data/yolov5/')

get_ipython().system('pwd')


# In[55]:


get_ipython().system(' python train.py --img 1024 --batch 8 --epochs 2 --data my_data.yaml --cfg models/yolov5s.yaml --name my_model')

get_ipython().system('WANDB_MODE="dryrun" python train.py --img $IMAGE_SIZE --batch $BATCH_SIZE --epochs $NUM_EPOCHS --data my_data.yaml --hyp my_hyp.yaml --weights $yolo_model_path')


# In[57]:


path = '/home/nvr5386/Desktop/RSNA/project_data/yolov5/runs/train/my_model/weights/'

os.listdir(path)


# In[58]:


os.chdir('/home/nvr5386/Desktop/RSNA/project_data/')

get_ipython().system('pwd')


# In[59]:


shutil.copyfile(
    '/home/nvr5386/Desktop/RSNA/project_data/yolov5/runs/train/my_model/weights/best.pt',
    '/home/nvr5386/Desktop/RSNA/project_data/best.pt')


get_ipython().system('ls')


# In[60]:


os.chdir('/home/nvr5386/Desktop/RSNA/project_data/yolov5')

get_ipython().system('pwd')


# In[85]:


os.listdir('/home/nvr5386/Desktop/RSNA/project_data/yolov5/runs/train/')


# In[86]:


exp_list = os.listdir('/home/nvr5386/Desktop/RSNA/project_data/yolov5/runs/train/')

exp_list


# In[87]:


exp = exp_list[0]

exp


# In[88]:


os.listdir(f'/home/nvr5386/Desktop/RSNA/project_data/yolov5/runs/train/{exp}')


# | <a id='validation_review'></a>

# In[89]:


os.listdir(f'/home/nvr5386/Desktop/RSNA/project_data/yolov5/runs/train/{exp}')


# In[90]:


os.chdir('/home/nvr5386/Desktop/RSNA/project_data/yolov5')

get_ipython().system('pwd')


# ## Display the training curves

# In[91]:


plt.figure(figsize = (15, 15))
plt.imshow(plt.imread(f'runs/train/{exp}/results.png'))


# ## Get the best mAP and best epoch

# In[69]:


get_ipython().system('ls')


# In[74]:


path = f'/home/nvr5386/Desktop/RSNA/project_data/yolov5/runs/train/{exp}/results.csv'

get_ipython().system('cat $path')


# In[75]:


filename = f'/home/nvr5386/Desktop/RSNA/project_data/yolov5/runs/train/{exp}/results.csv'

file_list = []

with open(filename) as f:
    file_line_list = f.readlines()
    
    
for i in range(0, len(file_line_list)):
    
  
    line_list = file_line_list[i].split()
    
    line_list = [x.strip() for x in line_list]

    file_list.append(line_list)
    
len(file_list)


# In[76]:


df = pd.DataFrame(file_list)

df.head(10)


# In[77]:


col_names = ['epoch', 'P', 'R', 'map0.5', 'map0.5:0.95']

df_results = df[[0, 8, 9, 10, 11]]

df_results.columns = col_names

df_results.head(10)


# In[78]:


best_map = df_results['map0.5'].max()

print('---------------------')

print('Best map0.5:', best_map)
print()

df = df_results[df_results['map0.5'] == best_map]

print(df.head())

print('---------------------')


# ## Display one batch of train images

# In[82]:


plt.figure(figsize = (15, 15))
plt.imshow(plt.imread(f'/home/nvr5386/Desktop/RSNA/project_data/yolov5/runs/train/{exp}/train_batch0.jpg'))


# ## Display true and predicted val set bboxes
# 
# Here we will display the true and predicted bboxes for two val batches.

# In[92]:


plt.figure(figsize = (15, 15))
plt.imshow(plt.imread(f'/home/nvr5386/Desktop/RSNA/project_data/yolov5/runs/train/{exp}/test_batch0_labels.jpg'))


# In[84]:


plt.figure(figsize = (15, 15))
plt.imshow(plt.imread(f'/home/nvr5386/Desktop/RSNA/project_data/yolov5/runs/train/{exp}/test_batch0_pred.jpg'))


# In[ ]:


plt.figure(figsize = (15, 15))
plt.imshow(plt.imread(f'runs/train/{exp}/test_batch1_labels.jpg'))


# In[ ]:


plt.figure(figsize = (15, 15))
plt.imshow(plt.imread(f'runs/train/{exp}/test_batch1_pred.jpg'))


# In[ ]:





# ## Make a prediction on the val set

# In[93]:


os.chdir('/home/nvr5386/Desktop/RSNA/project_data/')

get_ipython().system('pwd')


# In[94]:


if os.path.isdir('yolo_images_dir') == False:
    yolo_images_dir = 'yolo_images_dir'
    os.mkdir(yolo_images_dir)
    
    
    
val_fname_list = list(df_val['study_slice'])

for study_slice in val_fname_list:
    
    fname = study_slice + '.png'

    shutil.copyfile(
        f"{prep_data_path}/images_dir/{fname}",
        f"yolo_images_dir/{fname}")
    
    
len(os.listdir('yolo_images_dir'))


# In[95]:


get_ipython().system('ls')


# In[96]:


os.chdir('/home/nvr5386/Desktop/RSNA/project_data/yolov5')

get_ipython().system('pwd')


# In[98]:


test_images_path = '/home/nvr5386/Desktop/RSNA/project_data/yolo_images_dir'
yolo_model_path = '/home/nvr5386/Desktop/RSNA/project_data/best.pt'

get_ipython().system('python detect.py --source $test_images_path --weights $yolo_model_path --img $IMAGE_SIZE --save-txt --save-conf --exist-ok')


# ## Process the predictions

# In[ ]:


txt_files_list = os.listdir('runs/detect/exp/labels')

print(len(txt_files_list))
print(txt_files_list[0])


# In[ ]:


txt_files_list = os.listdir('runs/detect/exp/labels')

for i, txt_file in enumerate(txt_files_list):
    
    path = f'runs/detect/exp/labels/{txt_file}'
    
    cols = ['class', 'x-center', 'y-center', 'bbox_width', 'bbox_height', 'conf-score']

    df = pd.read_csv(path, sep=" ", header=None)
    
    df.columns = cols

    fname = txt_file.replace("txt", "png")
    
    df['id'] = fname
 
    if i == 0:
        
        df_test_preds = df
    else:
        
        df_test_preds = pd.concat([df_test_preds, df], axis=0)
       
    
    
print(len(txt_files_list))
print(df_test_preds['id'].nunique())
print(df_test_preds.shape)

df_test_preds.head()


# In[ ]:


df_val = df_val.reset_index(drop=True)

df_val['id'] =  df_val['study_slice'] + '.png'

val_pred_list = []

pred_list = list(df_test_preds['id'])

for i in range(0, len(df_val)):

    fname = df_val.loc[i, 'id']
    
    if fname in pred_list:
        
        val_pred_list.append(1)
    else:
        val_pred_list.append(0)
    
    
df_val['preds'] = val_pred_list

df_val['preds'].value_counts()


# ## Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix

CLASS_LIST = ['Normal', 'Fracture']
    
y_true = list(df_val['target'])

y_pred = list(df_val['preds'])

cm = confusion_matrix(y_true, y_pred)

print()
print(cm)
print(CLASS_LIST)


# In[ ]:


cm_plot_labels = ['normal', 'fracture']


text_size=12

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix', text_size=text_size)


# ## Classification Report

# In[ ]:


from sklearn.metrics import classification_report
    
report = classification_report(y_true, y_pred, target_names=cm_plot_labels)

print()
print(report)


# In[ ]:


if os.path.isdir('base_dir') == True:
    shutil.rmtree('base_dir')


# In[ ]:


if os.path.isdir('images_dir') == True:
    shutil.rmtree('images_dir')
    
if os.path.isdir('val_images_dir') == True:
    shutil.rmtree('val_images_dir')


# In[ ]:


get_ipython().system('ls')


# In[ ]:




