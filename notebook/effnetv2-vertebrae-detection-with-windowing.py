#!/usr/bin/env python
# coding: utf-8

# In[24]:


import gc
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torchvision as tv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm.notebook import tqdm

import wandb

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


# Effnet
WEIGHTS = tv.models.efficientnet.EfficientNet_V2_S_Weights.DEFAULT

TRAIN_IMAGES_PATH = '../../train_images'
TEST_IMAGES_PATH = '../../test_images'
METADATA_PATH = '../../'
EFFNET_CHECKPOINTS_PATH = '../../vertebrae-detection-checkpoints'

EFFNET_MAX_TRAIN_BATCHES = 10000
EFFNET_MAX_EVAL_BATCHES = 1000
ONE_CYCLE_MAX_LR = 0.0004
ONE_CYCLE_PCT_START = 0.3
SAVE_CHECKPOINT_EVERY_STEP = 500
N_MODELS_FOR_INFERENCE = 2


# Common
# os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_MODE"] = "online"

if os.environ["WANDB_MODE"] == "online":
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ['WANDB_API_KEY'] = UserSecretsClient().get_secret("WANDB_API_KEY")
    except:
        #  We are running on a laptop
        print('running locally')
        TRAIN_IMAGES_PATH = '../../data/RSNA/train_images'
        TEST_IMAGES_PATH = '../../rsna2022/test_images'
        EFFNET_CHECKPOINTS_PATH = '../../data/RSNA/seg_checkpoints'
        METADATA_PATH = '../../'
        os.environ['WANDB_API_KEY'] = '95e4d37f7a7583a07e09e923de3bdad15dc00589'

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 2
N_FOLDS = 5


# In[38]:


# Function to take care of teh translation and windowing. 
def window_image(img, window_center,window_width, intercept, slope, rescale=True):
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
    if rescale: 
        img = (img - img_min) / (img_max - img_min)*255.0 
    img = img.astype(np.uint8)
    return img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == dicom.multival.MultiValue: return int(x[0])
    else: return int(x)
    
def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


# In[40]:


def load_dicom(path):
   
    data=dicom.dcmread(path)
    data.PhotometricInterpretation = 'YBR_FULL'
    image = data.pixel_array
    window_center , window_width, intercept, slope = get_windowing(data)
    output = window_image(image, window_center, window_width, intercept, slope, rescale = False)
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR), image


im, meta = load_dicom(f'{TRAIN_IMAGES_PATH}/1.2.826.0.1.3680043.10001/1.dcm')
plt.figure()
plt.imshow(im)
plt.title('regular image')


# In[ ]:


import pandas as pd
df_seg = pd.read_csv(f'{METADATA_PATH}/meta_segmentation.csv')

split = GroupKFold(N_FOLDS)
for k, (train_idx, test_idx) in enumerate(split.split(df_seg, groups=df_seg.StudyInstanceUID)):
    df_seg.loc[test_idx, 'split'] = k

split = KFold(N_FOLDS)
for k, (train_idx, test_idx) in enumerate(split.split(df_seg)):
    df_seg.loc[test_idx, 'random_split'] = k

slice_max_seg = df_seg.groupby('StudyInstanceUID')['Slice'].max().to_dict()
df_seg['SliceRatio'] = 0
df_seg['SliceRatio'] = df_seg['Slice'] / df_seg['StudyInstanceUID'].map(slice_max_seg)

df_seg.sample(10)


# In[ ]:


class VertebraeSegmentDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms

    def __getitem__(self, i):
        path = os.path.join(self.path, self.df.iloc[i].StudyInstanceUID, f'{self.df.iloc[i].Slice}.dcm')
        try:
            img = load_dicom(path)[0]
            img = np.transpose(img, (2, 0, 1))  # Pytorch uses (batch, channel, height, width) order. Converting (height, width, channel) -> (channel, height, width)
            if self.transforms is not None:
                img = self.transforms(torch.as_tensor(img))
        except Exception as ex:
            print(ex)
            return None

        if 'C1' in self.df.columns:
            vert_targets = torch.as_tensor(self.df.iloc[i][['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']].astype('float32').values)
            return img, vert_targets
        return img

    def __len__(self):
        return len(self.df)


ds_seg = VertebraeSegmentDataSet(df_seg, TRAIN_IMAGES_PATH, WEIGHTS.transforms())
X, y = ds_seg[300]
X.shape, y.shape


# In[ ]:


class SegEffnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = tv.models.efficientnet_v2_s(weights=WEIGHTS)
        self.model = create_feature_extractor(effnet, ['flatten'])
        self.nn_vertebrae = torch.nn.Sequential(
            torch.nn.Linear(1280, 7),
        )

    def forward(self, x):
        # returns logits
        x = self.model(x)['flatten']
        return self.nn_vertebrae(x)

    def predict(self, x):
        pred = self.forward(x)
        return torch.sigmoid(pred)

# quick test
model = SegEffnetModel()
model.predict(torch.randn(1, 3, 512, 512))
del model


# In[ ]:


def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()


# In[ ]:


def filter_nones(b):
    return torch.utils.data.default_collate([v for v in b if v is not None])


# In[ ]:


def save_model(name, model, optim, scheduler):
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler
    }, f'{name}.tph')

def load_model(model, name, path='.'):
    data = torch.load(os.path.join(path, f'{name}.tph'), map_location=DEVICE)
    model.load_state_dict(data['model'])
    optim = torch.optim.Adam(model.parameters())
    optim.load_state_dict(data['optim'])
    return model, optim, data['scheduler']

# quick test
model = torch.nn.Linear(2, 1)
optim = torch.optim.Adam(model.parameters())
save_model('testmodel', model, optim, None)

model1, optim1, scheduler1 = load_model(torch.nn.Linear(2, 1), 'testmodel')
assert torch.all(next(iter(model1.parameters())) == next(iter(model.parameters()))).item(), "Loading/saving is inconsistent!"


# In[ ]:


def evaluate_segeffnet(model: SegEffnetModel, ds, max_batches=1e9, shuffle=False):
    torch.manual_seed(42)
    model = model.to(DEVICE)
    dl_test = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=os.cpu_count(), collate_fn=filter_nones)
    with torch.no_grad():
        model.eval()
        pred = []
        y = []
        progress = tqdm(dl_test, desc='Eval', miniters=100)
        for i, (X, y_vert) in enumerate(progress):
            with autocast():
                y_vert_pred = model.predict(X.to(DEVICE))
            pred.append(y_vert_pred.cpu().numpy())
            y.append(y_vert.numpy())
            acc = np.mean(np.mean((pred[-1] > 0.5) == y[-1], axis=0))
            progress.set_description(f'Eval acc: {acc:.02f}')
            if i >= max_batches:
                break
        pred = np.concatenate(pred)
        y = np.concatenate(y)
        acc = np.mean(np.mean((pred > 0.5) == y, axis=0))
        return acc, pred


# In[ ]:


get_ipython().run_cell_magic('wandb', '', "\n\ndef train_segeffnet(ds_train, ds_eval, logger, name):\n    torch.manual_seed(42)\n    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), collate_fn=filter_nones)\n\n\n    model = SegEffnetModel().to(DEVICE)\n    optim = torch.optim.Adam(model.parameters())\n    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=ONE_CYCLE_MAX_LR, epochs=1, steps_per_epoch=min(EFFNET_MAX_TRAIN_BATCHES, len(dl_train)), pct_start=ONE_CYCLE_PCT_START)\n    model.train()\n    scaler = GradScaler()\n\n    progress = tqdm(dl_train, desc='Train', miniters=10)\n    for batch_idx, (X,  y_vert) in enumerate(progress):\n\n        if batch_idx % SAVE_CHECKPOINT_EVERY_STEP == 0 and EFFNET_MAX_EVAL_BATCHES > 0:\n            eval_loss = evaluate_segeffnet(model, ds_eval, max_batches=EFFNET_MAX_EVAL_BATCHES, shuffle=True)[0]\n            model.train()\n            if logger is not None:\n                logger.log({'eval_acc': eval_loss})\n            if batch_idx > 0:  # don't save untrained model\n                save_model(name, model, optim, scheduler)\n\n        if batch_idx >= EFFNET_MAX_TRAIN_BATCHES:\n            break\n\n        optim.zero_grad()\n        with autocast():\n            y_vert_pred = model.forward(X.to(DEVICE))\n            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_vert_pred, y_vert.to(DEVICE))\n\n            if np.isinf(loss.item()) or np.isnan(loss.item()):\n                print(f'Bad loss, skipping the batch {batch_idx}')\n                del y_vert_pred, loss\n                gc_collect()\n                continue\n\n        scaler.scale(loss).backward()\n        scaler.step(optim)\n        scaler.update()\n        scheduler.step()\n\n        progress.set_description(f'Train loss: {loss.item():.02f}')\n        if logger is not None:\n            logger.log({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})\n\n\n    eval_loss = evaluate_segeffnet(model, ds_eval, max_batches=EFFNET_MAX_EVAL_BATCHES, shuffle=True)[0]\n    if logger is not None:\n        logger.log({'eval_acc': eval_loss})\n\n    save_model(name, model, optim, scheduler)\n    return model\n\n\nseg_models = []\nfor fold in range(N_FOLDS):\n    fname = os.path.join(f'{EFFNET_CHECKPOINTS_PATH}/segeffnetv2-f{fold}.tph')\n    if os.path.exists(fname):\n        print(f'Found cached model {fname}')\n        seg_models.append(load_model(SegEffnetModel(), f'segeffnetv2-f{fold}', EFFNET_CHECKPOINTS_PATH)[0].to(DEVICE))\n    else:\n        with wandb.init(project='RSNA-2022', name=f'SegEffNet-v2-fold{fold}') as run:\n            gc_collect()\n            ds_train = VertebraeSegmentDataSet(df_seg.query('split != @fold'), TRAIN_IMAGES_PATH, WEIGHTS.transforms())\n            ds_eval = VertebraeSegmentDataSet(df_seg.query('split == @fold'), TRAIN_IMAGES_PATH, WEIGHTS.transforms())\n            train_segeffnet(ds_train, ds_eval, run, f'segeffnetv2-f{fold}')\n")


# In[ ]:


for fold in range(N_FOLDS):
    fname = os.path.join(f'./segeffnetv2-f{fold}.tph')
    if os.path.exists(fname):
        print(f'Found cached model {fname}')
        seg_models.append(load_model(SegEffnetModel(), f'segeffnetv2-f{fold}', './')[0].to(DEVICE))
print(seg_models)


# In[ ]:


seg_models


# In[ ]:


with tqdm(seg_models, desc='Fold') as progress:
    for fold, model in enumerate(progress):
        ds = VertebraeSegmentDataSet(df_seg.query('split == @fold'), TRAIN_IMAGES_PATH, WEIGHTS.transforms())
        acc, pred = evaluate_segeffnet(model, ds, max_batches=1e9, shuffle=False)
        df_seg.loc[df_seg[df_seg.split == fold].index, ['C1_pred', 'C2_pred', 'C3_pred', 'C4_pred', 'C5_pred', 'C6_pred', 'C7_pred']] = pred
        progress.set_description(f'Acc: {acc}')


# In[ ]:


with tqdm(seg_models, desc='Fold') as progress:
    for fold, model in enumerate(progress):
        print(fold)


# In[ ]:


acc = (df_seg[[f'C{i}_pred' for i in range(1, 8)]] > 0.5).values == (df_seg[[f'C{i}' for i in range(1, 8)]] > 0.5).values
print('Effnetv2 accuracy per vertebrae', np.mean(acc, axis=0))
print('Effnetv2 average accuracy', np.mean(np.mean(acc, axis=0)))


# In[ ]:


def predict_vertebrae(df, seg_models: List[SegEffnetModel]):
    df = df.copy()
    ds = VertebraeSegmentDataSet(df, TRAIN_IMAGES_PATH, WEIGHTS.transforms())
    dl_test = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), collate_fn=filter_nones)
    predictions = []
    with torch.no_grad():
        with tqdm(dl_test, desc='Eval', miniters=10) as progress:
            for i, X in enumerate(progress):
                with autocast():
                    pred = torch.zeros(len(X), 7).to(DEVICE)
                    for model in seg_models:
                        pred += model.predict(X.to(DEVICE)) / len(seg_models)
                    predictions.append(pred)
    predictions = torch.concat(predictions).cpu().numpy()
    return predictions


# In[ ]:


df_train = pd.read_csv(os.path.join(METADATA_PATH, 'meta_train_clean.csv'))


# In[ ]:


for uid in np.random.choice(df_train.StudyInstanceUID, 3):
    pred = predict_vertebrae(df_train.query('StudyInstanceUID == @uid'), seg_models[:2])
    plt.figure(figsize=(20, 5))
    plt.plot(pred)
    plt.title(f'Vertebrae prediction by slice for UID: {uid}')


# In[ ]:


pred = predict_vertebrae(df_train, seg_models[:N_MODELS_FOR_INFERENCE])


# In[ ]:


df_train[[f'C{i}' for i in range(1, 8)]] = pred
df_train.to_csv('train_segmented.csv', index=False)

