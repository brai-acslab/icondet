
import numpy as np
import pandas as pd

import torch
import torchvision

from torchvision import transforms, datasets
from torch.utils.data import Dataset

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

from tqdm import tqdm_notebook as tqdm

import random
import time
import sys
import os
import math

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.utils import shuffle

from efficientnet_pytorch import EfficientNet
import torch.nn as nn

pd.set_option('display.max_columns', None)
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

import shutil
import os

def split_dataset_into_3(path_to_dataset, train_ratio, valid_ratio):
    """
    wrapper function to split the dataset into three subsets(test,validation,train)
    :param path_to_dataset:
    :param train_ratio:
    :param valid_ratio:
    :return:
    """
    _, sub_dirs, _ = next(iter(os.walk(path_to_dataset)))  # retrieve name of subdirectories
    sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]  # list for counting items in each sub directory(class)

    # directories where the splitted dataset will lie
    dir_train = os.path.join(os.path.dirname(path_to_dataset), 'train')
    dir_valid = os.path.join(os.path.dirname(path_to_dataset), 'validation')
    dir_test = os.path.join(os.path.dirname(path_to_dataset), 'test')

    for i, sub_dir in enumerate(sub_dirs):

        dir_train_dst = os.path.join(dir_train, sub_dir)  # directory for destination of train dataset
        dir_valid_dst = os.path.join(dir_valid, sub_dir)  # directory for destination of validation dataset
        dir_test_dst = os.path.join(dir_test, sub_dir)  # directory for destination of test dataset
        
        print(dir_train_dst)
        print(dir_valid_dst)
        print(dir_test_dst)

        # variables to save the sub directory name(class name) and to count the images of each sub directory(class)
        class_name = sub_dir
        sub_dir = os.path.join(path_to_dataset, sub_dir)
        sub_dir_item_cnt[i] = len(os.listdir(sub_dir))
        print(sub_dir)
        print(sub_dir_item_cnt[i])

        items = os.listdir(sub_dir)

        # transfer data to trainset
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio)):
            if not os.path.exists(dir_train_dst):
                os.makedirs(dir_train_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_train_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to validation
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio) + 1,
                              round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio))):
            if not os.path.exists(dir_valid_dst):
                os.makedirs(dir_valid_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_valid_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to testset
        for item_idx in range(round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio)) + 1, sub_dir_item_cnt[i]):
            if not os.path.exists(dir_test_dst):
                os.makedirs(dir_test_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_test_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

    return

split_dataset_into_3('/home/thioketone/prateeti_conjunctivitis/dataset', 0.9, 0.1)#only generating train and validation sets


from sklearn.utils import shuffle

def annotate(BASE_PATH):
    image=[]
    labels=[]
    encoded_labels=[]
    for file in os.listdir(BASE_PATH):
        if file=='healthy':
            for c in os.listdir(os.path.join(BASE_PATH, file)):
                if c!='annotations':
                    image.append(c)
                    labels.append(0)
                    encoded_labels.append('healthy')
        if file=='infected':
            for c in os.listdir(os.path.join(BASE_PATH, file)):
                if c!='annotations':
                    image.append(c)
                    labels.append(1)
                    encoded_labels.append('infected')
    data = {'Images':image, 'Labels':labels, 'Encoded_labels':encoded_labels} 
    data = pd.DataFrame(data)
    return data

    
train_df = annotate('/home/thioketone/prateeti_conjunctivitis/train')
train_df = shuffle(train_df)
test_df = annotate('/home/thioketone/prateeti_conjunctivitis/validation')
test_df = shuffle(test_df)

train_df.head()

    
def prepare_image(path, image_size = 256):

    # import
    image = cv2.imread(path)
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    #cv2.imshow("Img",image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
    # resize
    image = cv2.resize(image, (int(image_size), int(image_size)))

    # convert to tensor    
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image
   
# dataset
class EyeData(Dataset):
    
    # initialize
    def __init__(self, data, directory, transform = None):
        self.data      = data
        self.directory = directory
        self.transform = transform
        
    # length
    def __len__(self):
        return len(self.data)
    
    # get items    
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.loc[idx, 'Encoded_labels'], self.data.loc[idx, 'Images'])
        print(img_name)
        #print(self.data.loc[idx, 'Labels'])
        image    = prepare_image(img_name)  
        image    = self.transform(image)
        label    = torch.tensor(self.data.loc[idx, 'Labels'])
        return {'image': image, 'label': label}
    
    
##### EXAMINE SAMPLE BATCH

# transformations
sample_trans = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    ])

# dataset
sample = EyeData(data       = train_df, 
                 directory  = '/home/thioketone/prateeti_conjunctivitis/train',
                 transform  = sample_trans)

print(sample)
# data loader
sample_loader = torch.utils.data.DataLoader(dataset     = sample, 
                                            batch_size  = 10, 
                                            shuffle     = True, 
                                            num_workers = 4)

# display images
print(sample_loader)

for batch_i, data in enumerate(sample_loader):

    # extract data
    inputs = data['image']
    labels = data['label']
    
    
    # create plot
    fig = plt.figure(figsize = (15, 7))
    for i in range(len(labels)):
        print(labels[i])
        ax = fig.add_subplot(2, len(labels)/2, i + 1, xticks = [], yticks = [])     
        plt.imshow(inputs[i].numpy().transpose(1, 2, 0))
        ax.set_title(labels.numpy()[i])
        plt.savefig("original.pdf", bbox_inches='tight')

    break


def prepare_image(path, sigmaX = 10, do_random_crop = False):
    
    # import image
    image = cv2.imread(path)
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    redMask = cv2.inRange(hsv, (255, 0, 0), (255, 153, 255))
    image[redMask == 255] = (0, 255, 0)
    
    # perform smart crops
    image = crop_black(image, tol = 7)
    if do_random_crop == True:
        image = random_crop(image, size = (0.9, 1))
    
    # resize and color
    image = cv2.resize(image, (int(image_size), int(image_size)))
    #image = cv2.GaussianBlur(image, (7, 7), 0)
    
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    
    # circular crop
    image = circle_crop(image, sigmaX = sigmaX)
    

    # convert to tensor    
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image


### automatic crop of black areas
def crop_black(img, tol = 7):
      
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        
        if (check_shape == 0): 
            return img 
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img  = np.stack([img1, img2, img3], axis = -1)
            return img
        
        
### circular crop around center
def circle_crop(img, sigmaX = 10):   
        
    height, width, depth = img.shape
    
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape
    
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r-2), 1, thickness = -1)
      
    img = cv2.bitwise_and(img, img, mask = circle_img)
    return img 


### random crop
def random_crop(img, size = (0.9, 1)):

    height, width, depth = img.shape
    
    cut = 1 - random.uniform(size[0], size[1])
    
    i = random.randint(0, int(cut * height))
    j = random.randint(0, int(cut * width))
    h = i + int((1 - cut) * height)
    w = j + int((1 - cut) * width)

    img = img[i:h, j:w, :]    
    
    return img


def prepare_image_alt(path, sigmaX = 10, do_random_crop = False):
    
    # import image
    image = cv2.imread(path)
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)
    full_mask = upper_mask+lower_mask
    image = cv2.bitwise_and(image, image, mask=full_mask)
    if do_random_crop == True:
        image = random_crop(image, size = (0.9, 1))
    
    # resize and color
    resize = int(image_size)
    image = cv2.resize(image, (resize,resize))
    image = cv2.GaussianBlur(image, (7, 7), 0)
    
    # circular crop
    image = circle_crop(image, sigmaX = sigmaX)
    
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    # convert to tensor    
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image
   


### image preprocessing function
def prepare_image_altmod(path, sigmaX = 10, do_random_crop = False):
    
    # import image
    image = cv2.imread(path)
    # mask red
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([155,25,0])
    upper = np.array([179,255,255])
    mask = cv2.inRange(image, lower, upper)
    result = cv2.bitwise_and(result, result, mask=mask)
    # mask red end
    
    alpha = 1.0 # Contrast control (1.0-3.0)
    beta = 60 # Brightness control (0-100)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # perform smart crops
    if do_random_crop == True:
        image = random_crop(image, size = (0.9, 1))
    
    # resize and color
    resize = int(image_size)
    image = cv2.resize(image, (resize,resize))
    image = cv2.GaussianBlur(image, (7, 7), 0)
    
    # circular crop
    image = circle_crop(image, sigmaX = sigmaX)

    # convert to tensor    
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image

#collapse-show

### image preprocessing function
def prepare_image_altmodnew(path, sigmaX = 10, do_random_crop = False):
    
    # import image
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # perform smart crops
    image = crop_black(image, tol = 7)
    if do_random_crop == True:
        image = random_crop(image, size = (0.9, 1))
    
    # resize and color
    image = cv2.resize(image, (int(image_size), int(image_size)))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    
    # circular crop
    image = circle_crop(image, sigmaX = sigmaX)

    # convert to tensor    
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image


#collapse-show

##### DATASET
    
# dataset class
class EyeData(Dataset):

    # initialize
    def __init__(self, data, directory, transform = None):
        self.data      = data
        self.directory = directory
        self.transform = transform
        
    # length
    def __len__(self):
        return len(self.data)
    
    # get items    
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.loc[idx, 'Encoded_labels'], self.data.loc[idx, 'Images'])
        image    = prepare_image(img_name)  
        image    = self.transform(image)
        label    = torch.tensor(self.data.loc[idx, 'Labels'])
        return {'image': image, 'label': label}


##### EXAMINE SAMPLE BATCH

image_size = 256


# transformations
sample_trans = transforms.Compose([transforms.ToPILImage(),
                                   transforms.ToTensor(),
                                  ])

# dataset
sample = EyeData(data       = train_df, 
                 directory  = '/home/thioketone/prateeti_conjunctivitis/train',
                 transform  = sample_trans)

# data loader
sample_loader = torch.utils.data.DataLoader(dataset     = sample, 
                                            batch_size  = 10, 
                                            shuffle     = True)

# display images
for batch_i, data in enumerate(sample_loader):

    # extract data
    inputs = data['image']
    labels = data['label']
    
    # create plot
    fig = plt.figure(figsize = (15, 7))
    for i in range(len(labels)):
        ax = fig.add_subplot(2, len(labels)/2, i + 1, xticks = [], yticks = [])     
        plt.imshow(inputs[i].numpy().transpose(1, 2, 0))
        ax.set_title(labels.numpy()[i])
    plt.savefig("transform_6.pdf", bbox_inches='tight')
    break
    

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU...')
    device = torch.device('cpu')
else:
    print('CUDA is available. Training on GPU...')
    device = torch.device('cuda:0')

# set seed
def seed_everything(seed = 23):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed = 23
seed_everything(seed)




# import data
train = train_df
test = test_df

# check shape
print(train.shape, test.shape)
print('-' * 15)
print(train['Labels'].value_counts(normalize = True))
print('-' * 15)
print(test['Labels'].value_counts(normalize = True))



class EyeTrainData(Dataset):
    
    # initialize
    def __init__(self, data, directory, transform = None):
        self.data      = data
        self.directory = directory
        self.transform = transform
        
    # length
    def __len__(self):
        return len(self.data)
    
    # get items    
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.loc[idx, 'Encoded_labels'], self.data.loc[idx, 'Images'])
        image    = prepare_image(img_name, do_random_crop = True)
        image    = self.transform(image)
        label    = torch.tensor(self.data.loc[idx, 'Labels'])
        return {'image': image, 'label': label}
    
    
# dataset class: test
class EyeTestData(Dataset):
    
    # initialize
    def __init__(self, data, directory, transform = None):
        self.data      = data
        self.directory = directory
        self.transform = transform
        
    # length
    def __len__(self):
        return len(self.data)
    
    # get items    
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.loc[idx, 'Encoded_labels'], self.data.loc[idx, 'Images'])
        image    = prepare_image(img_name, do_random_crop = False)
        image    = self.transform(image)
        label    = torch.tensor(self.data.loc[idx, 'Labels'])
        return {'image': image, 'label': label}



batch_size = 20
image_size = 256

# train transformations
train_trans = transforms.Compose([transforms.ToPILImage(),
                                  transforms.RandomRotation((-360, 360)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor()
                                 ])

# validation transformations
valid_trans = transforms.Compose([transforms.ToPILImage(),
                                  transforms.ToTensor(),
                                 ])

# create datasets
train_dataset = EyeTrainData(data      = train, 
                             directory = '/home/thioketone/prateeti_conjunctivitis/train',
                             transform = train_trans)
valid_dataset = EyeTestData(data       = test, 
                            directory  = '/home/thioketone/prateeti_conjunctivitis/validation',
                            transform  = valid_trans)

# create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size  = batch_size, 
                                           shuffle     = True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                                           batch_size  = batch_size, 
                                           shuffle     = False)
    
# model name
model_name = 'enet_b4'

# initialization function
def init_model(train = True):
    
    ### training mode
    if train == True:
                 
        # load pre-trained model
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 2)
      
        
    ### inference mode
    if train == False:
        
        # load pre-trained model
        model = EfficientNet.from_name('efficientnet-b4')
        model._fc = nn.Linear(model._fc.in_features, 5)

        # freeze  layers
        for param in model.parameters():
            param.requires_grad = False
            
    return model


model = init_model()
print(model)
           
# loss function
criterion = nn.CrossEntropyLoss()
   
# epochs
max_epochs = 15
early_stop = 5
   
# learning rates
eta = 1e-3

# scheduler
step  = 5
gamma = 0.5

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = eta)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step, gamma = gamma)

# initialize model and send to GPU
model = init_model()
model = model.to(device)


#collapse-show
data_valid = test
data_train = train
# placeholders
oof_preds = np.zeros((len(test), 2))
val_fracs = []
val_kappas = []
val_losses = []
trn_losses = []
bad_epochs = 0

# timer
cv_start = time.time()

# training and validation loop
for epoch in range(max_epochs):

    ##### PREPARATION
    # timer
    epoch_start = time.time()

    # reset losses
    trn_loss = 0.0
    val_loss = 0.0

    # placeholders
    fold_preds = np.zeros((len(data_valid), 2))


    ##### TRAINING

    # switch regime
    model.train()

    # loop through batches
    for batch_i, data in enumerate(train_loader):

        # extract inputs and labels
        inputs = data['image']
        labels = data['label'].view(-1)
        inputs = inputs.to(device, dtype = torch.float)
        labels = labels.to(device, dtype = torch.long)
        optimizer.zero_grad()

        # forward and backward pass
        with torch.set_grad_enabled(True):
            preds = model(inputs)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()

        # compute loss
        trn_loss += loss.item() * inputs.size(0)
        
        
    ##### INFERENCE

     # switch regime
    model.eval()
    
    # loop through batches
    for batch_i, data in enumerate(valid_loader):
        
        # extract inputs and labels
        inputs = data['image']
        labels = data['label'].view(-1)
        inputs = inputs.to(device, dtype = torch.float)
        labels = labels.to(device, dtype = torch.long)

        # compute predictions
        with torch.set_grad_enabled(False):
            preds = torch.softmax(model(inputs), 1).detach()
            fold_preds[batch_i * batch_size:(batch_i + 1) * batch_size, :] = preds.cpu().numpy()

        # compute loss
        loss      = criterion(preds, labels)
        val_loss += loss.item() * inputs.size(0)
        
    # save predictions
    oof_preds = fold_preds


    ##### EVALUATION

    # evaluate performance
    fold_preds_round = fold_preds.argmax(axis = 1)
    val_kappa = metrics.cohen_kappa_score(data_valid['Labels'], fold_preds_round.astype('int'), weights = 'quadratic')
    val_frac = metrics.accuracy_score(data_valid['Labels'], fold_preds_round)

    # save perfoirmance values
    val_kappas.append(val_kappa)
    val_fracs.append(val_frac)
    val_losses.append(val_loss / len(data_valid))
    trn_losses.append(trn_loss / len(data_train))


    ##### EARLY STOPPING

    # display info
    print('- epoch {}/{} | lr = {} | trn_loss = {:.4f} | val_loss = {:.4f} | val_kappa = {:.4f} | accu = {:.4f} | {:.2f} min'.format(
        epoch + 1, max_epochs, scheduler.get_lr()[len(scheduler.get_lr()) - 1],
        trn_loss / len(data_train), val_loss / len(data_valid), val_kappa, val_frac,
        (time.time() - epoch_start) / 60))

    # check if there is any improvement
    if epoch > 0:       
        if val_kappas[epoch] < val_kappas[epoch - bad_epochs - 1]:
            bad_epochs += 1
        else:
            bad_epochs = 0

    # save model weights if improvement
    if bad_epochs == 0:
        oof_preds_best = oof_preds.copy()
        torch.save(model.state_dict(), '../model_{}.bin'.format(model_name))

    # break if early stop
    if bad_epochs == early_stop:
        print('Early stopping. Best results: loss = {:.4f}, kappa = {:.4f} (epoch {})'.format(
            np.min(val_losses), val_kappas[np.argmin(val_losses)], np.argmin(val_losses) + 1))
        print('')
        break

    # break if max epochs
    if epoch == (max_epochs - 1):
        print('Did not met early stopping. Best results: loss = {:.4f}, kappa = {:.4f} (epoch {})'.format(
            np.min(val_losses), val_kappas[np.argmin(val_losses)], np.argmin(val_losses) + 1))
        print('')
        break
        
# load best predictions
oof_preds = oof_preds_best

# print performance
print('')
print('Finished in {:.2f} minutes'.format((time.time() - cv_start) / 60))


torch.save({
	'accuracy': val_fracs,
	'kappas': val_kappas,
	'losses': val_losses,
	'trainlosses': trn_losses	
	}, 'results.file')


def init_model_tuned(train = True, trn_layers = 2):
    
    ### training mode
    if train == True:
                 
        # load pre-trained model
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 2)
        model.load_state_dict(torch.load('../models/model_{}.bin'.format(model_name, 1)))   
        
        # freeze first layers
        for child in list(model.children())[:-trn_layers]:
            for param in child.parameters():
                param.requires_grad = False
        
        
    ### inference mode
    if train == False:
        
        # load pre-trained model
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 2)
        model.load_state_dict(torch.load('../models/model_{}.bin'.format(model_name, 1)))   

        # freeze all layers
        for param in model.parameters():
            param.requires_grad = False
            
    return model

model = init_model()

# no. folds
num_folds = 4

# creating splits
skf    = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = seed)
splits = list(skf.split(train['id_code'], train['diagnosis']))

# placeholders
oof_preds = np.zeros((len(train), 1))

# timer
cv_start = time.time()


##### PARAMETERS

# loss function
criterion = nn.CrossEntropyLoss()

# epochs
max_epochs = 15
early_stop = 5

# learning rates
eta = 1e-3

# scheduler
step  = 5
gamma = 0.5


##### CROSS-VALIDATION LOOP
for fold in tqdm(range(num_folds)):
    
    ####### DATA PREPARATION

    # display information
    print('-' * 30)
    print('FOLD {}/{}'.format(fold + 1, num_folds))
    print('-' * 30)

    # load splits
    data_train = train.iloc[splits[fold][0]].reset_index(drop = True)
    data_valid = train.iloc[splits[fold][1]].reset_index(drop = True)

    # create datasets
    train_dataset = EyeTrainData(data      = data_train, 
                                 directory = '/home/thioketone/prateeti_conjunctivitis/train',
                                 transform = train_trans)
    valid_dataset = EyeTrainData(data      = data_valid, 
                                 directory = '/home/thioketone/prateeti_conjunctivitis/validation',
                                 transform = valid_trans)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size  = batch_size, 
                                               shuffle     = True, 
                                               num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                                               batch_size  = batch_size, 
                                               shuffle     = False, 
                                               num_workers = 4)
    
    
    ####### MODEL PREPARATION
    
    # placeholders
    val_kappas = []
    val_losses = []
    trn_losses = []
    bad_epochs = 0
    
    # load best OOF predictions
    if fold > 0:
        oof_preds = oof_preds_best.copy()
    
    # initialize and send to GPU
    model = init_model(train = True)
    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model._fc.parameters(), lr = eta)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = step, gamma = gamma)
    
    
    ####### TRAINING AND VALIDATION LOOP
    for epoch in range(max_epochs):

        ##### PREPARATION

        # timer
        epoch_start = time.time()

        # reset losses
        trn_loss = 0.0
        val_loss = 0.0

        # placeholders
        fold_preds = np.zeros((len(data_valid), 1))


        ##### TRAINING

        # switch regime
        model.train()
        
        # loop through batches
        for batch_i, data in enumerate(train_loader):

            # extract inputs and labels
            inputs = data['image']
            labels = data['label'].view(-1)
            inputs = inputs.to(device, dtype = torch.float)
            labels = labels.to(device, dtype = torch.long)
            optimizer.zero_grad()

            # forward and backward pass
            with torch.set_grad_enabled(True):
                preds = model(inputs)
                loss  = criterion(preds, labels)
                loss.backward()
                optimizer.step()

            # compute loss
            trn_loss += loss.item() * inputs.size(0)


        ##### INFERENCE
        
        # initialize
        model.eval()

        # loop through batches
        for batch_i, data in enumerate(valid_loader):

            # extract inputs and labels
            inputs = data['image']
            labels = data['label'].view(-1)
            inputs = inputs.to(device, dtype = torch.float)
            labels = labels.to(device, dtype = torch.long)

            # predictions
            with torch.set_grad_enabled(False):
                preds = model(inputs).detach()
                _, class_preds = preds.topk(1)
                fold_preds[batch_i * batch_size:(batch_i + 1) * batch_size, :] = class_preds.cpu().numpy()

            # loss
            loss      = criterion(preds, labels)
            val_loss += loss.item() * inputs.size(0)

        # save predictions
        oof_preds[splits[fold][1]] = fold_preds
        
        # scheduler step
        scheduler.step()


        ##### EVALUATION

        # evaluate performance
        fold_preds_round = fold_preds
        val_kappa = metrics.cohen_kappa_score(data_valid['Labels'], fold_preds_round.astype('int'), weights = 'quadratic')
        
        val_kappas.append(val_kappa)
        val_losses.append(val_loss / len(data_valid))
        trn_losses.append(trn_loss / len(data_train))

        
        ##### EARLY STOPPING
        
        # display info
        print('- epoch {}/{} | lr = {} | trn_loss = {:.4f} | val_loss = {:.4f} | val_kappa = {:.4f} | {:.2f} min'.format(
            epoch + 1, max_epochs, scheduler.get_lr()[len(scheduler.get_lr()) - 1],
            trn_loss / len(data_train), val_loss / len(data_valid), val_kappa,
            (time.time() - epoch_start) / 60))
        
        # check improvement
        if epoch > 0:       
            if val_kappas[epoch] < val_kappas[epoch - bad_epochs - 1]:
                bad_epochs += 1
            else:
                bad_epochs = 0

        # save model weights 
        if bad_epochs == 0:
            oof_preds_best = oof_preds.copy()
            torch.save(model.state_dict(), '../models/model_{}_fold{}.bin'.format(model_name, fold + 1))
                          
        # break if early stop
        if bad_epochs == early_stop:
            print('Early stopping. Best results: loss = {:.4f}, kappa = {:.4f} (epoch {})'.format(
                np.min(val_losses), val_kappas[np.argmin(val_losses)], np.argmin(val_losses) + 1))
            print('')
            break

        # break if max epochs
        if epoch == (max_epochs - 1):
            print('Did not meet early stopping. Best results: loss = {:.4f}, kappa = {:.4f} (epoch {})'.format(
                np.min(val_losses), val_kappas[np.argmin(val_losses)], np.argmin(val_losses) + 1))
            print('')
            break

# load best predictions
oof_preds = oof_preds_best
                        
# print performance
print('')
print('Finished in {:.2f} minutes'.format((time.time() - cv_start) / 60))

