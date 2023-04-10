import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn.functional as F
from tqdm.auto import tqdm,trange
from pathlib import Path
import pandas as pd
import numpy as np
import random
import learn2learn as l2l
import pickle

def test(meta_model, test_data, loss_func, device):
    test_loss = 0
    test_acc = 0
    test_sup_images = test_data['sup_images']
    test_sup_labels = test_data['sup_labels']
    test_qry_images = test_data['qry_images']
    all_pred = []
    for i in trange(len(test_sup_images)):
        learner = meta_model.clone(first_order = False)
        sup_image = torch.tensor(test_sup_images[i]).to(device)
        sup_label = torch.tensor(test_sup_labels[i]).to(device)
        qry_image = torch.tensor(test_qry_images[i]).to(device)
        for step in range(fas): # inner loop
            pred = learner(sup_image)
            train_loss = loss_func(pred, sup_label)
            learner.adapt(train_loss)
        output_pred = torch.argmax(learner(qry_image),1).reshape(-1,1).detach().cpu().numpy()
        all_pred += [item for sublist in output_pred for item in sublist]
    return all_pred

# set seed
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# read data
with open("./data/test.pkl",'rb') as f:
    test_data = pickle.load(f)
    
#prediction
model_path ="./model/resnet_task_augmentation_queryaug_v3/model_9500.pt"    
model = torch.load(model_path).to(device)
loss_func = nn.CrossEntropyLoss(reduction='mean')
fas =1
pred_res = test(model, test_data, loss_func, device)
df = pd.concat([pd.Series(range(15000)),pd.Series(pred_res)],axis = 1)
df = df.rename(columns={df.columns[0]:"Id",df.columns[1]:"Category"})
df.to_csv("prediction.csv", index =False)