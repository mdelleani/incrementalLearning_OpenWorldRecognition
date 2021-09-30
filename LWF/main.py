import resnet
import params
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
import random
from data import Dataset, Subset
from copy import deepcopy

def Train (ResNet, task, train_Dataset, new_train_indexes): #this mmethod is used to add neurons to the fc layer and train

    ResNet.train()
    train_set=Subset(train_Dataset,new_train_indexes,transform=params.transform_train)
    train_loader = DataLoader( train_set, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE, shuffle=True)

    if task > 0:
        #old network for distillation loss
        oldNet = deepcopy(ResNet)
        oldNet = oldNet.to(params.DEVICE)
        oldNet.train(False)

        #add *params.TASK_CLASSES* neurons to the fc layer:
        in_features = ResNet.fc.in_features             #save nÃ‚Â° of input features of old fc
        out_features = ResNet.fc.out_features           #save nÃ‚Â° of output features of old fc
        weight = ResNet.fc.weight.data                  #save weights of old fc

        ResNet.fc = nn.Linear(in_features, out_features + params.TASK_CLASSES) #new fc
        ResNet.fc.weight.data[:out_features] = weight   #weights for previous classes are the same
        ResNet.to(params.DEVICE)                        #otherwise the new layer is not on device

    criterion=nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.SGD(ResNet.parameters(), lr=params.LR, momentum=params.MOMENTUM, weight_decay=params.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,params.STEP_SIZE,gamma=params.GAMMA,last_epoch=-1)


    #training loop
    for epoch in range(params.N_EPOCHS):
        correct_preds=0
        n_images=0
        for images, labels, indices  in train_loader:
            images = images.float().to(params.DEVICE) ## need to be float
            labels = labels.to(params.DEVICE)
            indices = indices.to(params.DEVICE)
            Id = torch.eye(params.TASK_CLASSES*(task+1))
            onehot_labels = Id[labels].cuda()     #one-hot encoding
            optimizer.zero_grad()
            output = ResNet(images,features = False)
            # classification loss
            if task==0:
                loss = criterion(output, onehot_labels)
            # classification loss and distillation lost
            if task>0:
                q = torch.zeros(50000, params.TASK_CLASSES*(task)).cuda()
                g = torch.sigmoid(oldNet(images))
                q[indices] = g.data
                q_i = q[indices]
                onehot_labels[:,:params.TASK_CLASSES*(task)] = q_i[:,:params.TASK_CLASSES*(task)]
                loss=criterion(output,onehot_labels)

            loss.backward()
            optimizer.step()

            _ , preds = output.max(1)
            correct_preds += (preds == labels).sum()
            n_images += len(images)
        accuracy = correct_preds/n_images
        scheduler.step()
        print(f"in task {task} and epoch {epoch} the loss is {loss} and the accuracy is {accuracy}")
    return ResNet, train_set, train_loader



original_trainset = torchvision.datasets.CIFAR100(root= './data', train= True, transform= params.transform_train, download = True)
original_testset = torchvision.datasets.CIFAR100(root = './data', train = False, transform= params.transform_test, download = True)

# use our custom class for dataset
train_Dataset = Dataset(original_trainset, classes_per_task= params.TASK_CLASSES,transform= params.transform_train)
test_Dataset = Dataset(original_testset, classes_per_task = params.TASK_CLASSES,transform= params.transform_test)



#check if the splits in train and test are equal
assert (np.array_equal(train_Dataset.splits, test_Dataset.splits)), "The splits are different, check the code!"

# number of splits = number of tasks
splits =(train_Dataset.splits)
n_tasks = splits.shape[0]
splits = splits.tolist()
print("Successful split. Number of tasks: ", n_tasks)

train_indexes = []
test_indexes = []

ResNet=resnet.resnet32(num_classes=params.TASK_CLASSES)
ResNet.to(params.DEVICE)


random.seed(params.SEED)
np.random.seed(params.SEED)
torch.manual_seed(params.SEED)


for task in range(n_tasks):
  #indexes for this task
  known_classes= task*params.TASK_CLASSES

  #trainset and testset of the task
  train_indexes =  train_Dataset.__getIndexesGroups__(task*params.TASK_CLASSES) # splits[task]
  test_indexes = test_indexes + test_Dataset.__getIndexesGroups__(task*params.TASK_CLASSES)

  ResNet ,train_set, train_loader = Train(ResNet, task, train_Dataset, train_indexes)
  ResNet.eval()

#results on training set
  all_preds = []
  all_labels = []
  n_images = 0
  correct_preds=0
  for images, labels, _ in train_loader:
    images = images.float().to(params.DEVICE) ## need to be float
    labels = labels.to(params.DEVICE)
    output = ResNet(images,features = False)
    _ , preds = output.max(1)
    correct_preds += (preds == labels).sum()
    n_images += len(images)
    all_preds = np.concatenate((all_preds,preds.cpu()))
    all_labels = np.concatenate((all_labels,labels.cpu()))


  accuracy = correct_preds/n_images
  print(f"accuracy on training set: {accuracy}")

  #results on test set
  test_set = Subset(test_Dataset,test_indexes,transform=params.transform_test)
  test_loader = DataLoader( test_set, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE, shuffle=True)
  all_preds = []
  all_labels = []
  n_images = 0
  correct_preds=0
  for images, labels, _ in test_loader:
    images = images.float().to(params.DEVICE) ## need to be float
    labels = labels.to(params.DEVICE)
    output = ResNet(images,features = False)
    _ , preds = output.max(1)
    n_images += len(images)
    correct_preds += (preds == labels).sum()
    all_preds = np.concatenate((all_preds,preds.cpu()))
    all_labels = np.concatenate((all_labels,labels.cpu()))

  accuracy = correct_preds/n_images
  print(f"accuracy on test set: {accuracy}")


  #confusion matrix
  cm = confusion_matrix(all_labels,all_preds)
  df_cm = pd.DataFrame(cm, range((task+1)*params.TASK_CLASSES), range((task+1)*params.TASK_CLASSES))
  plt.figure(figsize = (10,7))
  sn.heatmap(df_cm, annot=False, cmap="viridis")
  plt.savefig(f"{task}_cf")
  ResNet = ResNet.train(True)
