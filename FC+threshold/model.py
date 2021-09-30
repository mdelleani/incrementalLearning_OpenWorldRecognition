import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import params
from data import Dataset, Subset
from torch.utils.data import DataLoader
import model
import resnet
from PIL import Image
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import pandas as pd
import random

def UpdateRepresentation(ResNet, task, train_Dataset, new_train_indexes, exemplar_indexes):

    ResNet.train()
    if task >0:  #concatenate the indexes  of the images of the train test with the indexes of the images of the exemplar set
        all_exemplar_indexes=[]
        for class_exemplar_indexes in exemplar_indexes:
            for i in class_exemplar_indexes:
                all_exemplar_indexes.append(i)
        train_indexes =np.array( np.concatenate((new_train_indexes,all_exemplar_indexes)) ,dtype = int)
        train_set=Subset(train_Dataset,train_indexes,transform=params.transform_train)

    else:
        train_set=Subset(train_Dataset,new_train_indexes,transform=params.transform_train)
    #train_loader
    train_loader = DataLoader( train_set, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE, shuffle=True)

    if task > 0:
        #old network for distillation loss
        oldNet = deepcopy(ResNet)
        oldNet = oldNet.to(params.DEVICE)
        oldNet.train(False)

        #add *params.TASK_CLASSES* neurons to the fc layer:
        in_features = ResNet.fc.in_features             #save number of input features of old fc
        out_features = ResNet.fc.out_features           #save number of output features of old fc
        weight = ResNet.fc.weight.data                  #save weights of old fc

        ResNet.fc = nn.Linear(in_features, out_features + params.TASK_CLASSES) #new fc
        ResNet.fc.weight.data[:out_features] = weight   #weights for previous classes are the same
        ResNet.to(params.DEVICE)                        #otherwise the new layer is not on device

    criterion=nn.BCEWithLogitsLoss(reduction="mean") #BCE for classification loss and distillation loss
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
            onehot_labels = Id[labels].cuda() #one-hot encoding
            optimizer.zero_grad()
            output = ResNet(images,features = False)
            # classification loss
            if task==0:
                loss = criterion(output, onehot_labels)
            # classification and distillation loss
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

def ReduceExemplarSet(Py, m):
    return Py[:m]  #top m most representative images for the given class

def ConstructExemplarSet(train_Dataset, new_train_indexes, m, ResNet, known_classes, exemplar_indexes):
    ResNet = ResNet.train(False)
    image_set = []
    for lbl in range( known_classes, known_classes+params.TASK_CLASSES): #for each new class
        image_set = []
        index_set = []
        for idx in new_train_indexes:                                    #for each index of the training set
            image, label, index = train_Dataset.__getitem_mapped__(idx)  #get the image
            image = image.float()
            if label == lbl:                                             #if its label is the same we are considering at the moment
                image_set.append(image.numpy())                          #store image
                index_set.append(index.numpy().item())                  #store index


        image_set=torch.from_numpy(np.array(image_set))                   #all images from the same class

        features = []
        with torch.no_grad():
            for image in image_set:                                          # for each image of the same class
                image=image.clone().detach().to(params.DEVICE)
                feature = ResNet(image.unsqueeze(0),features = True).data.cpu().numpy()
                feature = feature / np.linalg.norm(feature)                  # get the features and normalize
                features.append(feature[0])                                  # add an element to the class features
        features = np.array(features)
        mu = np.mean(features, axis=0)                                   # mean of the class
        mu = mu / np.linalg.norm(mu)                                     #class mean normalized


        taken_features = []                                              # sum from 1 to k-1 of phi(p_j) on the paper
        exemplar_set = []
        index_set=deepcopy(index_set)
        for k in range (m):                                               # for loop to take the m most representative exemplars
            mu_p = (1/(k+1))*(features + np.sum(taken_features, axis=0))
            i = np.argmin(np.sqrt(np.sum((mu-mu_p)**2 ,axis = 1)))       #index of the next image to take
            exemplar_set.append(index_set[i])                            # append index of next most representative image
            taken_features.append(features[i])                              # add the respective features to phi(p_j)
            features = np.delete(features, i, axis = 0)                  #delete feature and index to avoid to take the same image more than once
            index_set = np.delete(index_set, i, axis = 0)

            #print(i)

        exemplar_indexes.append(exemplar_set)      #exemplar_indexes is a nested list, i.e.  at exemplar_indexes[0] we have a list with all the exemplars indexes for class 0 and so on

    return exemplar_indexes
