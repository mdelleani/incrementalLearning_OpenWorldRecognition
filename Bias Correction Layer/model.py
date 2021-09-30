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

def UpdateRepresentation(ResNet, task, train_Dataset, new_train_indexes, exemplar_indexes,bias_layers):

    ResNet.train()
    for layer in bias_layers:
        layer.eval()

    if task>0:    #take a number of samples for each new classes equal to the numer of exemplars for the old class, so i have a balanced dataset of new and old classes (val_set)
        new_class_val_exemplars=[]
        known_classes= task*params.TASK_CLASSES
        m = int(params.K/(known_classes+params.TASK_CLASSES))
        m = int((m + .5)*0.1) #take only the 10% for the bias layer training

        for lbl in range( known_classes, known_classes+params.TASK_CLASSES): #for each new class
            index_set = []
            counter=0
            for idx in new_train_indexes:                                    #for each index of the training set
                image, label, index = train_Dataset.__getitem_mapped__(idx)  #get the image
                if label == lbl and counter<m:                               #if its label is the same we are considering at the moment
                    index_set.append(index.numpy().item())              #store index
                    counter+=1
            new_class_val_exemplars.append(index_set)



    if task >0:
        train_exemplar_indexes=[]
        val_exemplar_indexes=[]
        for class_exemplar_indexes in exemplar_indexes: # take the indexes of the exemplars of the old classes
            tot_ex=len(class_exemplar_indexes)
            train_exemplar_indexes=np.array(np.concatenate((train_exemplar_indexes,class_exemplar_indexes[:int(tot_ex*0.9)]))) #90% of the exemplars for normal train
            val_exemplar_indexes=np.array(np.concatenate((val_exemplar_indexes,class_exemplar_indexes[:int(tot_ex*0.1)]))) #10% of the exemplars for bias layer train
        train_indexes =np.array( np.concatenate((new_train_indexes,train_exemplar_indexes)) ,dtype = int)
        for iter in new_class_val_exemplars: #take the indexes of the samples for the new classes
            val_indexes=np.array( np.concatenate((val_exemplar_indexes,iter)) ,dtype = int)

        train_set=Subset(train_Dataset,train_indexes,transform=params.transform_train)
        val_set=Subset(train_Dataset,val_indexes,transform=params.transform_train) #val set is used for bias layer training it has samples/exemplars for all class (it's balanced)
        val_loader=DataLoader( val_set, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE, shuffle=True)


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

    criterion =nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ResNet.parameters(), lr=params.LR, momentum=params.MOMENTUM, weight_decay=params.WEIGHT_DECAY)
    bias_optimizer = optim.Adam(bias_layers[task].parameters(), lr=0.001)  #for each task only one bias layer is trained i.e. task 0 first bias layer, task 1 second bias layer and so on
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
            onehot_labels = Id[labels].cuda()
            optimizer.zero_grad()
            output = ResNet(images,features = False)
            #output= bias_forward(output,bias_layers,task)

            #classification loss
            if task==0:
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            # classification and distillation loss
            if task>0:
                m = nn.LogSoftmax(dim=1)
                T = 2
                alpha = (params.TASK_CLASSES*(task))/ (params.TASK_CLASSES*(task+1))
                p=bias_forward(output,bias_layers,task)
                pre_p =oldNet(images)
                pre_p=bias_forward(pre_p,bias_layers,task-1)
                pre_p=F.softmax(pre_p[:,:params.TASK_CLASSES*(task)]/T, dim=1)
                logp = m(p[:,:params.TASK_CLASSES*(task)]/T)
                loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
                loss_hard_target =criterion(p[:,:params.TASK_CLASSES*(task+1)],labels)
                loss = loss_soft_target * T * T + (1-alpha) * loss_hard_target
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            _ , preds = output.max(1)
            correct_preds += (preds == labels).sum()
            n_images += len(images)
        accuracy = correct_preds/n_images
        scheduler.step()
        print(f"in task {task} and epoch {epoch} the loss is {loss} and the accuracy is {accuracy}")

    if task>0:     #training phase for the n-th bias layer
        bias_layers[task].train()
        ResNet.eval()
        for epoch in range (200):
            for images,labels,indices in val_loader:
                images = images.float().to(params.DEVICE) ## need to be float
                labels = labels.to(params.DEVICE)
                indices = indices.to(params.DEVICE)
                p=ResNet(images)
                p=bias_forward(p,bias_layers,task)
                loss = criterion(p[:,:params.TASK_CLASSES*(task+1)],labels)
                bias_optimizer.zero_grad()
                loss.backward()
                bias_optimizer.step()





    return ResNet, train_set, train_loader,bias_layers

def ReduceExemplarSet(Py, m):
    return Py[:m]  #top m most representative images for the given class

def ConstructExemplarSet(train_Dataset, new_train_indexes, m, ResNet, known_classes, exemplar_indexes,bias_layers,task):
    ResNet = ResNet.train(False)
    #new_exemplars=[]
    image_set = []
    for lbl in range( known_classes, known_classes+params.TASK_CLASSES): #for each new class
        image_set = []
        index_set = []
        for idx in new_train_indexes:                                    #for each index of the training set
            image, label, index = train_Dataset.__getitem_mapped__(idx)  #get the image
            image = image.float()
            if label == lbl:                                             #if its label is the same we are considering at the moment
                image_set.append(image.numpy())                          #store image
                index_set.append(index.numpy().item())                                  #store index
        image_set=torch.from_numpy(np.array(image_set))

        features = []
        with torch.no_grad():
            for image in image_set:                                          # for each image of the same class
                image=image.clone().detach().to(params.DEVICE)
                feature = ResNet(image.unsqueeze(0),features = True).data.cpu().numpy()
                #feature=bias_forward(feature,bias_layers,task).data.cpu().numpy()
                feature = feature / np.linalg.norm(feature)                  # get the features
                features.append(feature[0])                                  # add an element to the class features
        features = np.array(features)                                    #add a cell containing all features from the same class
        mu = np.mean(features, axis=0)
        mu = mu / np.linalg.norm(mu)                                     #class mean normalized

        taken_features = []                                              # sum from 1 to k-1 of phi(p_j) on the paper
        exemplar_set = []
        index_set=deepcopy(index_set)
        for k in range (m):
            mu_p = (1/(k+1))*(features + np.sum(taken_features, axis=0))
            i = np.argmin(np.sqrt(np.sum((mu-mu_p)**2 ,axis = 1)))       #index of the next image to take
            exemplar_set.append(index_set[i])                            # append index of next most representative image
            taken_features.append(features[i])    # add the respective features to phi(p_j)

            features = np.delete(features, i, axis = 0)
            index_set = np.delete(index_set, i, axis = 0)

            #print(i)

        exemplar_indexes.append(exemplar_set)

    return exemplar_indexes


def ComputeMeansofExemplars(exemplar_indexes, ResNet,ds,bias_layers,task):           #you only need to do this once per task, that's why is out of classify method

    ResNet.train(False)
    exemplar_means = []
    for class_exemplars_indexes in exemplar_indexes:

        mean=0
        exemplars_ds=Subset(ds, class_exemplars_indexes,transform=params.transform_train)
        loader = DataLoader( exemplars_ds, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE)
        n_images=0
        for image,_,_ in loader:
            with torch.no_grad():
                image=image.float().to(params.DEVICE)
                features = ResNet(image,features = True)
                #features=bias_forward(features,bias_layers,task)
                features /= torch.norm(features, p=2)
                ma = torch.sum(features, dim=0)
                mean += ma
                n_images+=len(image)



        mean=mean/n_images
        mean = mean / mean.norm()
        mean=mean.cpu().numpy()
        exemplar_means.append(mean)
    return np.array(exemplar_means)                             #all means

def classify(features_batch, exemplar_means):
    preds = []
    features_batch/= torch.norm(features_batch, p=2)
    for features in features_batch:                                          #for each image
        features= features.data.cpu().numpy()
        distances = np.sqrt(np.sum((exemplar_means-features)**2 ,axis = 1))  #compute the distances from all exemplar means
        pred = np.argmin(distances)                                          #take the nearest class
        preds.append(pred)
    return preds



class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))
    def forward(self, x):
        return self.alpha * x + self.beta
    def printParam(self, i):
        print(i, self.alpha.item(), self.beta.item())


def bias_forward(input,bias_layers,task):  #forward the output of the ResNet trough the bias layers, the first 10 columns are the input for the first layer,
    out=[]                                  #the following 10 are the input of the second layer and so on
    for limit in range (task+1):
        in_data=input[:, limit*params.TASK_CLASSES:(limit+1)*params.TASK_CLASSES]
        out.append(bias_layers[limit](in_data))
    return torch.cat(out, dim = 1)
