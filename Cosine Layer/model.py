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
import math
import modified_linear
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms


import pandas as pd
import random



firstTask = True
cur_features = []
ref_features = []
old_scores = []
new_scores = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    global firstTask
    ref_features = inputs[0]


def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs


def UpdateRepresentation(ResNet, task, train_Dataset, new_train_indexes, exemplar_indexes):

    ResNet.train()


    if task >0:
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

    oldNet = None
    if task == 1:
        #old network for distillation loss
        oldNet = deepcopy(ResNet)
        oldNet = oldNet.to(params.DEVICE)
        oldNet.train(False)

        #add *params.TASK_CLASSES* neurons to the fc layer:
        in_features = ResNet.fc.in_features             #save nÃ‚Â° of input features of old fc
        out_features = ResNet.fc.out_features           #save nÃ‚Â° of output features of old fc
        new_fc = modified_linear.SplitCosineLinear(in_features, out_features, params.TASK_CLASSES)
            #save weights of old fc
        new_fc.fc1.weight.data = ResNet.fc.weight.data
        new_fc.sigma.data = ResNet.fc.sigma.data
        ResNet.fc = new_fc
        lambda_mult = out_features*1.0 / params.TASK_CLASSES
    elif task >1:
        oldNet = deepcopy(ResNet)
        oldNet = oldNet.to(params.DEVICE)
        oldNet.train(False)
        in_features = ResNet.fc.in_features
        out_features1 = ResNet.fc.fc1.out_features
        out_features2 = ResNet.fc.fc2.out_features
        print("in_features:", in_features, "out_features1:", \
        out_features1, "out_features2:", out_features2)
        new_fc = modified_linear.SplitCosineLinear(in_features, out_features1+out_features2, params.TASK_CLASSES)
        new_fc.fc1.weight.data[:out_features1] = ResNet.fc.fc1.weight.data
        new_fc.fc1.weight.data[out_features1:] = ResNet.fc.fc2.weight.data
        new_fc.sigma.data = ResNet.fc.sigma.data
        ResNet.fc = new_fc
        lambda_mult = (out_features1+out_features2)*1.0 / (params.TASK_CLASSES)

    ResNet.to(params.DEVICE)                        #otherwise the new layer is not on device

    less_forget = True
    adapt_lambda = True
    p_lambda = 5

    if task > 0 and less_forget and adapt_lambda:
        cur_lambda = p_lambda * math.sqrt(lambda_mult)
    else:
        cur_lamda = p_lambda
    if task > 0 and less_forget:
        print("###############################")
        print("Lambda for less forget is set to ", cur_lambda)
        print("###############################")

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
            onehot_labels = Id[labels].cuda()
            optimizer.zero_grad()
            output = ResNet(images,features = False)
            loss = calculateLossExt(ResNet,oldNet,task, images, labels)
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
                index_set.append(index.numpy().item())                                  #store index

                                     #add a cell containing all images from the same class
        image_set=torch.from_numpy(np.array(image_set))

        features = []
        with torch.no_grad():
            for image in image_set:                                          # for each image of the same class
                image=image.clone().detach().to(params.DEVICE)
                feature = ResNet(image.unsqueeze(0),features = True).data.cpu().numpy()
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


        exemplar_indexes.append(exemplar_set)

    return exemplar_indexes


def ComputeMeansofExemplars(exemplar_indexes, ResNet,ds):           #you only need to do this once per task, that's why is out of classify method

    ResNet.train(False)
    exemplar_means = []

    for class_exemplars_indexes in exemplar_indexes:

        mean=0
        #print(class_exemplars_indexes)
        exemplars_ds=Subset(ds, class_exemplars_indexes,transform=params.transform_train)
        loader = DataLoader( exemplars_ds, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE)
        for image,idx,_ in loader:
            with torch.no_grad():
                image=image.float().to(params.DEVICE)
                features = ResNet(image,features = True)
                features /= torch.norm(features, p=2)
                ma = torch.sum(features, dim=0)
                mean += ma


        mean=mean/len(idx)
        mean = mean / mean.norm()
        mean=mean.cpu().numpy()
        exemplar_means.append(mean)
    return np.array(exemplar_means)                             #all means

def classify(features_batch, exemplar_means):
    preds = []
    features_batch/= torch.norm(features_batch, p=2)
    for features in features_batch:                                          #for each image
        features= features.data.cpu().numpy()
        #exemplar_means=exemplar_means.data.cpu().numpy()
        distances = np.sqrt(np.sum((exemplar_means-features)**2 ,axis = 1))  #compute the distances from all exemplar means
        pred = np.argmin(distances)                                          #take the nearest class
        preds.append(pred)
    return preds


def calculateLossExt(tg_model, ref_model, task, inputs, targets, weight_per_class = None ):

    # define default params --> da ridvedere (potremmo customizzarlo)
    lamda = 5 # for LF
    K = 2 # default for margin Loss
    dist = 0.5 # for Margin
    lw_mr = 1


    if task > 0:
        ref_model.eval()

        num_old_classes = ref_model.fc.out_features
        handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)

        handle_cur_features = tg_model.fc.register_forward_hook(get_cur_features)
        handle_old_scores_bs = tg_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
        handle_new_scores_bs = tg_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        outputs = tg_model(inputs)
        ref_outputs = ref_model(inputs)

        loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), \
                                        torch.ones(inputs.shape[0]).to(params.DEVICE)) * lamda
        loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

        outputs_bs = torch.cat((old_scores, new_scores), dim=1)


        assert(outputs_bs.size()==outputs.size())

        #get groud truth scores
        gt_index = torch.zeros(outputs_bs.size()).to(params.DEVICE)
        gt_index = gt_index.scatter(1, targets.view(-1,1), 1).ge(0.5)
        gt_scores = outputs_bs.masked_select(gt_index)

        #get top-K scores on none gt classes
        none_gt_index = torch.zeros(outputs_bs.size()).to(params.DEVICE)
        none_gt_index = none_gt_index.scatter(1, targets.view(-1,1), 1).le(0.5)
        none_gt_scores = outputs_bs.masked_select(none_gt_index).reshape((outputs_bs.size(0), outputs.size(1)-1))

        hard_scores = none_gt_scores.topk(K, dim=1)[0]

        #the index of hard samples, i.e., samples of old classes
        hard_index = targets.lt(num_old_classes)
        hard_num = torch.nonzero(hard_index).size(0)

        if  hard_num > 0:
            gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, K)
            hard_scores = hard_scores[hard_index]
            assert(gt_scores.size() == hard_scores.size())
            assert(gt_scores.size(0) == hard_num)
            loss3 = nn.MarginRankingLoss(margin=0.5)(gt_scores.view(-1, 1), \
                        hard_scores.view(-1, 1), torch.ones(hard_num*K).to(params.DEVICE)) * lw_mr
        else:
            loss3 = torch.zeros(1).to(params.DEVICE)

        loss = loss1 + loss2 + loss3


        handle_ref_features.remove()
        handle_cur_features.remove()
        handle_old_scores_bs.remove()
        handle_new_scores_bs.remove()


    else:
        outputs = tg_model(inputs)
        loss =  nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

    return loss
