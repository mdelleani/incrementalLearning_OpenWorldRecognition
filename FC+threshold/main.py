import model
import resnet
import params
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
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

# Load data (or download if not exist) the original dataset

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


start_task = 0
closed_wrld_tasks = 5
threshold=0.5
train_indexes = []
test_indexes = []
open_test_indexes = []

# accuracy

above_train = []
above_test = []
below_train = []
below_test = []
above_test_open=[]
below_test_open=[]


total_accuracy_closedWorld = []
total_accuracy_openWorld=[]
total_armonic_mean=[]

ResNet=resnet.resnet32(num_classes= params.TASK_CLASSES)
ResNet.to(params.DEVICE)

exemplar_indexes = []
exemplar_images  = []
random.seed(params.SEED)
np.random.seed(params.SEED)
torch.manual_seed(params.SEED)

for task in range(closed_wrld_tasks, params.N_TASKS):
  open_test_indexes = open_test_indexes + test_Dataset.__getIndexesGroups__((task)* params.TASK_CLASSES)

open_test_set = Subset(test_Dataset,open_test_indexes,transform=params.transform_test)
for task in range(start_task,closed_wrld_tasks):
  known_classes= task*params.TASK_CLASSES
  #indexes for this task
  train_indexes =  train_Dataset.__getIndexesGroups__(task* params.TASK_CLASSES) # splits[task]
  test_indexes = test_indexes + test_Dataset.__getIndexesGroups__(task* params.TASK_CLASSES)


  ResNet ,train_set, train_loader = model.UpdateRepresentation(ResNet, task, train_Dataset, train_indexes, exemplar_indexes)


  m = int(params.K/(known_classes+params.TASK_CLASSES))
  m = int(m + .5)



  for y,Py in enumerate(exemplar_indexes):
    exemplar_indexes[y]=model.ReduceExemplarSet(Py,m)

  exemplar_indexes = model.ConstructExemplarSet(train_Dataset, train_indexes, m, ResNet, known_classes, exemplar_indexes)



    #end of task results
  ResNet.eval()


  tot_images=0
  #results on closed test set
  test_set = Subset(test_Dataset,test_indexes,transform=params.transform_test)
  test_loader = DataLoader( test_set, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE, shuffle=True)

  all_preds_closedWorld = []
  all_labels = []
  nclosed_images = 0
  correct_preds_closedWorld=0

  above_threshold_test = 0
  below_threshold_test = 0
  list_labels_test = []

  for images, labels, _ in test_loader:
    images = images.float().to(params.DEVICE) ## need to be float
    labels = labels.to(params.DEVICE)
    list_labels_test.append(labels.tolist())
    # closed with rejection using FullyC
    outputs = ResNet(images, features = False)
    m = nn.Softmax(dim=1)
    probabilities = m(outputs)
    max_prob = torch.max(probabilities, dim = 1)
    above_threshold = (max_prob.values> threshold).sum()
    below_threshold = max_prob.values.size()[0] - above_threshold
    above_threshold_test += above_threshold
    below_threshold_test += below_threshold
    new_outputs = ResNet.reject(outputs, threshold) # new_ouputs
    nclosed_images += len(images)
    all_preds_closedWorld = np.concatenate((all_preds_closedWorld,new_outputs.cpu()))
    all_labels = np.concatenate((all_labels,labels.cpu()))

  # compute correct
  correct_preds_closedWorld += (all_preds_closedWorld == all_labels).sum()


  # accuracy
  accuracy_closedWorld = correct_preds_closedWorld/nclosed_images
  print(f"accuracy on test set closedWorld: {accuracy_closedWorld}")
  total_accuracy_closedWorld.append(accuracy_closedWorld)

  # below
  below_test.append(below_threshold_test.item())
  # above
  above_test.append(above_threshold_test.item())
  print(f"Above threshold test set closedWorld task {task} :", above_threshold_test.item())
  above_train.append( above_threshold_test.item())
  print(f"Below threshold test set closedWorld task {task} :", below_threshold_test.item())
  below_train.append(below_threshold_test.item())

  # print classes
  test_lbl = np.concatenate(list_labels_test).tolist()
  #print("CLOSED TEST set: ", set(test_lbl))

   #results on open test set

  open_test_loader = DataLoader( open_test_set, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE, shuffle=True)

  all_preds_openWorld = []
  all_labels = []
  nopen_images = 0
  correct_preds_openWorld=0

  above_threshold_test = 0
  below_threshold_test = 0
  list_labels_test = []

  for images, labels, _ in open_test_loader:
    images = images.float().to(params.DEVICE) ## need to be float
    labels = labels.to(params.DEVICE)
    list_labels_test.append(labels.tolist())
    # closed with rejection using FullyC
    outputs = ResNet(images, features = False)
    m = nn.Softmax(dim=1)
    probabilities = m(outputs)
    max_prob = torch.max(probabilities, dim = 1)
    above_threshold = (max_prob.values> threshold).sum()
    below_threshold = max_prob.values.size()[0] - above_threshold
    above_threshold_test += above_threshold
    below_threshold_test += below_threshold
    nopen_images += len(images)

  # accuracy
  accuracy_openWorld =below_threshold_test/nopen_images
  print(f"accuracy on test set openWorld: {accuracy_openWorld}")
  total_accuracy_openWorld.append(accuracy_openWorld)

  # below
  below_test_open.append(below_threshold_test.item())
  # above
  above_test_open.append(above_threshold_test.item())
  print(f"Above threshold test set openWorld task {task} :", above_threshold_test.item())
  above_test.append(above_threshold_test.item())
  print(f"Below threshold test set openWorld task {task} :", below_threshold_test.item())
  below_test.append(below_threshold_test.item())



  test_lbl = np.concatenate(list_labels_test).tolist()
  #print("OPEN TEST set: ", set(test_lbl))

  armonic_mean=(2*accuracy_openWorld*accuracy_closedWorld)/(accuracy_openWorld+accuracy_closedWorld)
  total_armonic_mean.append(armonic_mean)
  print(f"Armonic mean task{task}: {armonic_mean}")

print("Above closed: ", above_train)
print("Above open: ", above_test)

print("Below closed: ", below_train)
print("Below open: ", below_test)

#print("TOTAL ACCURACY closedWorld: ", total_accuracy_closedWorld)
print("TOTAL ARMONIC MEANS", total_armonic_mean)
