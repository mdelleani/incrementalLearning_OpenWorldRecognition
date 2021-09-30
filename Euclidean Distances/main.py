
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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--std', action='store_true')
args = parser.parse_args()
std=args.std
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
train_indexes = []
test_indexes = []
open_test_indexes = []

# accuracy


total_accuracy_closedWorld = []
total_accuracy_openWorld=[]
total_armonic_mean=[]

ResNet=resnet.resnet32(num_classes=params.TASK_CLASSES)
ResNet.to(params.DEVICE)

exemplar_indexes = []
exemplar_images  = []
random.seed(params.SEED)
np.random.seed(params.SEED)
torch.manual_seed(params.SEED)

for task in range(closed_wrld_tasks, params.N_TASKS):
  open_test_indexes = open_test_indexes + test_Dataset.__getIndexesGroups__((task)*params.TASK_CLASSES)
open_test_set = Subset(test_Dataset,open_test_indexes,transform=params.transform_test)
for task in range(start_task,closed_wrld_tasks):
  known_classes= task*params.TASK_CLASSES
  #indexes for this task
  val_indexes = train_indexes + train_Dataset.__getIndexesGroups__(task*params.TASK_CLASSES)
  train_indexes =  train_Dataset.__getIndexesGroups__(task*params.TASK_CLASSES) # splits[task]

  test_indexes = test_indexes + test_Dataset.__getIndexesGroups__(task*params.TASK_CLASSES)


  ResNet ,train_set, train_loader = model.UpdateRepresentation(ResNet, task, train_Dataset, train_indexes, exemplar_indexes)

  m = int(params.K/(known_classes+params.TASK_CLASSES))
  m = int(m + .5)

  for y,Py in enumerate(exemplar_indexes):
    exemplar_indexes[y]=model.ReduceExemplarSet(Py,m)

  exemplar_indexes = model.ConstructExemplarSet(train_Dataset, train_indexes, m, ResNet, known_classes, exemplar_indexes)


  exemplar_means = model.ComputeMeansofExemplars(exemplar_indexes, ResNet,train_Dataset)
    #end of task results
  ResNet.eval()


  #results on training set
  train_set = Subset(train_Dataset,val_indexes,transform=params.transform_test)
  train_loader = DataLoader( train_set, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE, shuffle=True)
  all_preds = []
  all_distances= []
  all_labels = []
  all_margins =  []

  for images, labels, _ in train_loader:
    images = images.float().to(params.DEVICE)
    labels = labels.to(params.DEVICE)

    # using NME
    features = ResNet(images,features = True)
    preds, batch_distances, batch_margins = model.classify(features, exemplar_means)
    all_preds = np.concatenate((all_preds, preds))
    all_distances = np.concatenate((all_distances, batch_distances))
    all_margins = np.concatenate ((all_margins, batch_margins))


  #all_preds_mm = all_preds
  tr_mean=np.mean(all_margins)
  tr_std=np.std(all_margins)
  print("margins mean training", tr_mean)
  print("margins std training",tr_std)
  print("mean min distance training",np.mean(all_distances))
  print("\n\n")
  if std:
      tresh=tr_mean-tr_std
  else:
      tresh=tr_mean

  #results on closed test set
  test_set = Subset(test_Dataset,test_indexes,transform=params.transform_test)
  test_loader = DataLoader( test_set, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE, shuffle=True)
  all_preds = []
  all_distances= []
  all_labels = []
  all_margins =  []
  nclosed_images = 0

  for images, labels, _ in test_loader:
    images = images.float().to(params.DEVICE) ## need to be float
    labels = labels.to(params.DEVICE)
    # using NME
    features = ResNet(images,features = True)
    preds, batch_distances, batch_margins = model.classify(features, exemplar_means)
    all_preds = np.concatenate((all_preds, preds))
    all_distances = np.concatenate((all_distances, batch_distances))
    all_margins = np.concatenate ((all_margins, batch_margins))
    all_labels = np.concatenate((all_labels,labels.cpu()))
    nclosed_images+=len(images)



  #all_preds_mm = all_preds
  print("margins mean closed",np.mean(all_margins))
  print("margins std closed ",np.std(all_margins))
  print("min distance mean closed",np.mean(all_distances))
  print("\n \n")





  wrong_margins = all_margins <= tresh
  all_preds[wrong_margins] = 100

  correct_preds_test =(all_preds == all_labels).sum()

  accuracy_stest_closed = correct_preds_test/nclosed_images
  print(f"accuracy on closed test set: {accuracy_stest_closed}")
  print("\n\n")
  total_accuracy_closedWorld.append(accuracy_stest_closed)

  open_test_loader = DataLoader( open_test_set, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE, shuffle=True)
  all_preds = []
  all_labels = []
  all_distances = []
  all_margins = []
  nopen_images = 0

  for images, labels, _ in open_test_loader:
    images = images.float().to(params.DEVICE) ## need to be float
    labels = labels.to(params.DEVICE)
    #NME
    features = ResNet(images,features = True)
    preds, batch_distances, batch_margins = model.classify(features, exemplar_means)
    all_preds = np.concatenate((all_preds, preds))
    all_distances = np.concatenate((all_distances, batch_distances))
    all_margins = np.concatenate((all_margins, batch_margins))
    all_labels = np.concatenate((all_labels,labels.cpu()))
    nopen_images+=len(images)
  all_preds_mm = all_preds

  print("margins mean open", np.mean(all_margins))
  print("margins std open",np.std(all_margins))
  print("min distance mean open",np.mean(all_distances))

  wrong_margins = all_margins <= tresh
  all_preds[wrong_margins] = 100
  below_standard = (all_preds == 100).sum()
  accuracy_stest= below_standard/nopen_images
  print(f"accuracy on open test set : {accuracy_stest}")
  print("\n\n")
  total_accuracy_openWorld.append(accuracy_stest)
  armonic_mean=(2* accuracy_stest*accuracy_stest_closed)/( accuracy_stest+accuracy_stest_closed)
  print("armonic mean", armonic_mean)
  print("\n\n")
  total_armonic_mean.append(armonic_mean)
  ResNet = ResNet.train(True)

print("TOTAL ACCURACY closedWorld: ", total_accuracy_closedWorld)
print("TOTAL ACCURACY openWorld: ", total_accuracy_openWorld)
print("Armonic mean: ", total_armonic_mean)
