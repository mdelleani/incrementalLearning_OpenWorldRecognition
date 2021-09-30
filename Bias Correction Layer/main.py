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

from model import BiasLayer
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

train_indexes = []
test_indexes = []

print(params.LR)
ResNet=resnet.resnet32(num_classes= params.TASK_CLASSES)
ResNet.to(params.DEVICE)

exemplar_indexes = []
exemplar_images  = []
random.seed(params.SEED)
np.random.seed(params.SEED)
torch.manual_seed(params.SEED)


# one bias layer for each split of the total class, in this case we need 10 bias layers
bias_layer1 = BiasLayer().cuda()
bias_layer2 = BiasLayer().cuda()
bias_layer3 = BiasLayer().cuda()
bias_layer4 = BiasLayer().cuda()
bias_layer5 = BiasLayer().cuda()
bias_layer6 = BiasLayer().cuda()
bias_layer7 = BiasLayer().cuda()
bias_layer8 = BiasLayer().cuda()
bias_layer9 = BiasLayer().cuda()
bias_layer10 = BiasLayer().cuda()
bias_layers=[bias_layer1,bias_layer2,bias_layer3, bias_layer4, bias_layer5, bias_layer6, bias_layer7, bias_layer8, bias_layer9, bias_layer10]



for task in range(n_tasks):
  known_classes= task*params.TASK_CLASSES
  #indexes for this task
  train_indexes =  train_Dataset.__getIndexesGroups__(task* params.TASK_CLASSES) # splits[task]

  test_indexes = test_indexes + test_Dataset.__getIndexesGroups__(task* params.TASK_CLASSES)


  ResNet ,train_set, train_loader,bias_layers = model.UpdateRepresentation(ResNet, task, train_Dataset, train_indexes, exemplar_indexes,bias_layers)
  bias_optimizer = optim.Adam(bias_layers[task].parameters(), lr=0.001)

  m = int(params.K/(known_classes+params.TASK_CLASSES))
  m = int(m + .5)



  for y,Py in enumerate(exemplar_indexes):
    exemplar_indexes[y]=model.ReduceExemplarSet(Py,m)

  exemplar_indexes = model.ConstructExemplarSet(train_Dataset, train_indexes, m, ResNet, known_classes, exemplar_indexes,bias_layers,task)


  #exemplar_means = model.ComputeMeansofExemplars(exemplar_indexes, ResNet,train_Dataset,bias_layers,task)
    #end of task results
  ResNet.eval()

#results on training set
  all_preds = []
  all_labels = []
  n_images = 0
  correct_preds=0
  m = nn.Softmax(dim=1)
  for images, labels, _ in train_loader:
    images = images.float().to(params.DEVICE) ## need to be float
    labels = labels.to(params.DEVICE)
    output = ResNet(images,features = False)
    output=model.bias_forward(output,bias_layers,task)
    probabilities = m(output)
    preds=torch.argmax(probabilities,1)
    n_images += len(images)
    all_preds = np.concatenate((all_preds,preds.cpu()))
    all_labels = np.concatenate((all_labels,labels.cpu()))
  correct_preds += (all_preds == all_labels).sum()
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
    output=model.bias_forward(output,bias_layers,task)
    probabilities = m(output)
    preds=torch.argmax(probabilities,1)
    n_images += len(images)
    all_preds = np.concatenate((all_preds,preds.cpu()))
    all_labels = np.concatenate((all_labels,labels.cpu()))
  correct_preds += (all_preds == all_labels).sum()
  accuracy = correct_preds/n_images
  print(f"accuracy on test set: {accuracy}")


  #confusion matrix
  cm = confusion_matrix(all_labels,all_preds)
  df_cm = pd.DataFrame(cm, range((task+1)*params.TASK_CLASSES), range((task+1)*params.TASK_CLASSES))
  plt.figure(figsize = (10,7))
  sn.heatmap(df_cm, annot=False, cmap="viridis")
  plt.savefig(f"{task}_cf")
  ResNet = ResNet.train(True)
