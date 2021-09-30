import torch
import numpy as np
import random
from PIL import Image
import params


# Scheletro preso da : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel


class Dataset(torch.utils.data.Dataset):
	'Characterizes a dataset for PyTorch'

	def __init__(self,dataset, classes_per_task = 10,transform=None):
		'Initialization'
		self._dataset = dataset
		self._data = self._dataset.data
		self._targets = np.array(self._dataset.targets) # list -> numpy array
		self._classes = np.array(self._dataset.classes)  # list -> numpy array (str of classes in Dataset)
		self.splits = self.getSplits(classes_per_task)
		self.dictionary = self.getDictionary()
		self.new_targets = self.mapTargets()
		self.transform=transform

	def __len__(self):
		'Denotes the total number of samples'
		return len(self._targets)

	def __getitem__(self, index):
		'Generates one sample of data'
		# Select sample
		img = self._data[index]
		if self.transform is not None:
		    img=self.transform( Image.fromarray(img))

		label = self._targets[index]
		return img, label, index

	def __getitem_mapped__(self, index):
		'Generates one sample of data'
		# Select sample
		img = self._data[index]
		if self.transform is not None:
		    img=self.transform( Image.fromarray(img))
		label = self.new_targets[index]

		return img, torch.tensor(label), torch.tensor(index)


	def getSplits(self,classes_per_task):
		num_classes = len(self._classes)
		assert (num_classes%classes_per_task)==0, "The number of classes per task is not a dividendo"
		num_splits = int(num_classes/classes_per_task)
		remaining_classes = range(num_classes)
		splits  = [None] * num_splits

		for i in range(0,num_splits):
			random.seed(params.SEED)
			split = random.sample(remaining_classes, classes_per_task)
			splits[i] = split
			remaining_classes = list( set(remaining_classes) - set(split) )

		return np.array(splits)



	def __getIndexesGroups__(self, index = 0):
		#This method returns a list containing the indexes of all the images belonging to classes [starIndex, startIndex + 10]
		indexes = []
		self.searched_classes = self.splits[int(index/10)]
		i = 0
		for el in self._targets:
			if (el in self.searched_classes):
				indexes.append(i)
			i+=1
		return indexes

	def getDictionary(self):
		dictionary = {}
		splits = np.concatenate(self.splits)
		for i in range (params.TOTAL_CLASSES):
			dictionary[splits[i]] = i
		return dictionary

	def mapTargets(self):
		new_targets = []
		for t in self._targets:
			new_targets.append(self.dictionary[t])
		return new_targets

class Subset(Dataset):
	r"""
	Subset of a dataset at specified indices.
	Arguments:
		dataset (Dataset): The whole Dataset
		indices (sequence): Indices in the whole set selected for subset
	"""
	def __init__(self, dataset, indices,transform):
		self._dataset = dataset
		self._indices = indices
		self._data = dataset._data
		self._targets = dataset.new_targets
		self.transform=transform


	def __getitem__(self, idx):
		img = self._data[self._indices[idx]]
		if self.transform is not None:
		    img=self.transform( Image.fromarray(img))
		labels = self._targets[self._indices[idx]]
		return img, labels, idx

	def __len__(self):
		return len(self._indices)
