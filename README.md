# IncrementalLearning & OpenWorldRecognition
## Abstract
Image classifcation is one of the most challenging fields in artificial intelligence. In particular, two of the most studied and difficult tasks are incremental learning and open world recognition: the former consists in incrementally adding new classes to a model and the problem, which is known in literature as catastrophic forgetting, occurs when the model forgets about its known classes while updating its parame- ters in respect to the new ones. The latter refers to the capability of a model to recognize if an image belongs to a class that hasn't learned yet, so that the image can be classified as unknown. The aim of this project is to explore these two problems, with the help of some of the most famous re- lated papers, and to create a framework that combines the two tasks, which will be able t o learn new classes incremen- tally and at the same time to reject images from unknown classes. Finally, we propose our own modification

## Architecture:
ResNet-32
## Dataset:
CIFAR100

## [Master](https://github.com/MattiaDelleani/incrementalLearning_OpenWorldRecognition/tree/master)
Implementation of:
- Fine Tuning
- Learning without Forgetting
- iCarL

## [Ablation Studies](https://github.com/MattiaDelleani/incrementalLearning_OpenWorldRecognition/tree/ablationStudies)
This section describes the functioning of the classifiers and losses tested on the iCaRL framework.

## [Open World Recognition](https://github.com/MattiaDelleani/incrementalLearning_OpenWorldRecognition/tree/openWorldRecognition)
In the open world setting the classes are split in two groups
made by 50 classes each. The first group is the closed world,
is the only set of classes that the model will use to train
its parameters, and is also used for evaluation. The open
world instead is a set of classes never used for training, so
is only used in the test phase.
The ideal solution is: 
> The model doesn't reject any sample from the closed world test set, because they belong to known classes.

> The model rejects all the samples from the open world test set, because it has never trained on those classes.

Implementation of:
- Cosine Distances
- Cosine Similarity
- Euclidean Distances
- FullyConnected + Threshold

__Our modification__
> A sample is rejected if it's distance margin is less than the mean of the distance margins of the training set.

See chapter 9.2 of [Report](Report.pdf)
