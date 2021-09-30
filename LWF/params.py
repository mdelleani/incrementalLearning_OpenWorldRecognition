from torchvision import transforms


DEVICE = 'cuda'
TOTAL_CLASSES = 100
TASK_CLASSES = 10

assert (TOTAL_CLASSES % TASK_CLASSES)== 0, "Il numero di task dev'essere un divisore intero del numero totale di classi"

N_TASKS = int(TOTAL_CLASSES/TASK_CLASSES)

BATCH_SIZE = 128
N_EPOCHS = 80
WEIGHT_DECAY = 1e-5
LR = 2
STEP_SIZE = [ 49,63 ]
GAMMA = 0.2
MOMENTUM = 0.9

NUM_WORKERS = 0
SEED = 42

#transformation for train and test set, they are used multiple times
transform_train = transforms.Compose([transforms.RandomCrop(size= 32, padding= 4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023))])

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023))])
