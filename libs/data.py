import csv, inspect, json
import os.path, sys
from sklearn.model_selection import train_test_split
import pandas as pd
import PIL
from monai.apps import download_and_extract
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    EnsureType,
)
import numpy as np

import torch
import torch.autograd as autograd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import mnist_noniid

class AGNEWs(Dataset):
    def __init__(self, label_data_path, alphabet_path, l0 = 1014):
        """Create AG's News dataset object.

        Arguments:
            label_data_path: The path of label and data file in csv.
            l0: max length of a sample.
            alphabet_path: The path of alphabet json file.
        """
        self.label_data_path = label_data_path
        self.l0 = l0
        # read alphabet
        self.loadAlphabet(alphabet_path)
        self.load(label_data_path)
        
            
    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        y = self.y[idx]
        return X, y

    def loadAlphabet(self, alphabet_path):
        with open(alphabet_path) as f:
            self.alphabet = ''.join(json.load(f))

    def load(self, label_data_path, lowercase = True):
        self.label = []
        self.data = []
        with open(label_data_path, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            # num_samples = sum(1 for row in rdr)
            for index, row in enumerate(rdr):
                self.label.append(int(row[0]))
                txt = ' '.join(row[1:])
                if lowercase:
                    txt = txt.lower()                
                self.data.append(txt)

        self.y = torch.LongTensor(self.label)


    def oneHotEncode(self, idx):
        # X = (batch, 70, sequence_length)
        X = torch.zeros(len(self.alphabet), self.l0)
        sequence = self.data[idx]
        for index_char, char in enumerate(sequence[::-1]):
            if self.char2Index(char)!=-1:
                X[self.char2Index(char)][index_char] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)

    def getClassWeight(self):
        num_samples = self.__len__()
        label_set = set(self.label)
        num_class = [self.label.count(c) for c in label_set]
        class_weight = [num_samples/float(self.label.count(c)) for c in label_set]    
        return class_weight, num_class
    
class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.y = df['Male'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]
    
class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

def load_dataset(dataset, only_to_tensor = False):
    if only_to_tensor:
        transform=transforms.ToTensor()
    elif dataset.upper() == "MNIST" or dataset.upper() == "FMNIST":
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
    else:
        transform=transforms.ToTensor()

    datadir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/../data'
    train_data, test_data = None, None
    
    if dataset.upper() == "AGNEWS":
        train_path = os.path.abspath(os.path.join(os.getcwd(), datadir+"/ag_news_csv/train.csv"))
        train_data = AGNEWs(label_data_path=train_path, alphabet_path=datadir+"/ag_news_csv/alphabet.json", l0=1014)
        
        test_path = os.path.abspath(os.path.join(os.getcwd(), datadir+"/ag_news_csv/test.csv"))
        test_data = AGNEWs(label_data_path=test_path, alphabet_path=datadir+"/ag_news_csv/alphabet.json", l0=1014)
    if dataset.upper() == "MNIST":
        train_data = datasets.MNIST(root=datadir, train=True, transform=transform, download=True)
        test_data = datasets.MNIST(root=datadir, train=False, transform=transform, download=True)
    if dataset.upper() == "FMNIST":
        train_data = datasets.FashionMNIST(root=datadir, train=True, transform=transform, download=True)
        test_data = datasets.FashionMNIST(root=datadir, train=False, transform=transform, download=True)
        
    if dataset.upper() == "MEDNIST":
        resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
        md5 = "0bc7306e7427e00ad1c5526a6677552d"

        compressed_file = os.path.join(datadir, "MedNIST.tar.gz")
        data_dir = os.path.join(datadir, "MedNIST/MedNIST")
        if not os.path.exists(data_dir):
            download_and_extract(resource, compressed_file, data_dir, md5)
        
        class_names = sorted(x for x in os.listdir(data_dir)
                             if os.path.isdir(os.path.join(data_dir, x)))
        num_class = len(class_names)
        image_files = [
            [
                os.path.join(data_dir, class_names[i], x)
                for x in os.listdir(os.path.join(data_dir, class_names[i]))
            ]
            for i in range(num_class)
        ]
        num_each = [len(image_files[i]) for i in range(num_class)]
        image_files_list = []
        image_class = []
        for i in range(num_class):
            image_files_list.extend(image_files[i])
            image_class.extend([i] * num_each[i])
        num_total = len(image_class)
        image_width, image_height = PIL.Image.open(image_files_list[0]).size

        val_frac = 0.1
        test_frac = 0.1
        length = len(image_files_list)
        indices = np.arange(length)
        np.random.shuffle(indices)

        test_split = int(test_frac * length)
        val_split = int(val_frac * length) + test_split
        test_indices = indices[:test_split]
        val_indices = indices[test_split:val_split]
        train_indices = indices[val_split:]

        train_x = [image_files_list[i] for i in train_indices]
        train_y = [image_class[i] for i in train_indices]
        val_x = [image_files_list[i] for i in val_indices]
        val_y = [image_class[i] for i in val_indices]
        test_x = [image_files_list[i] for i in test_indices]
        test_y = [image_class[i] for i in test_indices]
        
        train_transforms = Compose(
            [
                LoadImage(image_only=True),
                AddChannel(),
                ScaleIntensity(),
                RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
                RandFlip(spatial_axis=0, prob=0.5),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
                EnsureType(),
            ]
        )

        val_transforms = Compose(
            [LoadImage(image_only=True), AddChannel(), ScaleIntensity(), EnsureType()])

        y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
        y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=num_class)])
        
        train_data = MedNISTDataset(train_x, train_y, train_transforms)
        valid_data = MedNISTDataset(val_x, val_y, val_transforms)
        test_data = MedNISTDataset(test_x, test_y, val_transforms)
        
    if dataset.upper() == "CELEBA":
        # 1. Download this file into dataset_directory:
        #  https://www.kaggle.com/jessicali9530/celeba-dataset
        # 2. Put the `img_align_celeba` directory into the `celeba` directory!
        # 3. Dataset directory structure should look like this (required by ImageFolder from torchvision):
        #  +- `dataset_directory`
        #     +- celeba
        #        +- img_align_celeba
        #           +- 000001.jpg
        #           +- 000002.jpg
        #           +- 000003.jpg
        #           +- ...
        
        df1 = pd.read_csv(datadir + '/celeba/list_attr_celeba.txt', sep="\s+", skiprows=1, usecols=['Male'])

        # Make 0 (female) & 1 (male) labels instead of -1 & 1
        df1.loc[df1['Male'] == -1, 'Male'] = 0
        
        df2 = pd.read_csv(datadir + '/celeba/list_eval_partition.txt', sep="\s+", skiprows=0, header=None)
        df2.columns = ['Filename', 'Partition']
        df2 = df2.set_index('Filename')
        
        df3 = df1.merge(df2, left_index=True, right_index=True)
        df3.to_csv(datadir + '/celeba/celeba-gender-partitions.csv')
        df4 = pd.read_csv(datadir + '/celeba/celeba-gender-partitions.csv', index_col=0)
        
        df4.loc[df4['Partition'] == 0].to_csv(datadir + '/celeba/celeba-gender-train.csv')
        df4.loc[df4['Partition'] == 1].to_csv(datadir + '/celeba/celeba-gender-valid.csv')
        df4.loc[df4['Partition'] == 2].to_csv(datadir + '/celeba/celeba-gender-test.csv')
        
        train_data = CelebaDataset(csv_path=datadir + '/celeba/celeba-gender-train.csv',
                              img_dir=datadir + '/celeba/img_align_celeba/',
                              transform=transform)

        valid_data = CelebaDataset(csv_path=datadir + '/celeba/celeba-gender-valid.csv',
                                      img_dir=datadir + '/celeba/img_align_celeba/',
                                      transform=transform)

        test_data = CelebaDataset(csv_path=datadir + '/celeba/celeba-gender-test.csv',
                                     img_dir=datadir + '/celeba/img_align_celeba/',
                                     transform=transform)
        
    if dataset.upper() == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        if only_to_tensor:
            transform_train=transforms.ToTensor()
            transform_test=transforms.ToTensor()

        train_data = datasets.CIFAR10(root=datadir, train=True, transform=transform_train, download=True)
        test_data = datasets.CIFAR10(root=datadir, train=False, transform=transform_test, download=True)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
    if dataset.upper() == "WISCONSIN":
        data_path = os.path.abspath(os.path.join(os.getcwd(), datadir+"/Wisconsin/data.csv"))
        df = pd.read_csv(data_path)
        df.drop(['id','Unnamed: 32'],axis=1,inplace=True)
        lab ={'B':0,'M':1}
        df = df.replace({'diagnosis':lab})

        predictors = df.iloc[:, 2:31]
        labels = df.iloc[:, 0:1]

        predictors_train, predictors_test, labels_train, labels_test = train_test_split(predictors,labels,test_size = 0.20)

        predictors_train = torch.tensor(np.array(predictors_train), dtype=torch.float)
        labels_train = torch.tensor(np.array(labels_train), dtype = torch.float)
        train_data = torch.utils.data.TensorDataset(predictors_train, labels_train)
        
        predictors_test = torch.tensor(np.array(predictors_test), dtype=torch.float)
        labels_test = torch.tensor(np.array(labels_test), dtype = torch.float)
        test_data = torch.utils.data.TensorDataset(predictors_test, labels_test)        

    return train_data, test_data


def random_split(data, ratio):
    split_arr = [int(len(data) * (1-ratio)), int(len(data) * ratio)]
    rem_data = len(data) - sum(split_arr)
    if rem_data > 0:
        split_arr[-1] = split_arr[-1] + rem_data
        
    return torch.utils.data.random_split(data, split_arr)

def split_data(train_data, clients, **kwargs):
    split_arr = [int(len(train_data) / len(clients)) for _ in range(len(clients))]
    rem_data = len(train_data) - sum(split_arr)
    if rem_data > 0:
        split_arr[-1] = split_arr[-1] + rem_data
    
    splitted_data = torch.utils.data.random_split(train_data, split_arr)
    clients_data = {client: splitted_data[index] for index, client in enumerate(clients)}
    
    if "non_iid" in kwargs and kwargs["non_iid"]:
        train_dataset_mnist, _ = mnist_noniid.mnist_extr_noniid(train_data, train_data, len(clients), 10, int(len(train_data)/(len(clients) * 10)), kwargs["rate_unbalance"])
        clients_data = {client: Subset(train_data, train_dataset_mnist[index].astype(int)) for index, client in enumerate(clients)}

    return clients_data


def load_client_data(clients_data, batch_size, test_ratio=None, **kwargs):
    train_loaders = {}
    test_loaders = {}

    if test_ratio is None:
        for client, data in clients_data.items():
            train_loaders[client] = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
            
    else:
        for client, data in clients_data.items():
            train_test = torch.utils.data.random_split(data, [int(len(data) * (1-test_ratio)), 
                                                              int(len(data) * test_ratio)])

            if batch_size != -1:
                train_loaders[client] = torch.utils.data.DataLoader(train_test[0], batch_size=batch_size, shuffle=True, **kwargs)
                test_loaders[client] = torch.utils.data.DataLoader(train_test[1], batch_size=batch_size, shuffle=True, **kwargs)
            else:
                train_loaders[client] = torch.utils.data.DataLoader(train_test[0], batch_size=len(train_test[0]), shuffle=True, **kwargs)
                test_loaders[client] = torch.utils.data.DataLoader(train_test[1], batch_size=len(train_test[1]), shuffle=True, **kwargs)

    return train_loaders, test_loaders