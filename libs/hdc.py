import enum
import torch
import numpy as np
from functools import partial
from multiprocessing import Pool, Process
from sklearn.metrics import classification_report

class Alpha(enum.Enum):
    Const = 0
    CosRow = 1
    CosCol = 2
    ICosRow = 3
    ICosCol = 4
    SinRow = 5
    SinCol = 6

class HDC(torch.nn.Module):
    def __init__(self, img_len, hvd, num_classes, device):
        super(HDC, self).__init__()
        #self.data = []
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.num_classes = num_classes
        self.hvd=hvd
        self.b_proj = np.random.randint(2, size=(img_len, hvd))
        self.b_proj[np.isclose(self.b_proj, 0)] = -1
        #self.proj = torch.from_numpy(self.b_proj).type(torch.FloatTensor)
        self.proj = torch.rand((img_len, hvd), device=device)
        self.proj_inv = self.get_proj_inv()
        self.train_vectors = torch.zeros((num_classes, hvd), device=device)
        
    def get_proj_inv(self):
        proj_trans = torch.transpose(self.proj, 0, 1)
        proj_mul_trans = self.proj @ proj_trans
        proj_mul_trans_inv = torch.inverse(proj_mul_trans)
        proj_inv = proj_trans @ proj_mul_trans_inv
        return proj_inv
        
    def avg(self, client_model_updates):
        for model_update in client_model_updates:
            self.train_vectors += model_update.train_vectors
        
    def train(self, train_loader, device):
        x_train, y_train = next(iter(train_loader))
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        hdc_train = x_train.reshape(x_train.shape[0], -1) @ self.proj

        for i in range(x_train.shape[0]):
            self.train_vectors[y_train[i]] += hdc_train[i]

        return self.test(train_loader, device)
    
    def re_train_p(self, alpha, mal_train_vectors, row, col):
        if alpha is Alpha.CosRow:
            alpha = self.cos(mal_train_vectors[row][col], self.train_vectors[row])
        if alpha is Alpha.CosCol:
            alpha = self.cos(mal_train_vectors[row][col], self.train_vectors[col])
        if alpha is Alpha.ICosRow:
            alpha = 1 - self.cos(mal_train_vectors[row][col], self.train_vectors[row])
        if alpha is Alpha.ICosCol:
            alpha = 1 - self.cos(mal_train_vectors[row][col], self.train_vectors[col])
        if alpha is Alpha.SinRow:
            alpha = (1 - self.cos(mal_train_vectors[row][col], self.train_vectors[row])**2)**0.5
        if alpha is Alpha.SinCol:
            alpha = (1 - self.cos(mal_train_vectors[row][col], self.train_vectors[col])**2)**0.5
        else:
            alpha = 0.5

        print([self.cos(mal_train_vectors[row][col], self.train_vectors[i:i+1]) for i in range(self.num_classes)])
        self.train_vectors[row] += alpha * mal_train_vectors[row][col]
        self.train_vectors[col] -= alpha * mal_train_vectors[row][col]
    
    def re_train(self, train_loader, device, alpha = Alpha.Const):
        x_train, y_train = next(iter(train_loader))
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        hdc_train = x_train.reshape(x_train.shape[0], -1) @ self.proj
        pred = torch.stack([self.cos(hdc_train, self.train_vectors[i:i+1]) for i in range(self.num_classes)], axis=-1)
        print (pred)

        mal_train_vectors = torch.zeros((self.num_classes, self.num_classes, self.hvd),device=device)
        
        out = pred.argmax(axis=1) == y_train
        indices = (out == False).nonzero(as_tuple=True)[0]

        for i in indices:
            mal_train_vectors[y_train[i]][pred.argmax(axis=1)[i]] += hdc_train[i]
        
        for row, true_class in enumerate(mal_train_vectors):
            with Pool(len(true_class)) as p:
                func = partial(self.re_train_p, alpha, mal_train_vectors, row)
                p_models = p.map(func, [col for col, mis_class in enumerate(true_class)])
                p.close()
                p.join()

        return self.test(train_loader, device)
    
    def test(self, test_loader, device):
        x_test, y_test = next(iter(test_loader))
        x_test, y_test = x_test.to(device), y_test.to(device)
        
        x_test = x_test.reshape(x_test.shape[0], -1) @ self.proj
        pred = torch.stack([self.cos(x_test, self.train_vectors[i:i+1]) for i in range(self.num_classes)], axis=-1)
        #acc = 100 * torch.mean((pred.argmax(axis=1) == y_test).float())
        return classification_report(y_test, pred.argmax(axis=1))