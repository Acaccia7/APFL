import copy
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_curve, confusion_matrix
from tqdm import tqdm

# from utils.data_utils import read_client_data
sys.path.insert(0, "/home/ubuntu/wyx/PFLDBN0807")
#print(sys.path)
# print(sys.path)

from utils.ALA import ALA
from utils import visualisation
from sklearn.metrics import recall_score

class clientALA(object):
    def __init__(self, args, id, train_loader, valid_loader, test_loader):
        self.model = copy.deepcopy(args.model)
        # self.dataset = args.dataset
        self.device = args.device
        self.id = id

        self.num_classes = args.num_classes

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader

        self.train_samples = len(train_loader.dataset)

        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx

        # BA_precision
        self.BA_pre = 0
        self.BA_F1 = 0

        self.precision = []
        self.accuracy = []
        self.accuracy_new = []
        self.recall = []
        self.F1 = []
        self.FNR = []
        self.FPR = []
        self.loss_epoch = []
        self.FAR = []

        self.flag_p = 1

        # train_data = read_client_data(self.dataset, self.id, is_train=True)
        self.ALA = ALA(self.id, self.loss, self.train_loader, self.batch_size,
                       self.rand_percent, self.layer_idx, self.eta, self.device, 0.01, 5)


    def fit(self):
        print(f'\n第{self.id}个客户端开始fit！')
        self.model.fit(self.train_loader)
        print(f'\n第{self.id}个客户端fit完毕！！')

    def train(self):
        weights = torch.ones(6)
        weights[1] = 20  
        weights[5] = 10  
        criterion = nn.CrossEntropyLoss(reduction='mean', weight=weights.cuda())
        optimizer = [
            getattr(torch.optim, 'Adam')(params=m.parameters(), lr=0.1, weight_decay=0, amsgrad=False)
            for m in self.model.models
        ]
        optimizer.append(getattr(torch.optim, 'Adam')(params=self.model.parameters(), lr=0.01, weight_decay=0, amsgrad=False))

        self.model.to(self.device)
        for step in range(self.local_steps):
            self.model.train()

            for inputs, labels in tqdm(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.squeeze(1)

                # zero the parameter gradients
                for opt in optimizer:
                    opt.zero_grad()

                # Passing the batch down the model
                outputs = self.model(inputs)

                # forward + backward + optimize
                loss = criterion(outputs, labels)
                loss.backward()

                # For every possible optimizer performs the gradient update
                for opt in optimizer:
                    opt.step()
        print(f'\nThe number {self.id} client is trained.')

    def local_initialization(self, received_global_model):
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)

    def test_metrics(self, model=None):
        # testloader = self.load_test_data()
        if model == None:
            model = self.model

        model.eval()

        test_loss = 0
        test_steps = 0
        test_total = 0
        test_correct = 0

        test_output_pred = []
        test_output_true = []
        test_output_pred_prob = []

        with torch.no_grad():
            for (inputs, labels) in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.squeeze(1)

                outputs = model(inputs)
                criterion = nn.CrossEntropyLoss(reduction='mean')
                loss = criterion(outputs, labels)
                test_loss += loss.cpu().item()
                test_steps += 1

                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                test_output_pred += outputs.argmax(1).cpu().tolist()
                test_output_true += labels.tolist()

        # test_output_pred_prob += nn.functional.softmax(outputs, dim=0).cpu().tolist()
        Labels = ["Benign", "Botnet ARES", "Brute Force", "DoS/DDoS", "PortScan", "Web Attack"]
        visualisation.plot_confusion_matrix(
            y_true=test_output_true,  # true label
            y_pred=test_output_pred,  # predicted value
            labels=Labels,
            save=True,
            save_dir='../images',
            filename=f'deep_belief_network_test_confusion_matrix{self.id}.pdf'
        )

        y_true_i = [int(label == 1) for label in test_output_true]
        y_pred_i = [int(label == 1) for label in test_output_pred]
        pre = recall_score(y_true_i, y_pred_i, pos_label=1)
        BA_F1 = f1_score(y_true_i, y_pred_i, pos_label=1)
        self.BA_pre = pre
        self.BA_F1=BA_F1
        acc = accuracy_score(test_output_true, test_output_pred)

        precision = precision_score(test_output_true, test_output_pred, average='weighted')

        recall = recall_score(test_output_true, test_output_pred, average='weighted')

        f1 = f1_score(test_output_true, test_output_pred, average='weighted')

        cm = confusion_matrix(test_output_true, test_output_pred)

        tp = np.diag(cm)
        fn = np.sum(cm, axis=1) - tp
        fp = np.sum(cm, axis=0) - tp
        tn = np.sum(cm) - tp - fn - fp

        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        macro_fpr = np.mean(fpr)
        macro_fnr = np.mean(fnr)

        micro_fpr = np.sum(fp) / (np.sum(fp) + np.sum(tn))
        micro_fnr = np.sum(fn) / (np.sum(fn) + np.sum(tp))

        far = fp / (fp + tp)

        macro_far = np.mean(far)

        micro_far = np.sum(fp) / (np.sum(fp) + np.sum(tp))


        self.precision.append(precision)
        self.accuracy.append(acc)
        self.recall.append(recall)
        self.F1.append(f1)
        self.FNR.append(micro_fnr)
        self.FPR.append(micro_fpr)
        self.loss_epoch.append(test_loss / test_steps)
        self.FAR.append(micro_far)

        return test_correct, test_total, test_loss / test_steps

    def train_metrics(self, model=None):
        if model == None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in self.train_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y.squeeze())
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def test_metrics_new(self, model=None):
        if model == None:
            model = self.model

        model.eval()

        test_loss = 0
        test_steps = 0
        test_total = 0
        test_correct = 0

        test_output_pred = []
        test_output_true = []
        test_output_pred_prob = []

        with torch.no_grad():
            for (inputs, labels) in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.squeeze(1)

                outputs = model(inputs)
                criterion = nn.CrossEntropyLoss(reduction='mean')
                loss = criterion(outputs, labels)
                test_loss += loss.cpu().item()
                test_steps += 1

                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                test_output_pred += outputs.argmax(1).cpu().tolist()
                test_output_true += labels.tolist()

        Labels = ["Benign", "Botnet ARES", "Brute Force", "DoS/DDoS", "PortScan", "Web Attack"]

        acc = accuracy_score(test_output_true, test_output_pred)

        self.accuracy_new.append(acc)

