import copy

import numpy as np
import torch
import time
import os
from flcore.clients.clientALA import *
from utils.dataset import load_data
from threading import Thread


def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x


class FedALA(object):
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_train_loss = []
        self.avg_acc = []

        self.times = times
        self.eval_gap = args.eval_gap

        self.set_clients(args, clientALA)


        self.BA_weight = []
        self.BA_memory = []
        self.BA_memoryF1 = []
        self.client_acc_newest = []
        self.client_acc_newest_aftertrain = []


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.Fit done!!")

        self.Budget = []
        self.id_client_old = []

    def fit(self, ):
        for c in self.clients:
            c.fit()

        self.selected_clients = self.select_clients()
        self.receive_models()

        self.aggregate_parameters()

    def train(self):
        for i in range(self.global_rounds + 1):
            print(f"\n-------------Round number: {i}-------------")
            s_t = time.time()

            self.selected_clients = self.select_clients(i)

            print("\nSend global model")

            self.send_models()

            if i != 0 and i % self.eval_gap == 0:
                print("\nEvaluate global model")
                self.evaluate()
                self.cul_BA_weight(i)
                print(self.client_acc_newest)

            print("\nTrain global model")
            threads = [Thread(target=client.train)
                       for client in self.selected_clients]
            [t.start() for t in threads]
            [t.join() for t in threads]



            self.evaluate_new()
            for c in self.selected_clients:
                print("c.accuracy_new", c.accuracy_new)
            self.client_acc_newest_aftertrain = [c.accuracy_new[-1] for c in self.selected_clients]
            print(self.client_acc_newest_aftertrain)
            self.client_acc_newest_aftertrain = softmax(self.client_acc_newest_aftertrain)
            print(self.client_acc_newest_aftertrain)

            # print("\nTrain global model")
            # for client in self.selected_clients:
            #     client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print(self.Budget)
            print('-' * 50, self.Budget[-1])

        print("\nBest global accuracy.")
        print("\n最大acc:", max(self.rs_test_acc))
        print(sum(self.Budget[2:]) / len(self.Budget[2:]))
        print("\nBA_precision:", self.BA_memory)
        print("\nBA_F1score:", self.BA_memoryF1)

        self.cul_matrix()

    def cul_matrix(self, ):
        avg_pre = []
        avg_acc = []
        avg_recall = []
        avg_F1 = []
        avg_FNR = []
        avg_FPR = []
        avg_FAR = []
        avg_loss = []
        print('avg_acc1', self.avg_acc)


    def cul_BA_weight(self, i):
        everyClientBAPre = [c.BA_pre for c in self.selected_clients]
        self.BA_memory.append(sum(everyClientBAPre) / len(self.selected_clients))
        print(f'\n第{i}轮对Botnet ARES的预测准确度:{sum(everyClientBAPre) / len(self.selected_clients)}')
        everyClientBAPre = softmax([c.BA_pre for c in self.clients])
        self.BA_weight = everyClientBAPre
        # BAF1
        everyClientBF1 = [c.BA_F1 for c in self.selected_clients]
        self.BA_memoryF1.append(sum(everyClientBF1) / len(self.selected_clients))
        print(f'\n第{i}轮对Botnet ARES的预测F1score:{sum(everyClientBF1) / len(self.selected_clients)}')
        everyClientBF1 = softmax([c.BA_F1 for c in self.clients])
        self.BA_weight = everyClientBF1

        # self.client_acc_newest = softmax([c.accuracy[-1] for c in self.selected_clients])  # newest

    def set_clients(self, args, clientObj):
        DATA_DIR = os.path.join(os.path.abspath('../'), "data")
        list_train_loader, list_test_loader = load_data(
            data_path=DATA_DIR,
            balanced=False,
            batch_size=1024,
        )
        for i in range(self.num_clients):
            # train_data = read_client_data(self.dataset, i, is_train=True)
            # test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args,
                               id=i,
                               train_loader=list_train_loader[i],
                               valid_loader=list_test_loader[i],
                               test_loader=list_test_loader[i])
            self.clients.append(client)

    def select_clients(self, i):
        if i >= 10:
            selected_clients = self.clients[10:]
        else:
            selected_clients = self.clients[:20]
        print("\nselected_clients:", len(selected_clients))
        id_client_new = []
        for c in selected_clients:
           id_client_new.append(c.id)
        print("\nselected_clients_id:", id_client_new)
        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        threads = []
        for client in self.selected_clients:
            t = Thread(target=client.local_initialization, args=(self.global_model,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

    def add_parameters(self, w, client_model):
        params_g = []
        params_c = []
        for m in self.global_model.models:
            params_g += list(m.parameters())
        for m in client_model.models:
            params_c += list(m.parameters())

        params_g += list(self.global_model.parameters())
        params_c += list(client_model.parameters())

        for server_param, client_param, acc_w in zip(params_g, params_c, self.client_acc_newest_aftertrain):
        #for server_param, client_param in zip(params_g, params_c):
            # server_param.data += client_param.data.clone() * (w * 0.5 + BA_w * 0.5)
            server_param.data += client_param.data.clone() * (w * 0.5 + acc_w * 0.5)
            # server_param.data = client_param.data.clone() * w
            # print('\nw:',w ,'acc_w:', acc_w)

    def add_parameters_0(self, w, client_model):
        params_g = []
        params_c = []
        for m in self.global_model.models:
            params_g += list(m.parameters())
        for m in client_model.models:
            params_c += list(m.parameters())

        params_g += list(self.global_model.parameters())
        params_c += list(client_model.parameters())

        for server_param, client_param, acc_w in zip(params_g, params_c, self.client_acc_newest_aftertrain):
        #for server_param, client_param in zip(params_g, params_c):
            server_param.data = client_param.data.clone() * (w * 0.5 + acc_w * 0.5)

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        self.add_parameters_0(self.uploaded_weights[0], self.uploaded_models[0]) 
        # for param in self.global_model.parameters():
        #     param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights[1:], self.uploaded_models[1:]):
            self.add_parameters(w, client_model)

    def aggregate_parameters_new(self):
        assert (len(self.uploaded_models) > 0)
        self.add_parameters_0(self.uploaded_weights[0], self.uploaded_models[0]) 

        for w, client_model in zip(self.uploaded_weights[1:], self.uploaded_models[1:]):
            self.add_parameters(w, client_model)


    def test_client(self, c, tot_correct, num_samples):
        ct, ns, lo = c.test_metrics()
        print(f'Client {c.id}: Acc: {ct * 1.0 / ns}, loss: {lo}')
        tot_correct.append(ct * 1.0)
        num_samples.append(ns)

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []


        for c in self.selected_clients:
           ct, ns, lo = c.test_metrics()
           tot_correct.append(ct * 1.0)
           num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            print(f'Client {c.id}: Train loss: {cl * 1.0 / ns}')
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        self.avg_acc.append(test_acc)
        print(self.avg_acc)
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        # print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        # print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def test_client_new(self, c):
        c.test_metrics_new()

    def test_metrics_new(self):
        threads = []
        for c in self.selected_clients:
            t = Thread(target=self.test_client_new, args=(c,))      
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def evaluate_new(self, acc=None, loss=None):
        stats = self.test_metrics_new()