import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
except Exception as e:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler

with open('Fedtest_features.pkl', 'rb') as f:
    Fed_test_x = pickle.load(f)

with open('Fedtest_labels.pkl', 'rb') as f:
    Fed_test_y = pickle.load(f)

undersampler = RandomUnderSampler(random_state=42)

X_undersampled, y_undersampled = undersampler.fit_resample(Fed_test_x, Fed_test_y)

oversampler = RandomOverSampler(random_state=42)

X_oversampled, y_oversampled = oversampler.fit_resample(Fed_test_x, Fed_test_y)

Fed_test_x = X_oversampled
Fed_test_y = y_oversampled

def split_noniid(train_idcs, train_labels, alpha, n_clients):
    '''
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    n_classes = 6
    print("n_classes:", n_classes)

    label_distribution = np.random.dirichlet([alpha] * num_clients, n_classes)

    class_idcs = [np.where(train_labels == y)[0] for y in range(n_classes)]
    # class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten()
    #        for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    return client_idcs


num_clients = 5
alpha = 0.5 

X_train_clients = [[] for _ in range(num_clients)]
y_train_clients = [[] for _ in range(num_clients)]
X_test_clients = [[] for _ in range(num_clients)]
y_test_clients = [[] for _ in range(num_clients)]

X_train, X_test, y_train, y_test = train_test_split(Fed_test_x, Fed_test_y, test_size= 9 / 10, random_state=0,
                                                    stratify=Fed_test_y)

client_idcs = split_noniid(np.arange(len(X_train)), y_train, alpha, num_clients)

for i, idcs in enumerate(client_idcs):
    # X_train_clients[i], y_train_clients[i] = X_train[idcs], y_train[idcs]
    X_train_clients[i], y_train_clients[i] = X_train.iloc[idcs], y_train.iloc[idcs]

for i in range(num_clients):
    X_train_clients[i], X_test_clients[i], y_train_clients[i], y_test_clients[i] = train_test_split(X_train_clients[i],
                                                                                                    y_train_clients[i],
                                                                                                    test_size=0.25,
                                                                                                    random_state=0,
                                                                                                    stratify=
                                                                                                    y_train_clients[i])


labels_per_client = [df.iloc[:, 0].tolist() for df in y_train_clients]


num_clients = len(labels_per_client)
num_classes = 6  

label_counts = np.zeros((num_clients, num_classes))
for i, labels in enumerate(labels_per_client):
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts[i, unique_labels] = counts

labels = np.arange(num_classes)
plt.bar(np.arange(num_clients), label_counts[:, 0], align='center', alpha=0.5)
for i in range(1, num_classes):
    plt.bar(np.arange(num_clients), label_counts[:, i], bottom=np.sum(label_counts[:, :i], axis=1), align='center',
            alpha=0.5)

plt.title("Distribution of Labels across Clients")
plt.xlabel("Clients")
plt.ylabel("Frequency")

plt.xticks(np.arange(num_clients))
plt.gca().set_xticklabels(["Client {}".format(i) for i in range(num_clients)])

plt.legend(["Label {}".format(i) for i in range(num_classes)])

clientnumbers= str(labels_per_client)

plt.show()
######################

print(len(X_train))
for i in range(num_clients):
    print(len(X_train_clients[i]))

DATA_DIR = os.path.join(os.path.abspath("."), "data")
i = 0
for x_tr_c, y_tr_c in zip(X_train_clients, y_train_clients):
    x_tr_c.to_pickle(os.path.join(DATA_DIR, 'processed', f'train/train_features{i}.pkl'))
    y_tr_c.to_pickle(os.path.join(DATA_DIR, 'processed', f'train/train_labels{i}.pkl'))
    i += 1

i = 0
for x_te_c, y_te_c in zip(X_test_clients, y_test_clients):
    x_te_c.to_pickle(os.path.join(DATA_DIR, 'processed', f'test/test_features{i}.pkl'))
    y_te_c.to_pickle(os.path.join(DATA_DIR, 'processed', f'test/test_labels{i}.pkl'))
    i += 1
pass

save_path = os.path.join(DATA_DIR, 'split0.5.png')
plt.savefig(save_path,dpi=300)