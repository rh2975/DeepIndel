import numpy as np
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import nn

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

"""	
class Model(Module):
    def __init__(self,num_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y
"""	

class Model(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(Model, self).__init__()

        def lenet_with_dropout():
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(256, 120)
            self.relu3 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout_rate)  # added dropout layer
            self.fc2 = nn.Linear(120, 84)
            self.relu4 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout_rate)  # added dropout layer
            self.fc3 = nn.Linear(84, num_classes)
            self.relu5 = nn.ReLU()        

        def resnet50():
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, num_classes)
        
        # lenet_with_dropout()
        resnet50()


    def forward(self, x):
        def lenet_forward():
            y = self.conv1(x)
            y = self.relu1(y)
            y = self.pool1(y)
            y = self.conv2(y)
            y = self.relu2(y)
            y = self.pool2(y)
            y = y.view(y.shape[0], -1)
            y = self.fc1(y)
            y = self.relu3(y)
            y = self.dropout1(y)  # added dropout layer
            y = self.fc2(y)
            y = self.relu4(y)
            y = self.dropout2(y)  # added dropout layer
            y = self.fc3(y)
            y = self.relu5(y)
            return y

        def resnet_forward():
            y = self.conv1(x)
            y = self.bn1(y)
            y = self.relu(y)
            y = self.maxpool(y)

            # print(f"Before layer1: y_shape = {y.size()}")
            # print(f"self.layer_shape = {self.layer1(y).size()}")

            y = self.layer1(y)# + y
            y = self.layer2(y)# + y
            y = self.layer3(y)# + y
            y = self.layer4(y)# + y

            y = self.avgpool(y)
            y = y.view(y.shape[0], -1)
            y = self.fc(y)
            return y

        # y = lenet_forward()
        y = resnet_forward()

        return y




def convert_nxn(original_array):
    total_images = original_array.shape[0]
    # print("total_images: {}".format(total_images))

    padded_arr = np.pad(original_array, ((0,0), (3,1), (3,3)), mode='constant')

    # Resize the padded array to (total_images, 30, 30)
    new_arr = np.zeros((total_images, 1, 30, 30))

    for i in range(total_images):
        new_arr[i, 0] = np.resize(padded_arr[i], (30, 30))

    return new_arr


def train_test_data_splitter1(data_array,data_label,train_percent,validation_percent):

    data_array , data_label = shuffler(data_array,data_label)

    total_images = data_array.shape[0]

    train_size = int(total_images * train_percent)

    train_data = data_array[:train_size]
    train_label = data_label[:train_size]
    

    test_data = data_array[train_size:]
    test_label = data_label[train_size:]


    valid_size = int(train_size * validation_percent)
    # randomly selecting validation data from train data
    shuffled_indices = np.random.permutation(train_size)
    valid_indices = shuffled_indices[:valid_size]
    
    valid_data = train_data[valid_indices]
    valid_label = train_label[valid_indices]
    

    return train_data,train_label,test_data,test_label,valid_data,valid_label

def train_test_data_splitter(data_array,data_label,train_percent,validation_percent):

    data_array , data_label = shuffler(data_array,data_label)

    total_images = data_array.shape[0]

    train_size = int(total_images * train_percent)
    train_data = data_array[:train_size]
    train_label = data_label[:train_size]

    validation_size = int(total_images * validation_percent)
    valid_data = data_array[train_size:train_size+validation_size]
    valid_label = data_label[train_size:train_size+validation_size]

    test_data = data_array[train_size+validation_size:]
    test_label = data_label[train_size+validation_size:]


    return train_data,train_label,test_data,test_label,valid_data,valid_label

def data_merger(data1,label1,data2,label2):
    data = np.concatenate((data1,data2),axis=0)
    label = np.concatenate((label1,label2),axis=0)
    return data,label

def shuffler(data,label):
    shuffled_indices = np.random.permutation(data.shape[0])
    data = data[shuffled_indices]
    label = label[shuffled_indices]
    return data,label

def randomnly_select_n_images(data,label,n):
    shuffled_indices = np.random.permutation(data.shape[0])
    data = data[shuffled_indices]
    label = label[shuffled_indices]
    return data[:n],label[:n]

TRUE = POSITIVE = 1
FALSE = NEGATIVE = 0

def recall_score(y_true, y_pred):

    tp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == TRUE and y_pred[i] == POSITIVE:
            tp += 1
        if y_true[i] == TRUE and y_pred[i] == NEGATIVE:
            fn += 1

    return tp / (tp + fn) if (tp + fn) != 0 else -1

def precision_score(y_true, y_pred):
    
    tp = 0
    fp = 0
    for i in range(len(y_true)):
        if y_true[i] == TRUE and y_pred[i] == POSITIVE:
            tp += 1
        if y_true[i] == FALSE and y_pred[i] == POSITIVE:
            fp += 1

    return tp / (tp + fp) if (tp + fp) != 0 else -1

def print_scores(y_true, y_pred):
    print("\t\t  accuracy : {:.3f}".format(accuracy_score(y_true, y_pred)))
    print("\t\t  f1 score : {:.3f}".format(f1_score(y_true, y_pred,average='macro')))
    print("\t\t  precision: {:.3f}".format(precision_score(y_true, y_pred)))
    print("\t\t  recall   : {:.3f}".format(recall_score(y_true, y_pred)))
    print(classification_report(y_true, y_pred))
    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 512
    all_epoch = 200
    threshold_sample_size = 19700

    in_file = 'in1.npz'
    del_file = 'del1.npz'
    nonIndel_file = 'nonIndel.npz'

    print("in data shape: {}".format(np.load(in_file)['data'].shape))
    print("del data shape: {}".format(np.load(del_file)['data'].shape))
    print("nonIndel data shape: {}".format(np.load(nonIndel_file)['data'].shape))


    in_data , in_label = np.load(in_file)['data'], np.load(in_file)['label']
    del_data , del_label = np.load(del_file)['data'], np.load(del_file)['label']
    nonIndel_data , nonIndel_label = np.load(nonIndel_file)['data'], np.load(nonIndel_file)['label']

    truncated_in_data, truncated_in_label = randomnly_select_n_images(in_data,in_label,threshold_sample_size + 11300)
    in_data, in_label = truncated_in_data, truncated_in_label
    
    # truncated_del_data, truncated_del_label = randomnly_select_n_images(del_data,del_label,threshold_sample_size + 3000)
    # del_data, del_label = truncated_del_data, truncated_del_label

    truncated_nonIndel_data, truncated_nonIndel_label = randomnly_select_n_images(nonIndel_data,nonIndel_label,threshold_sample_size)
    nonIndel_data, nonIndel_label = truncated_nonIndel_data, truncated_nonIndel_label

    print("Truncated In data shape: {}".format(in_data.shape))
    print("Truncated del data shape: {}".format(del_data.shape))
    print("Truncated nonIndel data shape: {}".format(nonIndel_data.shape))

    in_data = convert_nxn(in_data)
    del_data = convert_nxn(del_data)
    nonIndel_data = convert_nxn(nonIndel_data)

    indel_data, indel_label = data_merger(in_data,in_label,del_data,del_label)
    full_data, full_label = data_merger(indel_data,indel_label,nonIndel_data,nonIndel_label)
    full_data, full_label = shuffler(full_data, full_label)

    print("full_data shape: {}".format(full_data.shape))



    train_percent = 0.75
    valid_percent = 0.15

    train_data,train_label,test_data,test_label,validation_data,validation_label = train_test_data_splitter(full_data, full_label ,train_percent,valid_percent)

  
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
    validation_dataset = torch.utils.data.TensorDataset(torch.from_numpy(validation_data), torch.from_numpy(validation_label))

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    
    # print("type",type(train_label))
    # print("type",type(validation_label))

    # print(train_label[:10])
    # print(validation_label[:10])


    print("train size: {}".format(len(train_dataset)))
    print("test size: {}".format(len(test_dataset)))
    print("validation size: {}".format(len(validation_dataset)))
    
    print("=========================================")
    
    num_classes = 3 # 0 for nonIndel , 1 for Insertion and 2 for Deletion

    model = Model(num_classes).to(device)
    sgd = SGD(model.parameters(), lr=1e-1)
    loss_fn = CrossEntropyLoss()

    prev_acc = 0
    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)

            sgd.zero_grad()
            predict_y = model(train_x.float())

            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            sgd.step()

        model.eval()

        print('Epoch: {}'.format(current_epoch+1))

        # =========================================        training model    =========================================
        all_predict_y = []
        all_train_label = []
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            predict_y = model(train_x.float()).detach()
            predict_y =torch.argmax(predict_y, dim=-1)
            all_predict_y.append(predict_y)
            all_train_label.append(train_label)
      
        all_predict_y = torch.cat(all_predict_y, dim=0)
        all_train_label = torch.cat(all_train_label, dim=0)
        print("\tTraining : ")
        print_scores(all_train_label.to('cpu').numpy(), all_predict_y.to('cpu').numpy())
        
       

        # =========================================        validation model    =========================================
        all_predict_y = []
        all_validation_label = []
        for idx, (validation_x, validation_label) in enumerate(validation_loader):
            validation_x = validation_x.to(device)
            validation_label = validation_label.to(device)
            predict_y = model(validation_x.float()).detach()
            predict_y =torch.argmax(predict_y, dim=-1)
            all_predict_y.append(predict_y)
            all_validation_label.append(validation_label)

        all_predict_y = torch.cat(all_predict_y, dim=0)
        all_validation_label = torch.cat(all_validation_label, dim=0)
        print("\tValidation : ")
        print_scores(all_validation_label.to('cpu').numpy(), all_predict_y.to('cpu').numpy())
        

        # =========================================        test model    =========================================
        all_predict_y = []
        all_test_label = []
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predict_y = model(test_x.float()).detach()
            predict_y =torch.argmax(predict_y, dim=-1)
            all_predict_y.append(predict_y)
            all_test_label.append(test_label)

        all_predict_y = torch.cat(all_predict_y, dim=0)
        all_test_label = torch.cat(all_test_label, dim=0)
        print("\tTest : ")
        acc = accuracy_score(all_test_label.to('cpu').numpy(), all_predict_y.to('cpu').numpy())
        print_scores(all_test_label.to('cpu').numpy(), all_predict_y.to('cpu').numpy())


        # if not os.path.isdir("models"):
        #     os.mkdir("models")
        # torch.save(model, 'models/pepper_{:.3f}.pkl'.format(acc))

        # if np.abs(acc - prev_acc) < 1e-16:
        #     break
        # prev_acc = acc
    print("Model finished training")