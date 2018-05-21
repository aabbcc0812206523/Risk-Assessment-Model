'''
第二问的GPU运行版本版本 用于判别数据属于违约客户还是正常客户

测试环境：CUDA8.0 + NVIDA TITAN X PASCEL + LINUX UNBUNTU

用来预测信用额度

Predict类提供正向传播和网络结构

train函数用于训练模型

use_prettained用于加载预训练的模型

test_acc提供了正负样本测试和训练集的检验接口

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import Data

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Model params
input_size = 28  # Random noise dimension coming into generator, per output vector
hidden_size = 50  # Generator complexity
output_size = 2  # size of generated output vector
minibatch_size = 50

learning_rate = 2e-4  # 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 100000
print_interval = 10


class Predict(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Judge, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2_1 = nn.Linear(hidden_size, hidden_size)
        self.map2_2 = nn.Linear(hidden_size, hidden_size)
        self.map2_3 = nn.Linear(hidden_size, hidden_size)
        self.map2_4 = nn.Linear(hidden_size, hidden_size)
        self.map2_5 = nn.Linear(hidden_size, hidden_size)  # new
        self.map2_6 = nn.Linear(hidden_size, hidden_size)  # new
        self.map2_7 = nn.Linear(hidden_size, hidden_size)  #
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # print('initial x is ', x)
        x = F.tanh(self.map1(x))
        x = F.tanh(self.map2_1(x))
        x = F.tanh(self.map2_2(x))
        x = F.relu(self.map2_3(x))
        x = F.relu(self.map2_4(x))
        x = F.relu(self.map2_5(x))  # new
        x = F.relu(self.map2_6(x))  ##newnew
        x = F.relu(self.map2_7(x))  ##newnew
        # x = F.sigmoid(self.map3(x))  # 服务器上假的
        x = F.tanh(self.map3(x))
        # print('x is ',x)
        # print("x in :",self.map1(x))
        # print(x.size())
        return x


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


credible_data = Data.Data('data_1.npy', batch_size=minibatch_size)
incredible_data = Data.Data('data_2.npy', batch_size=minibatch_size, smote=True)
test_data = Data.Data('data_3.npy')
G = Predict(input_size=input_size, hidden_size=hidden_size, output_size=output_size).cuda()


def train():
    criterion = nn.BCEWithLogitsLoss().cuda()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=optim_betas)

    # print(credible_data.get_train_data()[:, :-1])

    high_2 = 0.085
    cre_times = 1
    incre_times = 0
    for epoch in range(num_epochs):
        # print(epoch)
        for i in range(cre_times):
            G.zero_grad()
            data = Variable(torch.Tensor(list(credible_data.get_train_data())), requires_grad=False).cuda()
            gen_input = data[:, :-2]
            # print("input",gen_input)
            # real_trans_val = F.sigmoid(Variable(torch.Tensor(list(credible_data.get_train_data()[:, -2])), requires_grad=False).cuda())
            predict = G(gen_input)[:, 0]
            # predict_trans_val = torch.pow(predict,2)
            real_val = data[:, -2]
            real_val_train = torch.pow(real_val, 0.5) / 160
            # print(predict_trans_val)
            square_error_1 = torch.mean(torch.pow(predict - real_val_train, 2))
            square_error_1.backward()
            optimizer.step()

        for i in range(incre_times):
            G.zero_grad()
            gen_input = Variable(torch.Tensor(list(credible_data.get_train_data()[:, :-2])), requires_grad=False).cuda()
            real_val = Variable(torch.Tensor(list(credible_data.get_train_data()[:, -2] / 1000.0)),
                                requires_grad=False).cuda()
            predict = G(gen_input)[:, 0]
            square_error_2 = torch.mean(torch.pow(predict - real_val, 2)) * 1000
            square_error_2.backward()
            optimizer.step()

        if epoch % print_interval == 0:
            print(epoch, 'mean_square_loss:', extract(square_error_1)[0])
            relative_error = test_acc(0)
            if relative_error < high_2:
                high_2 = relative_error
                torch.save(G.state_dict(), 'Predict %.6f.pkl' % (relative_error))


test_cre = Variable(torch.Tensor(list(credible_data.get_all_valid_data()))).cuda()
test_incre = Variable(torch.Tensor(list(incredible_data.get_all_valid_data()[:, :-2]))).cuda()


def test_acc(type):
    # G.load_state_dict(torch.load('params 1.00 1.00 - 副本.pkl'))
    correct_num = 0

    if type == 0:
        total_num = credible_data.valid_data_num
        real_val = test_cre[:, -2]
        predict = G(test_cre[:, :-2])[:, 0]
        # print(predict)
        # predict_trans_val = torch.log(1/predict-1)
        predict_trans_val = torch.pow(predict * 160, 2) + 0.000001  # 25691.16
        square_error = torch.mean(torch.abs(real_val - predict_trans_val) / (real_val + 1))
        real_val_train = torch.pow(real_val, 0.5) / 160
        square_error_1 = torch.mean(torch.pow(predict - real_val_train, 2))
        # mean_multi_1 = predict_trans_val*(predict_trans_val>1) / real_val
        # mean_multi_2 = real_val*(predict_trans_val>1) / predict_trans_val
        # mean_multi = torch.mean(torch.max(mean_multi_1,mean_multi_2))
        print('mean_multi', "real mean power error:", extract(square_error)[0], 'valid loss',
              extract(square_error_1)[0])
        f.write('mean_multi'+ "real mean power error:"+ str(extract(square_error)[0])+ 'valid loss'+
              str(extract(square_error_1)[0]))

        return extract(square_error)[0]
        # print(real_val,predict)
        # return float(correct_num_1) / total_num, (correct_num_1, total_num)

    if type == 1:
        #import pandas as pd
        #writer = pd.ExcelWriter('Save_Excel.xlsx')

        real_val = test_cre[:, -2]
        predict = G(test_cre[:, :-2])[:, 0]
        predict_trans_val = torch.pow(predict * 160, 2) # 25691.16
        data_np = test_cre.cpu().data.numpy()
        result_np = predict_trans_val.cpu().data.numpy()
        np.savetxt('result.csv', result_np, delimiter=',')
        np.savetxt('data.csv', data_np, delimiter=',')
        #data_df=pd.DataFrame(data_np)
        #data_df.to_excel(writer, 'page_1')
        #result_df=pd.DataFrame(result_np)
        #result_df.to_excel(writer, 'page_2')
        #writer.save()
        print(result_np)

def use_prettained(filename):
    model_dict = G.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = torch.load(filename)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    G.load_state_dict(model_dict)

#f = open('output_data.txt','w')
#use_prettained('GAN_para.pkl')
G.load_state_dict(torch.load('Predict 0.074947.pkl'))
test_acc(1)
# test_acc(0)
#train()
# print(G)
# test_acc(4)
#f.close()