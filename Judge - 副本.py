import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import Data



# Model params
input_size = 28  # Random noise dimension coming into generator, per output vector
hidden_size = 50  # Generator complexity
output_size = 1  # size of generated output vector
minibatch_size = 50

learning_rate = 2e-4  # 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 10000
print_interval = 20


class Judge(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Judge, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2_1 = nn.Linear(hidden_size, hidden_size)
        self.map2_2 = nn.Linear(hidden_size, hidden_size)
        self.map2_3 = nn.Linear(hidden_size, hidden_size)
        self.map2_4 = nn.Linear(hidden_size, hidden_size)
        self.map2_5 = nn.Linear(hidden_size, hidden_size)  # new
        self.map2_6 = nn.Linear(hidden_size, hidden_size)  # new
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
        x = F.sigmoid(self.map3(x))
        # print('x is ',x)
        # print("x in :",self.map1(x))
        # print(x.size())
        return x


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


credible_data = Data.Data('E:\王雨飞文件\数学建模\校赛\\数模校内赛题目\问题B：不完全对称信息下的客户授信评估问题\data_1.npy', batch_size=minibatch_size)
incredible_data = Data.Data('E:\王雨飞文件\数学建模\校赛\\数模校内赛题目\问题B：不完全对称信息下的客户授信评估问题\data_2.npy', batch_size=minibatch_size, smote=True)
test_data = Data.Data('E:\王雨飞文件\数学建模\校赛\\数模校内赛题目\问题B：不完全对称信息下的客户授信评估问题\data_3.npy')
G = Judge(input_size=input_size, hidden_size=hidden_size, output_size=output_size)


def train():
    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=optim_betas)

    #print(credible_data.get_train_data()[:, :-1])

    high_1 = 0
    high_2 = 0
    for epoch in range(num_epochs):
        G.zero_grad()
        gen_input = Variable(torch.Tensor(list(credible_data.get_train_data()[:, :-2])), requires_grad=False)
        # print("input",gen_input)
        judge_result = G(gen_input)
        # print(judge_result)
        judge_cre_error = criterion(judge_result, Variable(torch.zeros(minibatch_size, 1)))
        judge_cre_error.backward()

        # repaet
        gen_input = Variable(torch.Tensor(list(credible_data.get_train_data()[:, :-2])), requires_grad=False)
        # print("input",gen_input)
        judge_result = G(gen_input)
        # print(judge_result)
        judge_cre_error = criterion(judge_result, Variable(torch.zeros(minibatch_size, 1)))
        judge_cre_error.backward()

        gen_input = Variable(torch.Tensor(list(incredible_data.get_train_data()[:, :-2])), requires_grad=False)
        judge_result = G(gen_input)
        judge_incre_error = criterion(judge_result, Variable(torch.ones(minibatch_size, 1)))
        judge_incre_error.backward()


        #print(incredible_data.get_train_smote_data())
        gen_input = Variable(torch.Tensor(list(incredible_data.get_train_smote_data()[:, :-2])), requires_grad=False)
        judge_result = G(gen_input)
        judge_incre_error = criterion(judge_result, Variable(torch.ones(minibatch_size, 1)))
        judge_incre_error.backward()


        optimizer.step()  # Only optimizes G's parameters

        if epoch % print_interval == 0:
            cre_acc, cre_data = test_acc(0)
            incre_acc, incre_data = test_acc(1)
            # print('credibl_accurancy:', cre_acc)
            # print('incredible_accurancy:', incre_acc)
            TP_rate = cre_data[0] / cre_data[1]
            FP_rate = 1 - incre_data[0] / incre_data[1]
            # print(FP_rate,TP_rate)
            auc = FP_rate * TP_rate * 0.5 + (TP_rate + 1) * (1 - FP_rate) * 0.5
            print(epoch, 'auc:', auc, 'credible_loss:', extract(judge_cre_error)[0], 'incredible_loss',
                  extract(judge_incre_error)[0])
            if auc > high_2:
                high_2 = auc
                torch.save(G.state_dict(), 'params_new_new_noamount %.6f.pkl' % (auc))


def test_acc(type):
    # G.load_state_dict(torch.load('params 1.00 1.00 - 副本.pkl'))
    correct_num = 0
    if type == 0:
        total_num = credible_data.valid_data_num
        for i in credible_data.get_valid_data_itr(total_num):
            data = Variable(torch.Tensor(list(i[:-2])), requires_grad=False)
            val_result = G(data)
            if float(val_result) < 0.5:
                correct_num += 1

        print('credible_callback:', float(correct_num) / total_num, 'correct_num', correct_num, 'total', total_num)
        return float(correct_num) / total_num, (correct_num, total_num)

    elif type == 1:
        total_num = incredible_data.valid_data_num
        for i in incredible_data.get_valid_data_itr(total_num):
            data = Variable(torch.Tensor(list(i[:-2])), requires_grad=False)
            val_result = G(data)
            if float(val_result) > 0.5:
                correct_num += 1

        print('incredible_callback:', float(correct_num) / total_num, 'correct_num', correct_num, 'total', total_num)
        return float(correct_num) / total_num, (correct_num, total_num)

    elif type == 2:
        total_num = incredible_data.train_data_num
        for i in incredible_data.get_train_data_itr(total_num):
            data = Variable(torch.Tensor(list(i[:-2])), requires_grad=False)
            val_result = G(data)
            if float(val_result) > 0.5:
                correct_num += 1

        print('acc:', float(correct_num) / total_num, 'total', total_num)

    elif type == 3:
        total_num = credible_data.train_data_num
        for i in credible_data.get_train_data_itr(total_num):
            data = Variable(torch.Tensor(list(i[:-2])), requires_grad=False)
            val_result = G(data)
            if float(val_result) < 0.5:
                correct_num += 1

        print('acc:', float(correct_num) / total_num, 'total', total_num)



    elif type == 4:
        data = Variable(torch.Tensor(list(test_data.data[:,:-2])), requires_grad=False)
        val_result = G(data)
        print(val_result)
        #print('acc:', float(correct_num) / total_num, 'total', total_num)


def use_prettained(filename):
    model_dict = G.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = torch.load(filename)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    G.load_state_dict(model_dict)


use_prettained('GAN_para.pkl')
#G.load_state_dict(torch.load('params_new_new_noamount 0.942047.pkl'))
# test_acc(1)
# test_acc(0)
train()
#print(G)
#test_acc(4)