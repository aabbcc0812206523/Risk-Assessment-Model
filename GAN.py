import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Data import Data
from torch.autograd import Variable

# Data params
data_mean = 4
data_stddev = 1.25

# Model params
g_input_size = 3  # Random noise dimension coming into generator, per output vector
g_hidden_size = 50  # Generator complexity
g_output_size = 28  # size of generated output vector
d_input_size = 28  # Minibatch size - cardinality of distributions
d_hidden_size = 50  # Discriminator complexity
d_output_size = 1  # Single dimension for 'real' vs. 'fake'
minibatch_size = 30

d_learning_rate = 2e-4  # 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 10000
print_interval = 200
d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 2


# ##### DATA: Target data and generator input data

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian


def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian


# ##### MODELS: Generator model and discriminator model

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2_1 = nn.Linear(hidden_size, hidden_size)
        self.map2_2 = nn.Linear(hidden_size, hidden_size)
        self.map2_3 = nn.Linear(hidden_size, hidden_size)
        self.map2_4 = nn.Linear(hidden_size, hidden_size)
        self.map2_5 = nn.Linear(hidden_size, hidden_size)  # new
        self.map2_6 = nn.Linear(hidden_size, hidden_size)  # new
        self.map2_7 = nn.Linear(hidden_size, hidden_size)  # new
        self.map2_8 = nn.Linear(hidden_size, hidden_size)  # new
        self.map3_own = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.tanh(self.map1(x))
        x = F.relu(self.map2_1(x))
        x = F.relu(self.map2_2(x))
        x = F.relu(self.map2_3(x))
        x = F.relu(self.map2_4(x))
        x = F.tanh(self.map2_5(x))
        x = F.tanh(self.map2_6(x))
        x = F.tanh(self.map2_7(x))  # new
        x = F.tanh(self.map2_8(x))  ##newnew
        x = self.map3_own(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2_1 = nn.Linear(hidden_size, hidden_size)
        self.map2_2 = nn.Linear(hidden_size, hidden_size)
        self.map2_3 = nn.Linear(hidden_size, hidden_size)
        self.map2_4 = nn.Linear(hidden_size, hidden_size)
        self.map2_5 = nn.Linear(hidden_size, hidden_size)  # new
        self.map2_6_own = nn.Linear(hidden_size, hidden_size)  # new
        self.map3_own = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.tanh(self.map1(x))
        x = F.tanh(self.map2_1(x))
        x = F.tanh(self.map2_2(x))
        x = F.relu(self.map2_3(x))
        x = F.relu(self.map2_4(x))
        x = F.relu(self.map2_5(x))  # new
        x = F.relu(self.map2_6_own(x))  ##newnew
        x = F.sigmoid(self.map3_own(x))
        return x


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)


credible_data = Data('E:\王雨飞文件\数学建模\校赛\\数模校内赛题目\问题B：不完全对称信息下的客户授信评估问题\data_1.npy', batch_size=minibatch_size)
incredible_data = Data('E:\王雨飞文件\数学建模\校赛\\数模校内赛题目\问题B：不完全对称信息下的客户授信评估问题\data_2.npy', batch_size=minibatch_size)
d_sampler = get_distribution_sampler(data_mean, data_stddev)
gi_sampler = get_generator_input_sampler()
G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)


def train():
    max = 999
    for epoch in range(num_epochs):
        for d_index in range(d_steps):
            # 1. Train D on real+fake
            D.zero_grad()

            #  1A: Train D on real
            d_real_data = Variable(torch.Tensor(list(credible_data.get_train_data()[:, :-2])), requires_grad=False)
            # print(d_real_data)
            d_real_decision = D(d_real_data)
            d_real_error = criterion(d_real_decision, Variable(torch.ones(minibatch_size, 1)))  # ones = true
            d_real_error.backward()  # compute/store gradients, but don't change params

            d_real_data = Variable(torch.Tensor(list(incredible_data.get_train_data()[:, :-2])), requires_grad=False)
            # print(d_real_data)
            d_real_decision = D(d_real_data)
            d_real_error = criterion(d_real_decision, Variable(torch.ones(minibatch_size, 1)))  # ones = true
            d_real_error.backward()  # compute/store gradients, but don't change params


            #  1B: Train D on fake
            d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
            d_fake_decision = D(d_fake_data)

            correct_num = sum(d_fake_decision<0.5)


            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(minibatch_size, 1)))  # zeros = fake
            d_fake_error.backward()
            d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            #print(gen_input)
            g_fake_data = G(gen_input)
            dg_fake_decision = D(g_fake_data)
            #print(dg_fake_decision)
            #print(dg_fake_decision)
            g_error = criterion(dg_fake_decision,
                                Variable(torch.ones(minibatch_size, 1)))  # we want to fool, so pretend it's all genuine

            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters

        if epoch % print_interval == 0:
            print('batch:', epoch,
                  'd_real_error:', extract(d_real_error)[0], 'd_judge_error:', extract(d_fake_error)[0])
            print('g_error',
                  extract(g_error)[0],'sample')#,g_fake_data)

            #effect = pow(extract(d_real_error)[0]-0.5,2)+pow(extract(d_fake_error)[0]-0.5,2)
            #if effect < max:
                #torch.save(D.state_dict(), 'gan_para.pkl')
                #max=effect
                #print('******')

def test_acc(type,data=[]):
    # G.load_state_dict(torch.load('params 1.00 test=credible_data.valid_data[0][:-2]1.00 - 副本.pkl'))
    correct_num = 0
    if type == 0:
        total_num = credible_data.valid_data_num
        for i in credible_data.get_valid_data_itr(total_num):
            data = Variable(torch.Tensor(list(i[:-2])), requires_grad=False)
            val_result = D(data)
            if float(val_result) > 0.5:
                correct_num += 1

        print('credible_callback:', float(correct_num) / total_num, 'correct_num', correct_num, 'total', total_num)
        return float(correct_num) / total_num, (correct_num, total_num)

    if type==1:
        test = data[:-2]



train()
#torch.save(D.state_dict(), 'gan_para.pkl')
#D.load_state_dict(torch.load('gan_para.pkl'))
#test_acc(0)
