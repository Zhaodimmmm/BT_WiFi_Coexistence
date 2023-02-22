# -*-coding:utf-8-*-
from builtins import print

import numpy as np
import random
import scipy.signal
import torch.nn.functional as F
from gym.spaces import Box, Discrete, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.init as init

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


# from util import Encoder, Decoder, padding_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs, rbgMap, invFlag):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, rbgMap, invFlag, act=None):
        pi = self._distribution(obs, rbgMap, invFlag)
        logp_a = None
        #print("pi",pi)
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
            # print('logp_a',logp_a)
        return pi, logp_a


#
# class MLPCategoricalActor(Actor):
#
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
#         super().__init__()
#         self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
#
#     def _distribution(self, obs):
#         logits = self.logits_net(obs)
#         return Categorical(logits=logits)
#
#     def _log_prob_from_distribution(self, pi, act):
#         return pi.log_prob(act)


class MultiCategoricalActor(Actor):

    def __init__(self, observation, act_dim, hidden_sizes, activation):
        super().__init__()
        print(observation, act_dim)
        self.max_req = observation["Requests"][0]
        self.enb_cnt = observation["RbgMap"][0]
        self.rbg_cnt = observation["RbgMap"][1]
        self.obs_dim = np.sum([np.prod(v) for k, v in observation.items()])
       
        # assert max(act_dim) == min(act_dim)  # 每个元素都相等
        self.out_dim = act_dim
        self.dropout = 0
        #self.gcn_net=GcnNet(1,2,3,3)
        self.l1 = nn.Linear(self.obs_dim, 512)
        self.l2 = nn.Linear(512, 1024)
        self.l3 = nn.Linear(1024, 1024)
        self.l4 = nn.Linear(1024, self.out_dim)

        # self.logits_net = mlp([self.obs_dim] + list(hidden_sizes) + [self.out_dim], activation).to(device)
    def _distribution(self, obs, rbgMap, invFlag):
        # assert len(obs.shape) < 3
        batch_size = 1 if len(obs.shape) == 1 else obs.shape[0]
        # batch_size = 1
        # 根据rbgMap构造mask
        rm1 = rbgMap.int().reshape(batch_size, -1).unsqueeze(2).expand(-1, -1, self.max_req)
        # print("rm1", rm1.shape)
        #input()
        rm2 = torch.zeros((*rm1.shape[:-1], 1), dtype=torch.int, device=rm1.device)
        rmask = torch.cat((rm2, rm1), 2).bool()


        temp = invFlag.int().reshape(batch_size, self.enb_cnt, -1)
        #print('tem',temp)
        #input()
        am1 = temp.unsqueeze(2).expand(-1, -1, self.rbg_cnt, -1)
        am1 = am1.reshape(batch_size, -1, self.max_req)
        am2 = torch.zeros((*am1.shape[:-1], 1), dtype=torch.int, device=am1.device)
        amask = torch.cat((am2, am1), 2).bool()
        #print('amask',amask.shape)
        #input()
        inp = torch.cat((obs, rbgMap.float(), invFlag.float()), 0 if len(obs.shape) == 1 else 1)
        inp = inp.reshape(batch_size, -1)
        # print("bbb",inp)
        x = torch.tanh(self.l1(inp))
        # print("x1",x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.l2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        # print("2", x.shape)
        x = torch.tanh(self.l3(x))
        logits = self.l4(x)
        # print("x1",logits)

        # logits = self.logits_net(inp)
        # print("aaa",logits.shape)
        # print('rmask.shape', rmask.shape)
        # logits = logits.reshape(rmask.shape)
        # logits = logits.masked_fill_(rmask | amask, -np.inf)
        # print(logits.shape)
        logits = logits.reshape(batch_size, self.max_req, -1)

        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        if len(act.shape) == 2:  # 两个维度，第一个维度为batch_size，第二个维度为每个动作的维数
            lp = pi.log_prob(act)
            #print('lp_forward.sum(lp, 1)',lp.shape)
            return torch.sum(lp, 1)  # 按照行为单位相加
        else:
            return torch.sum(pi.log_prob(act))


# class MLPGaussianActor(Actor):
#
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
#         super().__init__()
#         log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
#         self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
#         self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
#
#     def _distribution(self, obs):
#         mu = self.mu_net(obs)
#         std = torch.exp(self.log_std)
#         return Normal(mu, std)
#
#     def _log_prob_from_distribution(self, pi, act):
#         return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, observation_space, hidden_sizes, activation):
        super().__init__()
        obs_dim = np.sum([np.prod(v) for k, v in observation_space.items()])
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs.float()), -1)  # Critical to ensure v has right shape.保证是一个数


class RA_ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(256, 512, 1024, 512, 256), activation=nn.Tanh, use_cuda=True):
        super().__init__()
        # obs=np.prod(observation_space["Requests"].shape)
        action_dim = np.prod(action_space)
        self.pi = MultiCategoricalActor(observation_space, action_dim, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(observation_space, hidden_sizes, activation)
        self.use_cuda = use_cuda
        if use_cuda:
            self.pi = self.pi.to(device)
            self.v = self.v.to(device)

    def step(self, obs, rbg, flag):
        # print('obs', obs)
        # print('rbg', rbg)
        # print('flag', flag)
        # print('obs shape', obs.shape, 'rbg.shape', rbg.shape, 'flag shape', flag.shape)
        if self.use_cuda:
            obs = obs.to(device)
            rbg = rbg.to(device)
            flag = flag.to(device)

        with torch.no_grad():
            pi = self.pi._distribution(obs, rbg, flag)
            # print('pi', pi)
            a = pi.sample()
            print('a', a)
            #print('logitshape', a.shape)
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            inp = torch.cat((obs, rbg.float(), flag.float()), 0 if len(obs.shape) == 1 else 1)
            v = self.v(inp)

        if self.use_cuda:
            return a.cpu().flatten().numpy(), v.cpu().numpy(), logp_a.cpu().flatten().numpy()
        else:
            return a.flatten().numpy(), v.numpy(), logp_a.flatten().numpy()

    def act(self, obs):
        return self.step(obs)[0]
