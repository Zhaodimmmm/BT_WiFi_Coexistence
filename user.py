import numpy as np
import pandas as pd
import math
import random
from cqi_bler_model import *
import torch
from torch.distributions.categorical import Categorical
factor = 1/99.0
class user:
    def __init__(self, ontime, enb_center, cbrrate,offtime=2, min_radius=0,max_radius=300,min_theta=0,max_theta=2*np.pi):
        '''
        :param maxdistfromorigin: 距离中心点的最大距离单位米
        :param ontime: 单位ms,指数分布的均值
        :param offtime: 单位ms,指数分布的均值
        '''
        self.min_radius = min_radius
        self.max_radius=max_radius
        self.maxdistfromorigin = 300
        self.position = np.array([0, 0, 0], dtype="float64")
        self.direction = 0
        self.speed = 3
        self.beamcenter = np.array([0, 0, 0], dtype="float64")  # 用户分布的圆形区域的中心点坐标
        self.enbcenter = enb_center
        self.speed_vector = self.random_direction_speed()
        # 业务
        self.throughput = 0
        self.request = 0  # 0表示无请求 1表示有请求
        self.ontime = 0  # 业务持续时间
        self.offtime_restore=offtime
        self.offtime = np.random.exponential(offtime)
        self.traffictype = {'text': ontime/4, 'voice': ontime/2, 'video': ontime}  # 业务类型指数分布均值参数
        self.qci_type = {'None': 0, 'text': 1, 'voice': 2, 'video': 3}
        self.qci = 0
        self.waiting_data_finally = 0  # 采取动作后剩余数据
        self.waitingbit = 0  # 当前时刻剩余待传数据
        self.cbrrate = cbrrate  # 单位bit 每毫秒 到达率
        #self.transmit_rate_one_channal = 10
        self.newarrivaldata = 0  # 新到数据
        #self.pre_txdata=0 #上一时刻传输的bit数据
        self.current_txdata = 0  # 当前时刻传输数据
        self.current_tti_waiting_data = 0#
        self.total_txdata = 0  # 总的传输数据量
        self.type = None  # 最终生成的业务类型
        self.number_of_rbg_nedded = 0  # 所需
        self.index = 0
        self.average_throughput = 0#bps
        self.nextposition = np.array([0, 0, 0], dtype="float64")
        self.time_duration = 0#时延
        self.initial_random_position(self.enbcenter,self.min_radius,self.max_radius,min_theta,max_theta)

    # 产生随机方向
    def random_direction_speed(self):
        self.direction = np.cos(np.random.uniform(0, math.pi, size=3))
        #self.speed = np.random.uniform(2, 4, size=3)
        self.speed_vector = self.speed * self.direction
        return self.speed_vector

    # 更新模式1，每一次更新都是随机方向
    def model1_update(self, tb, bler,sinr, time_duration=0.001):
        self.speed_vector = self.random_direction_speed()
        # print(speed_vector)
        self.nextposition[0] += self.speed_vector[0] * time_duration#x
        self.nextposition[1] += self.speed_vector[1] * time_duration#y
        if (np.sum(np.square(self.nextposition - self.beamcenter))) ** 0.5 >= self.maxdistfromorigin:
            self.speed_vector = -self.speed_vector
            self.position[0] += self.speed_vector[0] * time_duration
            self.position[1] += self.speed_vector[1] * time_duration
        self.traffic_updata(tb, bler,sinr)

    # 更新模式2每次更新沿着确定方向，碰到边界更改方向
    def model2_update(self, tb, bler, sinr,time_duration=0.001):
        self.nextposition[0] = self.position[0] + self.speed_vector[0] * time_duration
        self.nextposition[1] = self.position[1] + self.speed_vector[1] * time_duration
        if (np.sum(np.square(self.nextposition - self.beamcenter))) ** 0.5 <= self.maxdistfromorigin:
            self.position[0] += self.speed_vector[0] * time_duration
            self.position[1] += self.speed_vector[1] * time_duration
        else:
            while (True):
                self.speed_vector = self.random_direction_speed()
                self.nextposition[0] = self.position[0] + self.speed_vector[0] * time_duration
                self.nextposition[1] = self.position[1] + self.speed_vector[1] * time_duration
                if (np.sum(np.square(self.nextposition - self.beamcenter))) ** 0.5 <= self.maxdistfromorigin:
                    self.position[0] += self.speed_vector[0] * time_duration
                    self.position[1] += self.speed_vector[1] * time_duration
                    break
        self.traffic_updata(tb, bler,sinr)

    # 随机选择三种业务并按照指数分布随机产生业务的持续时间
    def trafficduration(self):
        #type = 'None'
        if self.offtime > 0:
            self.offtime -= 1
            if self.offtime < 0:
                self.offtime = 0
                ################
                traffic_choice = np.random.choice([1, 2, 3])
                if traffic_choice == 1:
                    self.ontime = np.random.exponential(self.traffictype['text'])
                    type = 'text'
                    self.qci = self.qci_type[type]
                elif traffic_choice == 2:
                    self.ontime = np.random.exponential(self.traffictype['voice'])
                    type = 'voice'
                    self.qci = self.qci_type[type]
                else:
                    self.ontime = np.random.exponential(self.traffictype['video'])
                    type = 'video'
                    self.qci = self.qci_type[type]
        elif self.offtime == 0 and self.ontime > 0:
            self.ontime -= 1
            if self.ontime < 0:
                self.ontime = 0
                self.offtime = np.random.exponential(self.offtime_restore)
                self.qci = 0

        return self.ontime

    # 根据采取动作更新等待数据，时延等
    def waiting_data(self, tb, bler,sinr):#tb的单位时间是bit
        self.waiting_data_finally = self.current_tti_waiting_data - tb
        if self.request == 1  and self.waiting_data_finally >= 0:
            self.current_txdata = tb
        elif self.request == 1  and self.waiting_data_finally < 0:
            self.current_txdata = self.waitingbit + self.newarrivaldata
        else:
            self.current_txdata = 0
        # if rand_number <= bler:
        #     self.waiting_data_finally = self.waitingbit + self.newarrivaldata
        # else:
        #     self.waiting_data_finally = self.waitingbit + self.newarrivaldata - tb
        #
        # if self.request == 1 and rand_number > bler and self.waiting_data_finally >= 0:
        #     self.current_txdata = tb
        # elif self.request == 1 and rand_number > bler and self.waiting_data_finally < 0:
        #     self.current_txdata = self.waitingbit + self.newarrivaldata
        # else:
        #     self.current_txdata = 0
        if tb == 0 and self.request == 1:
            self.time_duration += 1#时延加1
        if self.waiting_data_finally < 0:
            self.waiting_data_finally = 0
        self.throughput = ((self.current_txdata / 0.001)) / 1000 ** 2  # 瞬时吞吐量 单位mbps
        self.waitingbit = self.waiting_data_finally  # 更新执行完动作之后等待的bit
        self.newarrivaldata = self.cbrrate * 1 if self.ontime > 1 else self.cbrrate * self.ontime #相当于执行动作更新完成了，我要看下一时刻输入的状态中是否有新到的数据
        self.current_tti_waiting_data = self.waitingbit + self.newarrivaldata  # 更新了下一时刻的等待的字节，
        if self.current_tti_waiting_data > 0:
            self.request = 1
        else:
            self.request = 0
        ######判断传输完等待数据所需资源块数目

        #self.number_of_rbg_nedded = RbgCountRequired(cqi, self.current_tti_waiting_data)
        ###############################
        self.index += 1
        self.total_txdata += self.current_txdata
        #self.average_throughput = (self.total_txdata / (self.index / 1000)) / 1000 ** 2#Mbps
        self.average_throughput = ((1-factor)*self.average_throughput+(factor*(self.current_txdata/0.001)))#bps
    # 产生业务
    def traffic_updata(self, tb, bler,sinr):
        self.trafficduration()
        self.waiting_data(tb, bler,sinr)

    # 利用极坐标法以某一中点为圆心，radius为半径的圆内产生随机位置
    def initial_random_position(self,enb_center,min_radius,max_radius,min_theta,max_theta):
        #theta = np.random.random() * 2 *np.pi
        theta = np.random.uniform(min_theta, max_theta)
        r = np.random.uniform(min_radius, max_radius)
        self.position[0] = np.cos(theta) * r + enb_center[0]
        self.position[1] = np.sin(theta) * r + enb_center[1]


# 更新函数
def updata(user, tb, bler,sinr, last_time_request, cqi,last_cqi):
    user_list = user
    tb_list = np.zeros(len(user_list))
    #bler_list = np.zeros(len(user_list))
    bler_list=[]
    sinr_l=[]
    # cqi_list = np.random.randint(15,16,size=(len(user_list)),dtype='int')
    cqi_list=last_cqi
    for i in range(len(user_list)):
        bler_list.append([0])
        sinr_l.append(([0]))
    for i in range(len(last_time_request)):
        tb_list[last_time_request[i]] = tb[i]
        bler_list[last_time_request[i]] = bler[i]
        cqi_list[last_time_request[i]] = cqi[i]
        sinr_l[last_time_request[i]] = sinr[i]
    for i in range(len(user_list)):
        user_list[i].model2_update(tb_list[i], bler_list[i],sinr_l[i])

    user_position_xyz = get_user_position(user_list)
    traffic_info, user_request = get_user_traffic_info(user_list)
    return user_position_xyz, traffic_info, user_request


# 初始化函数
# def initial_all_user(distance, numofuser,enb_center, ontime, offtime):
#     userlist1 = [user(ontime, enb_center[0], random.randint(3000, 6120), offtime, distance) for i in range(numofuser)]
#     userlist2 = [user(ontime, enb_center[1], random.randint(3000, 6120), offtime, distance) for i in range(numofuser)]
#     userlist3 = [user(ontime, enb_center[2], random.randint(3000, 6120), offtime, distance) for i in range(numofuser)]
#     userlist =userlist1+userlist2+userlist3
#     return userlist

def initial_all_user(distance, numofuser,enb_center, ontime, offtime):
    userlist1=[]
    userlist2=[]
    userlist3=[]
    for i in range(int(numofuser/2)):
        user1=user(ontime, enb_center[0], random.randint(5120, 5120), offtime, 0.1*distance,0.3*distance,0,2*np.pi)
        userlist1.append(user1)
    for i in range(int(numofuser/2)):
        user1=user(ontime, enb_center[0], random.randint(5120,5120), offtime, 0.8 * distance, 0.9 * distance,-(1/6)*np.pi,(1/6)*np.pi)
        userlist1.append(user1)
    for i in range(int(numofuser/2)):
        user1=user(ontime, enb_center[1], random.randint(5120, 5120), offtime, 0.1*distance,0.3*distance,0,2*np.pi)
        userlist2.append(user1)
    for i in range(int(numofuser/2)):
        user1=user(ontime, enb_center[1], random.randint(5120, 5120), offtime, 0.8 * distance, 0.9 * distance,(5/6)*np.pi,(7/6)*np.pi)
        userlist2.append(user1)
    for i in range(int(numofuser/2)):
        user1=user(ontime, enb_center[2], random.randint(5120, 5120), offtime, 0.1*distance,0.3*distance,0,2*np.pi)
        userlist3.append(user1)
    for i in range(int(numofuser/2)):
        user1=user(ontime, enb_center[2], random.randint(5120, 5120), offtime, 0.8 * distance, 0.9 * distance,(2/6)*np.pi,(4/6)*np.pi)
        userlist3.append(user1)
    userlist =userlist1+userlist2+userlist3
    return userlist


# 获取发起业务请求用户的位置和编号
def get_user_position(user):
    userlist = user
    user_position_XYZ = []
    for i in range(len(userlist)):
        if userlist[i].request == 1:
            position = userlist[i].position
            user_position_XYZ.append(position)  # 只保留位置信息

    return user_position_XYZ


# 获取用户的信息，包括等待数据，新到数据，时延，吞吐量等
def get_user_traffic_info(user):
    userlist = user
    user_request = []
    traffic_info = []
    for i in range(len(userlist)):
        if userlist[i].request == 1:
            user_request.append(i)
        traffic_info.append(
            (i, userlist[i].newarrivaldata, userlist[i].current_tti_waiting_data, userlist[i].request,
             userlist[i].current_txdata,
             userlist[i].time_duration, userlist[i].qci,userlist[i].total_txdata, userlist[i].throughput,
             userlist[i].average_throughput))
    traffic_info = np.array(traffic_info, dtype='float')
    traffic_info = pd.DataFrame(traffic_info,
                                columns=['user', 'newdata', 'waitingdata', 'request', 'last_time_txdata',
                                         'time_duration',
                                         'qci',
                                         'total_txdata', 'throughput(mbps)',
                                         'average_throughput'])

    return traffic_info, user_request


def get_all_user_position_and_request(user):
    userlist = user
    position_and_req = []
    for i in range(len(userlist)):
        position = userlist[i].position.tolist()
        position_and_req.append((i, position[0], position[1], position[2], userlist[i].request))
    position_and_req = np.array(position_and_req, dtype='float')

    return position_and_req


if __name__ == "__main__":
    a=np.array([1,2,3])
    b=np.array([4,5,1])

    c=[1,2,3]
    print(len(c))
