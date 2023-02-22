import numpy as np
import pandas as pd
from user import *
from cqi_bler_model import *

import matplotlib.pyplot as plt  # 约定俗成的写法plt
from sklearn import preprocessing
import copy
from sklearn.preprocessing import normalize
# from ppo import *
# from core import  *
process = False
PL = True
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)
# enb_list = [[0, 0, 0], [-600, 0, 0], [600, 0, 0], [-300, 519.6, 0], [-300, -519.6, 0], [300, 519.6, 0],
#             [300, -519.6, 0]]
# enb_list = [[-86, 0, 0], [86, 0, 0], [0, -150, 0]]
#enb_list=[[0,0,0],[0,173.2,0],[0,-173.2,0]]#,[150,86.6,0],[-150,86.6,0],[150,-86.6,0],[-150,-86.6,0]]
# enb_list = [[0, 0, 0]]
np.set_printoptions(linewidth=200)
# table_bler = 1 - np.linspace(0, 1, 16)
# enb_cnt=len(enb_list)
PF = False

class Env:
    def __init__(self):
        self.enb_list = [[0, 0, 0]]
        self.enb_cnt = len(self.enb_list)
        self.enb_radius = 100   # 区域的半径
        self.user_perenb = 36
        self.user_number = self.user_perenb * self.enb_cnt
        self.userlist = 0
        self.request_list = 0
        self.ue_pos = np.zeros((self.user_number, 3))
        self.enb_pos = np.zeros((self.user_number, 3))
        self.distacne = np.zeros(self.user_number)
        self.label = np.zeros(self.user_number)
        self.tti = 0
        self.rbgnumber = g_nofRbg  # 在enb_data_table中设置rbgnumber的个数
        self.sbcqi = np.random.randint(0, 1, size=(self.user_number, self.rbgnumber))  # cqi 初始化15  用户数*资源块个数
        self.cqi = np.random.randint(0, 1, size=self.user_number)   # 初始化为用户数*1，数值为15
        self.RbgMap = np.zeros((self.enb_cnt, self.rbgnumber))    # 资源块的占用情况，初始化为基站数*资源块数目,0表示未占用
        self.InvFlag = np.random.randint(1, 2, size=(self.enb_cnt, self.user_number))   # 有效用户数，1表示无效，0是有效的，维度是基站数*用户数
        self.bler = np.zeros(self.user_number)   # 误块率，初始值是用户数*1，数值为0，误块率表示出错的块占传输块的比例,是反映网络服务质量的指标,对时延,吞吐量都有影响。
        self.sys_interf = np.zeros(self.user_number)
        self.rbg_capacity = get_tb(15, 1)   # 一个rbg信道上所传输的最大tb大小，单位是bit
        self.current_cqi_reqest = 0
        self.current_bler_request = 0
        self.request_position_xyz_info = 0   # 用户与接入基站的信息
        # self.cellid = np.random.randint(0, 1, size=(self.user_number))#用户数
        self.observation_space = {'Requests': (self.user_number, 22), 'RbgMap': (self.enb_cnt, self.rbgnumber),
                                  'InvFlag': (self.enb_cnt, self.user_number)}   # 资源块的使用情况，还有就是每个基站下用户是否有效
        self.action_space = (self.user_number * self.enb_cnt, self.rbgnumber)   # 动作空间就是系统下的资源个数*用户数，也就是给每个用户分配多个资源，或者就是对于每个资源来说应该分给那个用户或者部分
        self.extra_infor = {}
        self.last_tti_state = 0

    def reset(self, on, off):
        # 每次reset重置bler表，宽带cqi和子带cqi表
        seed = 3650
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.extra_infor = {}
        self.tti = 1
        self.bler = np.zeros(self.user_number)
        self.sys_interf = np.zeros(self.user_number)
        self.ue_pos = np.zeros((self.user_number,3))
        self.enb_pos = np.zeros((self.user_number,3))
        self.distacne = np.zeros(self.user_number)
        self.label = np.zeros(self.user_number)
        self.number_of_rbg_nedded = np.zeros(self.user_number)
        self.cqi = np.random.randint(0, 1, size=self.user_number)
        self.sbcqi = np.random.randint(0, 1, size=(self.user_number, self.rbgnumber))
        # 初始化所有用户位置
        self.userlist = initial_all_user(self.enb_radius, self.user_perenb, self.enb_list, ontime=on, offtime=off)
        # 初始化所有用户业务
        for i in range(len(self.userlist)):
            self.userlist[i].model2_update(tb=0, bler=[0], sinr=[0])
        # 获取初始化状态和初始状态的业务请求
        S0, self.request_list = get_user_traffic_info(self.userlist)  # 第一个参数表示状态的pandas，第二个参数是一个列表，里面表示是用户编号
        # 获取有请求用户的位置信息
        position_xyz0 = get_user_position(self.userlist)  # 返回值 列表的形式存储有请求0用户的坐标
        # 根据最近距离寻找接入基站ag
        enb_user_info = self.user_connect_enb(position_xyz0)  # 返回的是一个用户坐标，和接入基站的坐标和基站的编号所组成的列表，列表中每个元素是一个元组 UE HE ENB
        # 调用cat_data函数将用户位置和接入基站位置拼接在一起，enb_number代表用户接入基站的编号（1-7）
        cat_reqandposition_xyz, enb_number = self.cat_data(self.request_list, enb_user_info)  # 第一个是np的二维数组，行表示用户，列表示位置
        # 保存这一时刻请求用户的位置和接入基站位置的位置信息
        if len(cat_reqandposition_xyz) > 0:
            self.ue_pos[self.request_list] = cat_reqandposition_xyz[:, 1:4]
            self.enb_pos[self.request_list] = cat_reqandposition_xyz[:, 4:7]
            self.distacne[self.request_list] = cat_reqandposition_xyz[:, 7]
            self.label[self.request_list] = cat_reqandposition_xyz[:, -1]
        self.request_position_xyz_info = cat_reqandposition_xyz
        action = np.zeros((self.request_position_xyz_info.shape[0], self.rbgnumber))
        _1, _2, cqi, sbcqi, sys_interf, _, rbg_interf = calculate_cqi_bler(self.request_position_xyz_info, action)
        bytes = S0['waitingdata'].to_numpy()[self.request_list]
        for i in range(len(cqi)):
            self.number_of_rbg_nedded[self.request_list[i]] = RbgCountRequired(cqi[i], bytes[i])
        self.cqi[self.request_list] = cqi
        self.sbcqi[self.request_list] = sbcqi
        self.sys_interf[self.request_list] = sys_interf
        # 添加cqi,bler,enb_number，sbcqi信息
        S0['number_of_rbg_nedded']=self.number_of_rbg_nedded
        S0['cqi'] = self.cqi
        S0['bler'] = self.bler
        S0['enb_number'] = enb_number
        S0['distance'] = self.distacne
        S0['label']= self.label
        for i in range(self.rbgnumber):
            S0['s'+str(i)] = self.sbcqi[:, i]
        # S0['sbcqi'] = self.sbcqi.tolist()
        S0['interf'] = self.sys_interf
        S0['enb_pos']=self.enb_pos.tolist()
        S0['ue_pos'] = self.ue_pos.tolist()
        self.last_tti_state = S0
        self.InvFlag = self.generate_InvFlag(S0['enb_number'].to_numpy())
        request_data = copy.deepcopy(S0.iloc[:, 0:22].to_numpy())
        if process:
            min_max = preprocessing.MinMaxScaler()
            request_data = min_max.fit_transform(request_data)
        request_data = normalize(request_data, axis=0)
        S_PPO_0 = {'Requests': request_data.flatten(), 'RbgMap': self.RbgMap.flatten(),
                   'InvFlag': self.InvFlag.flatten()}
        return S0, S_PPO_0

    def step(self, action=0):
        # print(self.last_tti_state.iloc[:, 0:14])
        self.extra_infor = {}
        last_time_request = self.request_list  # 分配资源前的有请求的用户列表,这个列表存储了有请求用户的编号
        # last_bler = self.bler[last_time_request]#表示发起请求的用户所对应的误块率，初始为0，其实可以认为是当前时刻的误块率
        last_sbcqi = self.sbcqi[last_time_request]
        action1 = action
        if not PF:
            action = self.reshape_act_tensor(action, last_time_request)  # 处理成有请求用户*资源块的形式1代表被占用了，当前时刻的请求以及分配的动作
        # tb_list, rbg_list = get_request_user_tb(last_cqi, action)
        last_rbg_needed = self.last_tti_state[self.last_tti_state["request"] == 1]["number_of_rbg_nedded"].to_numpy(
            dtype='int').flatten()  # 计算了上一时刻发起请求的用户所要传输数据需要多少rbg,是一维数值
        #############根据当前时刻采取动作更新下一时刻的bler和cqi#########################
        next_bler, _bler, next_cqi, sbcqi, sys_interf, _sinr, rbg_interf = calculate_cqi_bler(self.request_position_xyz_info, action)  # 根据分配的动作更新和用户的信息产生要更新的一些指标
        # 根据采取动作更新用户状态
        tb_list, rbg_list = get_request_user_tb(sbcqi, action)  # 这个是根据更新后的指标，计算下一时刻需要的一些信息
        position_xyz, next_state, self.request_list = updata(self.userlist, tb_list, _bler, _sinr,
                                                             last_time_request, next_cqi, self.cqi)  # 更新状态表
        # print(self.last_tti_state)
        print('-------------------------------')
        # 3原位置
        # 计算一个tti后新的用户与基站之间的关系
        enb_user_info = self.user_connect_enb(position_xyz)
        cat_reqandposition_xyz, enb_number = self.cat_data(self.request_list, enb_user_info)
        #print(cat_reqandposition_xyz)
        if len(cat_reqandposition_xyz) > 0:
            self.ue_pos[self.request_list] = cat_reqandposition_xyz[:, 1:4]
            self.enb_pos[self.request_list] = cat_reqandposition_xyz[:, 4:7]
            self.distacne[self.request_list] = cat_reqandposition_xyz[:, 7]
            self.label[self.request_list] = cat_reqandposition_xyz[:, -1]
        self.request_position_xyz_info = cat_reqandposition_xyz
        action = np.zeros((self.request_position_xyz_info.shape[0], self.rbgnumber))
        bler,_bler, cqi, sbcqi1, sys_interf1, _, rbg_interf1 = calculate_cqi_bler(self.request_position_xyz_info, action)
        # 更新下一状态中的宽带子带cqi
        self.cqi[self.request_list] = cqi
        self.sbcqi[self.request_list] = sbcqi1
        self.sys_interf[self.request_list] = sys_interf1
        self.bler[self.request_list] = bler
        bytes = next_state['waitingdata'].to_numpy()[self.request_list]
        #self.cqi[last_time_request] = next_cqi
        #self.bler[last_time_request] = next_bler
        #self.sbcqi[last_time_request] = sbcqi
        self.sys_interf[last_time_request] = sys_interf
        #cqi=self.cqi[self.request_list]
        self.number_of_rbg_nedded = np.zeros(self.user_number)
        for i in range(len(cqi)):
            self.number_of_rbg_nedded[self.request_list[i]] = RbgCountRequired(cqi[i], bytes[i])
        # 添加cqi,bler,enb_number，sbcqi信息
        next_state['number_of_rbg_nedded'] = self.number_of_rbg_nedded
        next_state['cqi'] = self.cqi
        next_state['bler'] = self.bler
        next_state['enb_number'] = enb_number
        next_state['distance'] = self.distacne
        next_state['label'] = self.label
        for i in range(self.rbgnumber):
            next_state['s' + str(i)] = self.sbcqi[:, i]
        next_state['interf'] = self.sys_interf
        next_state['enb_pos'] = self.enb_pos.tolist()
        next_state['ue_pos'] = self.ue_pos.tolist()
        # if PL:
        #     next_state['ue_pos'].to_csv('./3.txt')
        ###########3位置变更 bler变更
        self.last_tti_state.iloc[:, 4] = next_state.iloc[:, 4]
        self.last_tti_state.iloc[:, 12] = next_state.iloc[:, 12]
        self.last_tti_state.iloc[:, 22] = next_state.iloc[:, 22]
        self.extra_infor = self.generate_extra_info(self.last_tti_state, rbg_list, last_time_request, last_rbg_needed,action1)
        ###################3
        self.last_tti_state = next_state
        self.InvFlag = self.generate_InvFlag(next_state['enb_number'].to_numpy())
        request_data = copy.deepcopy(next_state.iloc[:, 0:22].to_numpy())
        if process:
            min_max = preprocessing.MinMaxScaler()
            request_data = min_max.fit_transform(request_data)
        request_data = normalize(request_data,axis=0)
        # print(next_state['enb_number'].to_numpy())
        # print('\n',next_state)
        done = False
        S_PPO_next = {'Requests': request_data.flatten(),
                      'RbgMap': self.RbgMap.flatten(),
                      'InvFlag': self.InvFlag.flatten()}
        self.tti += 1
        return next_state, S_PPO_next, self.extra_infor, done,rbg_interf

    def generate_extra_info(self, state, rbg_list, req, rbg_needed,action):
        if len(req) == 0:
            return self.extra_infor
        enb_user_connectlist = state['enb_number'].to_numpy()
        user_rbgbumber_dict = dict(zip(req, rbg_list))
        statify = np.array(rbg_list) /rbg_needed
        statify = np.nan_to_num(statify)
        satify = np.where(statify > 1, 1, statify)
   
      
        # satify = np.mean(np.where(np.array(rbg_list) / rbg_needed > 1, 1, np.array(rbg_list) / rbg_needed)) if len(req) != 0 else 1#小区需要的和分给他的

        # satify=np.nan_to_num(satify)
     
        for i in range(self.enb_cnt):
            enb_info = state[state['enb_number'] == i + 1]
            if enb_info.empty:
                continue
            else:
                action1 = action[int(i*len(action)/3):int((i+1)*len(action)/3)]
                # print(action1)
                vector = np.where(action1 == 0)
                a = len(set(action1))
                if len(vector[0]) > 0:
                    a = a-1
                index = np.where(enb_user_connectlist == i + 1)
                rbg_number_used = 0
                enb_req_total = len(index[0])
                unassigned_total = 0
                enb_rbg_list = []
                for j in index[0]:
                    rbg_number_used += user_rbgbumber_dict[j]
                    enb_rbg_list.append(user_rbgbumber_dict[j])
                    if user_rbgbumber_dict[j] == 0:
                        unassigned_total += 1
                enb_cur_tx = enb_info['last_time_txdata'].to_numpy()
                enb_number = enb_info['enb_number'].to_numpy()
                per_rbg_tx = [enb_cur_tx[i]/enb_rbg_list[i] if enb_rbg_list[i]!=0 else 0 for i in range(len(enb_cur_tx))]
                per_rbg_tx = np.array(per_rbg_tx)#平均一个资源块上传输的bit
                per_rbg_utility = per_rbg_tx/self.rbg_capacity#平均每个资源块的与一个资源快
                if np.sum(per_rbg_utility) == 0:
                    per_rbg_utility = 0
                else:
                    mask = per_rbg_utility != 0
                    per_rbg_utility = np.mean(per_rbg_utility[mask]) if len(per_rbg_utility[mask]) != 0 else 1
                self.extra_infor['enb' + str(i + 1)] = {'enb_req_total': enb_req_total,  # 总用户请求个数
                                                        'assign_req': a,  # 分配用户数
                                                        'unassigned_total': unassigned_total,  # 没有分配到资源的个数
                                                        'number_of_rbg_nedded': enb_info[enb_info['request']==1]['number_of_rbg_nedded'].sum(),#所有用户需要的rbg个数的和
                                                        'rbg_used': rbg_number_used,#一个小区下分配了多少个资源给这些需要rbg的用户
                                                        'newdata': enb_info['newdata'].sum(),#该小区上一时刻总共到达的数据bit
                                                        'waitingdata': enb_info['waitingdata'].sum(),#该小区总公共等待的数据
                                                        'last_time_txdata': enb_info['last_time_txdata'].sum(),#当前时刻分配后小区总共传输成功的字节数
                                                        'time_duration': enb_info['time_duration'].sum(),
                                                        'total_txdata': enb_info['total_txdata'].sum(),#上一时刻该小区总公传输的字节数
                                                        'shunshi_throughput':enb_info['throughput(mbps)'].sum(),#瞬时吞吐量
                                                        'average_throughput': enb_info['average_throughput'].sum(),#平均吞吐量
                                                        'cqi': enb_info['cqi'].sum(),#上一时刻小区的cqi之和？
                                                        'rbg_usable': self.rbgnumber,#rbg最多个数
                                                        'per_rbg_utility': per_rbg_utility,#每个rbg的利用率
                                                        'bler': enb_info['bler'].sum(),###传输完之后的bler
                                                        'satify': satify,
                                                        'interf':enb_info['interf'].sum()}
        return self.extra_infor
    # 数据处理函数 将用户位置和接入基站位置拼接在一起，enb_number代表用户接入基站的编号（1-7）
    def cat_data(self, request, enb_user_info):
        data = enb_user_info
        req = request
        cat_reqandposition_xyz = []
        enb_connect_number = np.zeros(self.user_number)
        # print(position)
        # print(req)
        # input()

        for i in range(len(req)):
            # print(data[i][1][2])
            cat_reqandposition_xyz.append(
                [req[i], data[i][0][0], data[i][0][1], data[i][0][2], data[i][1][0], data[i][1][1], data[i][1][2],data[i][3],data[i][4]])
            enb_connect_number[req[i]] = data[i][2]

        cat_reqandposition_xyz = np.array(cat_reqandposition_xyz)
        return cat_reqandposition_xyz, enb_connect_number

    def generate_InvFlag(self, data):
        flag = np.random.randint(1, 2, size=(self.enb_cnt, self.user_number))
        for i in range(self.enb_cnt):
            b = np.where(data == i + 1)
            flag[i][b] = 0
        return flag

    def reshape_act_tensor(self, act, request_list):
        #print("..........",act)
        #input()
        act_matrix = np.zeros((len(request_list), self.rbgnumber), dtype='int64')
        # print('act', act)
        # print('act_matrix', act_matrix.shape)
        # print('request_list', request_list)
        assert len(act.shape) == 1, "act维度不为(x,)"

        for i in range(len(request_list)):
            ch = act[request_list[i]]
            act_matrix[i][ch] = 1

        # for i in range(len(request_list)):
        #     index = np.where(act == request_list[i] + 1)
        #     index = index[0]
        #     print('index', index)
        #     for y in range(len(index)):
        #         act_matrix[i][index[y] % self.rbgnumber] = 1
        # print('act_matrix', act_matrix)
        return act_matrix

    def calculate_system_throughput_and_fairness(self, user):
        user_list = user
        sum_throughput = 0
        sum_of_squares = 0
        for i in range(len(user_list)):
            sum_throughput += user_list[i].average_throughput
            sum_of_squares += (user_list[i].average_throughput) ** 2
        fairness = (sum_throughput ** 2) / len(user_list) * sum_of_squares

        return sum_throughput, fairness

    # 计算接入基站，返回接入基站坐标和用户坐标
    def user_connect_enb(self, position):
        position_list = position
        enb_number_user_position_list = []
        for i in range(len(position_list)):
            userposition_and_enb_position = self.caculate_distance(position_list[i])
            enb_number_user_position_list.append(userposition_and_enb_position)
        return enb_number_user_position_list

    # 最近距离选取接入基站
    def caculate_distance(self, position):
        distance_list = []
        for i in range(self.enb_cnt):
            distance = math.sqrt(
                math.pow((position[0] - self.enb_list[i][0]), 2) + math.pow((position[1] - self.enb_list[i][1]), 2) + math.pow(
                    (position[2] - self.enb_list[i][2]), 2))
            distance_list.append(distance)
        enb_connect = distance_list.index(min(distance_list))
        label = self.calc_radian(position, self.enb_list[enb_connect])
        return (position, self.enb_list[enb_connect], enb_connect + 1, distance_list[enb_connect], label)

    # 计算平均吞吐量和公平性
    def calc_radian(self, ue, enb):
        ue = np.array(ue, dtype=float)
        enb = np.array(enb, dtype=float)
        dis = ue - enb
        x_distance = dis[0]
        y_distance = dis[1]
        # 如果x中有数据为0，将其替换为eps(float的最小正值)
        eps = np.finfo(x_distance.dtype).eps
        x_distance = np.where(x_distance == 0.0, eps, x_distance)
        angle = np.divide(y_distance, x_distance)
        if x_distance == eps:
            label = math.pi / 2
        elif x_distance > 0.0:
            label = math.atan(angle)
        else:
            label = math.atan(angle) + math.pi
        label = label + (np.pi / 2)
        return label


if __name__ == '__main__':
    on = np.random.exponential(5)
    print(on)
