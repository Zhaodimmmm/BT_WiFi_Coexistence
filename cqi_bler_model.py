import numpy as np

import time
import random
import math as m
from table import *


class U2Schannels:
    # Simulator of the U2S channels
    def __init__(self, n_RBG, rbgsize, ENB_pow):
        # self.h_bs = 25
        # self.h_ms = 1.5
        self.Decorrelation_distance = 50
        self.rbgSize = rbgsize
        self.shadow_std = 8
        self.n_RB = n_RBG
        self.U_Nosie = 0
        self.U_Interference = 0
        self.noiseFigure = 2
        self.SINR = 0
        self.c = 3 * (10 ** 8)
        self.Sat_power_dB = ENB_pow  # 所有卫星的发射功率一样
        # self.update_shadow([])

    def get_frequency(self, freq):
        self.frequency = freq

    def get_gain(self, sat_txgain, ue_rxgain):
        self.g_t = sat_txgain  # db
        self.g_r = ue_rxgain

    def GainDb(self, phi, theta):
        Gsmax = 50.82
        Ls1 = -20
        ThetaC = 21.56
        Theta1 = 2.58
        Theta2 = 6.32
        Theta3Db = 0.2
        if ThetaC < phi:
            return 0
        if (phi > 0) & (phi < (Theta1 * Theta3Db)):
            return Gsmax - 3 * pow(phi / Theta3Db, 2)
        if (phi > (Theta1 * Theta3Db)) & (phi < (Theta2 * Theta3Db)):
            return Gsmax + Ls1
        if (phi > (Theta2 * Theta3Db)) & (phi < ThetaC):
            return Gsmax + Ls1 + 20 - 25 * m.log10(phi / Theta3Db)

    # def CosineGetGainDb(self, phi, theta):
    #     Beamwidth = 13  # The 3dB beamwidth (degrees)
    #     Orientation = 0.0  # The angle (degrees) that expresses the orientation of the antenna on the x-y plane relative to the x axis
    #     MaxGain = 26  # The gain (dB) at the antenna boresight (the direction of maximum gain)
    #     m_beamwidthRadians = Beamwidth * m.pi / 180.0
    #     m_exponent = -3.0 / (20 * m.log10(m.cos(m_beamwidthRadians / 4.0)))
    #     m_orientationRadians = Orientation * m.pi / 180.0
    #     phi = phi - m_orientationRadians
    #     while (phi <= -m.pi):
    #         phi += m.pi + m.pi
    #     while (phi > m.pi):
    #         phi -= m.pi + m.pi
    #     ef = pow(m.cos(phi / 2.0), m_exponent)
    #     # print(ef)
    #     gainDb = 20 * m.log10(ef)
    #     return gainDb + MaxGain
    #
    # def AntennaModel(self, distance):  # 需要
    #     n = distance.shape[0]  # listangle 包含ue,beamcenter,sate三点的坐标假设都是（x,y,z）
    #     uebeamGain = []
    #     for i in range(n):
    #         sat_ue_dis = m.sqrt((distance[i, 7] - distance[i, 1]) ** 2 + (distance[i, 8] - distance[i, 2]) ** 2 + (
    #                 distance[i, 9] - distance[i, 3]) ** 2)
    #         beam_ue_dis = m.sqrt(
    #             (distance[i, 4] - distance[i, 1]) ** 2 + (distance[i, 5] - distance[i, 2]) ** 2 + (
    #                     distance[i, 6] - distance[i, 3]) ** 2)
    #
    #         sat_beam_dis = m.sqrt(
    #             (distance[i, 7] - distance[i, 4]) ** 2 + (distance[i, 8] - distance[i, 5]) ** 2 + (
    #                     distance[i, 9] - distance[i, 6]) ** 2)
    #
    #         phi = m.acos((sat_ue_dis * sat_ue_dis + sat_beam_dis * sat_beam_dis - beam_ue_dis * beam_ue_dis)
    #                      / (2.0 * sat_ue_dis * sat_beam_dis))
    #         theta = 0
    #         uebeamGain.append(self.GainDb(phi, theta))
    #     return np.array(uebeamGain)

    def update_pathloss(self, distance):  # distance 是一个有请求用户数×7的数组 (imsi,ue_pos,enb_pos)
        row = distance.shape[0]
        self.n_UE = row
        # print('n_UE', self.n_UE)
        self.PathLoss = np.zeros([row, row])
        for i in range(row):#用户
            for j in range(row):#干扰用户接入的基站
                d1 = abs(distance[i, 1] - distance[j, 4])
                d2 = abs(distance[i][2] - distance[j][5])
                d3 = abs(distance[i][3] - distance[j][6])
                temp_dis = m.hypot(d1, d2)
                distance1 = m.hypot(temp_dis, d3)
                numerator = (4 * m.pi * distance1 * self.frequency) ** 2
                denominator = m.pow(self.c, 2)
                self.PathLoss[i][j] = 10 * np.log10(numerator / denominator) - self.g_r + self.g_t  # 接入基站下的所有用户的路径损耗
        return self.PathLoss  # dB，行表示的是第一个有请求的用户会受到的干扰小区带来的路径损耗

    # def update_shadow(self, delta_distance_list):
    #     if len(delta_distance_list) == 0:  # initialization
    #         self.Shadow = np.random.normal(0, self.shadow_std, self.n_UE)
    #     else:
    #         delta_distance = np.asarray(delta_distance_list)
    #         self.Shadow = np.exp(-1 * (delta_distance / self.Decorrelation_distance)) * self.Shadow + \
    #                       np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0,
    #                                                                                                                   self.shadow_std,
    #                                                                                                                   self.n_UE)
    # def update_fast_fading(self):
    #     h = 1 / np.sqrt(2) * (np.random.normal(size=(self.n_UE, self.n_RB)) + 1j * np.random.normal(
    #         size=(self.n_UE, self.n_RB)))
    #     self.FastFading = 20 * np.log10(np.abs(h))
    def Compute_Noise(self):
        KT_dbm_Hz = -174.0
        KT_W_Hz = np.power(10, (KT_dbm_Hz - 30) / 10)
        nF = np.power(10, self.noiseFigure / 10)
        self.U_Nosie = np.zeros([self.n_UE, self.n_RB]) + KT_W_Hz * nF  # W

    def Compute_Interference(self, actions):
        # action 用户×资源块 是numpy
        self.S2U_channels_abs = self.PathLoss  # + self.Shadow
        S2U_channels_with_fastfading = np.repeat(self.S2U_channels_abs[np.newaxis, :, :], self.n_RB,
                                                 axis=0)  # 维度为用户数×资源块
        self.S2U_channels_with_fastfading = S2U_channels_with_fastfading  # - self.FastFading
        R_power_dB = self.Sat_power_dB - self.S2U_channels_with_fastfading
        R_power_W = 10 ** (R_power_dB / 10)  # 所有接收功率包含干扰的
        self.U_Interference = np.zeros([self.n_UE, self.n_RB])
        for i in range(self.n_RB):
            # print('actions', actions)
            # print('self.n_RB', self.n_RB)
            R_power_W1 = R_power_W[i]
            # print('R_power_W1', R_power_W1)
            self.Signal = np.diag(R_power_W1)  # 每个有请求用户的有用功率
            # print('self.Signal', self.Signal)
            self.Signal = np.repeat(self.Signal[:, np.newaxis], self.n_RB, axis=1)
            # print('self.Signal', self.Signal.shape)
            for k in range(self.n_UE):
                # print('actions[k, i]', actions[k, i])
                if actions[k, i] == 0:
                    continue
                P_interference_temp = actions[:, i] * R_power_W1[k]   # 用户数*1，求得干扰
                # print('P_interference_temp', P_interference_temp)
                P_interference = np.delete(P_interference_temp, k)    # 去除第k个用户自己本身的功率，然后
                # print('P_interference', P_interference)
                # print('P_interference', P_interference.size)
                # print('k', k)
                # print('i', i)
                # input()
                for m in range(P_interference.size):
                    self.U_Interference[k, i] += P_interference[m]  # 得到用户k在第i个资源块上受到的干扰功率W
                    # print('self.U_Interference[k, i]', self.U_Interference[k, i])
        sys_interference = np.sum(self.U_Interference, axis=1)
        rbg_interference = np.sum(self.U_Interference, axis=0)
        #print("ssss", self.U_Interference)
        #print("ssss", sys_interference)
        self.U_Interference += self.U_Nosie  # 功率w
        # print(self.Signal)
        # print(self.U_Interference)
        self.SINR = self.Signal / self.U_Interference
        return self.SINR, sys_interference, rbg_interference   # 用户数*rb

    def Compute_MiBler(self, mib, ecrId, cbSize):
        b = 0
        c = 0
        cbIndex = 1
        while ((cbIndex < 9) and (cbMiSizeTable[cbIndex] <= cbSize)):
            cbIndex += 1
        cbIndex -= 1
        b = bEcrTable[cbIndex][ecrId]
        if (b < 0):
            i = cbIndex
            while ((i < 9) and (b < 0)):
                b = bEcrTable[i][ecrId]
                i += 1
        c = cEcrTable[cbIndex][ecrId]
        if (c < 0):
            i = cbIndex
            while ((i < 9) and (c < 0)):
                c = cEcrTable[i][ecrId]
                i += 1
        mbler = float(0.5 * (1 - m.erf((mib - b) / (m.sqrt(2) * c))))
        return mbler

    def Compute_MIB(self, sinr, map, mcs):
        MIsum = 0
        sinrcopy = sinr
        for i in map:
            sinrLin = sinrcopy[i]
            if mcs <= MI_QPSK_MAX_ID:
                if sinrLin > MI_map_qpsk_axis[MI_MAP_QPSK_SIZE - 1]:
                    MI = 1
                else:
                    scalingCoeffQpsk = (MI_MAP_QPSK_SIZE - 1) / (
                            MI_map_qpsk_axis[MI_MAP_QPSK_SIZE - 1] - MI_map_qpsk_axis[0])
                    sinrIndexDouble = (sinrLin - MI_map_qpsk_axis[0]) * scalingCoeffQpsk + 1
                    sinrIndex = max(0, m.floor(sinrIndexDouble))
                    MI = MI_map_qpsk[sinrIndex]
            else:
                if mcs > MI_QPSK_MAX_ID and mcs <= MI_16QAM_MAX_ID:
                    if sinrLin > MI_map_16qam_axis[MI_MAP_16QAM_SIZE - 1]:
                        MI = 1
                    else:
                        scalingCoeff16qam = (MI_MAP_16QAM_SIZE - 1) / (
                                MI_map_16qam_axis[MI_MAP_16QAM_SIZE - 1] - MI_map_16qam_axis[0])
                        sinrIndexDouble = (sinrLin - MI_map_16qam_axis[0]) * scalingCoeff16qam + 1
                        sinrIndex = max(0, m.floor(sinrIndexDouble))
                        MI = MI_map_16qam[sinrIndex]
                else:
                    if sinrLin > MI_map_64qam_axis[MI_MAP_64QAM_SIZE - 1]:
                        MI = 1
                    else:
                        scalingCoeff64qam = (MI_MAP_64QAM_SIZE - 1) / (
                                MI_map_64qam_axis[MI_MAP_64QAM_SIZE - 1] - MI_map_64qam_axis[0])
                        sinrIndexDouble = (sinrLin - MI_map_64qam_axis[0]) * scalingCoeff64qam + 1
                        sinrIndex = max(0, m.floor(sinrIndexDouble))
                        MI = MI_map_64qam[sinrIndex]
            MIsum += MI
        MI = MIsum / len(map)
        return MI

    def Compute_Bler(self, sinr, map, size, mcs):
        tbMi = self.Compute_MIB(sinr, map, mcs)
        MI = tbMi
        Reff = 0
        Z = 6144
        B = size * 8
        C = 0
        Cplus = 0
        Kplus = 0
        Cmin = 0
        Kmin = 0
        B1 = 0
        deltaK = 0
        if B <= Z:
            C = 1
            B1 = B
        else:
            L = 24
            C = m.ceil(B / (Z - L))
            B1 = B + C * L
        min = 0
        max = 187
        mid = 0
        while True:
            mid = int((min + max) / 2)
            if (B1 > cbSizeTable[mid] * C):
                if B1 < cbSizeTable[mid + 1] * C:
                    break
                else:
                    min = mid + 1
            else:
                if B1 > cbSizeTable[mid - 1] * C:
                    break
                else:
                    max = mid - 1
            if (cbSizeTable[mid] * C == B1) or (min >= max):
                break
        if B1 > cbSizeTable[mid] * C:
            mid += 1
        KplusId = mid
        Kplus = cbSizeTable[mid]
        if C == 1:
            Cplus = 1
            Cmin = 0
            Kmin = 0
        else:
            Kmin = cbSizeTable[KplusId - 1 if KplusId > 1 else 0]
            deltaK = Kplus - Kmin
            Cmin = m.floor(((C * Kplus) - B1) / deltaK)
            Cplus = C - Cmin
        errorRate = 1.0
        ecrId = 0
        ecrId = McsEcrBlerTableMapping[mcs]  # 不考虑harq
        if C != 1:
            cbler = self.Compute_MiBler(MI, ecrId, Kplus)
            errorRate *= pow(1 - cbler, Cplus)
            cbler = self.Compute_MiBler(MI, ecrId, Kmin)
            errorRate *= pow(1 - cbler, Cmin)
            errorRate = 1 - errorRate
        else:
            errorRate = self.Compute_MiBler(MI, ecrId, Kplus)
        return errorRate

    def compute_CQI(self):
        temp_SINR = self.SINR
        temp_SINR = np.repeat(temp_SINR, self.rbgSize, axis=1)
        row = temp_SINR.shape[0]
        column = temp_SINR.shape[1]
        sub_channel = int(column / self.rbgSize)
        CQI = np.zeros((row, sub_channel), dtype=int)
        self.bler = np.zeros((row, sub_channel))
        for i in range(row):
            rbId = 0
            rbgMap = []
            for j in range(column):
                rbgMap.append(rbId)
                rbId += 1
                k = int(j / self.rbgSize)
                if rbId % self.rbgSize == 0 or j == column - 1:
                    mcs = 0
                    tem = []
                    while mcs <= 28:
                        # print(temp_SINR[i])
                        # input()
                        BLER = self.Compute_Bler(temp_SINR[i], rbgMap, int(GetDlTbSizeFromMcs(mcs, self.rbgSize) / 8),
                                                 mcs)
                        tem.append(BLER)
                        if BLER > 0.1:
                            break
                        mcs += 1
                    # print(int(j/self.rbgSize))
                    #if BLER>0.1:
                        #self.bler[i][k]=0.1
                    #else:
                    self.bler[i][k] = BLER
                    #print(tem[len(tem) - 2],tem[-1],len(tem),BLER)
                    #input()
                    if mcs > 0:
                        mcs -= 1
                    if BLER > 0.1 and mcs == 0:
                        CQI[i][k] = 0
                    elif mcs == 28:
                        CQI[i][k] = 15
                    else:
                        s = SpectralEfficiencyForMcs[mcs]
                        CQI[i][k]=0
                        while CQI[i][k] < 15 and SpectralEfficiencyForCqi[CQI[i, k] + 1] < s:
                            CQI[i, k] += 1
        # wb_cqi = np.zeros(row, dtype=int)
        # j = 0
        # for i in CQI:
        #     for each in i:
        #         wb_cqi[j] += each
        #     wb_cqi[j] /= len(i)
        #     j += 1
        #print("belr",self.bler)
        #print("sinr",temp_SINR)
        wb_cqi = np.sum(CQI, axis=1) / sub_channel
        wb_cqi = np.array(wb_cqi, dtype='int')
        # print(CQI.shape)
        # input()
        return CQI, wb_cqi  # 返回两个参数，子带cqi和宽带cqi

    def compute_trueChannel(self, action):#求得每个用户的误块率,取误块率最大的那个认为是该用户的最差信道
        temp = self.bler
        tem=self.SINR
        temp = temp * action#
        tem = tem *action
        bler=[]
        sinr=[]
        for i in range(action.shape[0]):
            a=action[i]
            index=np.where(a==1)
            b=temp[i][index[0]]
            c=tem[i][index[0]]
            bler.append(b)
            sinr.append(c)
        res = np.zeros(self.n_UE)
        j = 0
        for i in temp:
            max = 0
            for each in i:
                if each > max:
                    max = each
            res[j] = max
            j += 1
        return res, bler, sinr


def calculate_cqi_bler(distance, act):

    if len(act) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0),np.zeros((1, g_nofRbg)),np.zeros(0),np.zeros(0),np.zeros((1, g_nofRbg))
    env = U2Schannels(g_nofRbg, g_nRbgSize, 30)
    env.get_frequency(2.4e9)
    env.get_gain(0, 0)
    path = env.update_pathloss(distance)
    env.Compute_Noise()
    sinr, interf, rbg_interf= env.Compute_Interference(act)
    #print("inter",interf)
    sbcqi, w_cqi = env.compute_CQI() # wb_cqi
    #sbcqi = env.compute_CQI()[0]  # sb_cqi
    bler ,_bler,_sinr = env.compute_trueChannel(act)  # bler
    return bler, _bler, w_cqi, sbcqi, interf, _sinr, rbg_interf


def get_request_user_tb(sbcqi, act):#计算tbsize
    rbgnumber_list = np.sum(act, axis=1) if len(act) != 0 else np.sum(act, axis=0)#有请求用户的矩阵
    sbcqi=sbcqi*act
    sbcqi_list=[]
    for i in range(act.shape[0]):
        a = act[i]
        index = np.where(a == 1)
        b = sbcqi[i][index[0]].tolist()
        sbcqi_list.append(b)
    tb_list = np.zeros(len(sbcqi))
    for i in range(len(sbcqi)):
        tb_list[i] = get_tb(sbcqi_list[i], 1)
    return tb_list, rbgnumber_list   # 返回参数 第一个是tbsize，第二个是分配资源个数
