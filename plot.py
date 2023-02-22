'''
Author: lbh 985878624@qq.com
Date: 2021-12-21 13:06:40
LastEditors: lbh 985878624@qq.com
LastEditTime: 2023-02-12 22:50:35
FilePath: /LLLL/fdd/plot.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.ndimage.filters import gaussian_filter1d

plt.style.use('ggplot')
file = './result_new/2022-12-31_dqn0.1r3/0.1dqn.txt'
df_news = pd.read_table(file, header=None)
# df_news=df_news.drop(0)
# print(df_news)
path = './jpg'
try:
    os.makedirs(path)
except:
    pass
x = df_news[0]
rbg_used = df_news[13]
rbg_used =pd.to_numeric(rbg_used,errors='coerce')
rbg_used.to_numpy()

# bler = df_news[14]
# bler =pd.to_numeric(bler,errors='coerce')
# bler.to_numpy()

interf=df_news[15]
interf =pd.to_numeric(interf,errors='coerce')
interf.to_numpy()

# interf1=df_news[24]
# # interf1 =pd.to_numeric(interf1,errors='coerce')
# # interf1.to_numpy()

# r1 = df_news[25]
# r1 =pd.to_numeric(r1,errors='coerce')
# r1.to_numpy()

# r2 = df_news[26]
# r2 =pd.to_numeric(r1,errors='coerce')
# r2.to_numpy()

# r3 = df_news[27]
# r3 =pd.to_numeric(r1,errors='coerce')
# r3.to_numpy()

# fair = df_news[5]
# fair =pd.to_numeric(fair,errors='coerce')
# fair.to_numpy()

reward = df_news[1]
reward =pd.to_numeric(reward,errors='coerce')
reward.to_numpy()

tx = df_news[9]
tx=pd.to_numeric(tx,errors='coerce')
tx=tx.to_numpy()

new = df_news[10]
new=pd.to_numeric(new,errors='coerce')
new=new.to_numpy()

throughput = tx / new
fig, ax = plt.subplots()
x=x.tolist()

reward=reward.tolist()
print(x,reward)
plt.figure()
plt.plot(x,reward)

# ax.cla()
 
# # ax.grid(False)
# ax.plot(reward, 'r', lw=1)  # draw line chart

# plt.pause(0.1)

# fig, ax = plt.subplots(3, 2, figsize=(14, 7))
# ax[0, 0].plot(r1, c='r', alpha=0.3)
# ax[0, 0].plot(gaussian_filter1d(r1, sigma=5), c='r', label='rbg_used')
# ax[0, 0].set_xlabel('epoch')
# ax[0, 0].set_ylabel('rbg_used')
# ax[0, 1].plot(r2, c='r', alpha=0.3)
# ax[0, 1].plot(gaussian_filter1d(r2, sigma=5), c='r', label='bler')
# ax[0, 1].set_xlabel('epoch')
# ax[0, 1].set_ylabel('interf1')
# ax[1, 0].plot(reward, c='r', alpha=0.3)
# ax[1, 0].plot(gaussian_filter1d(reward, sigma=5), c='r', label='throughput')
# ax[1, 0].set_xlabel('epoch')
# ax[1, 0].set_ylabel('throughput')
# ax[1, 1].plot(fair, c='r', alpha=0.3)
# ax[1, 1].plot(gaussian_filter1d(fair, sigma=5), c='r', label='fair')
# ax[1, 1].set_xlabel('epoch')
# ax[1, 1].set_ylabel('fair')
# ax[2, 0].plot(r3, c='r', alpha=0.3)
# ax[2, 0].plot(gaussian_filter1d(r3, sigma=5), c='r', label='r1')
# ax[2, 0].set_xlabel('epoch')
# ax[2, 0].set_ylabel('interf')
# ax[2, 1].plot(tx, c='b', alpha=0.3)
# ax[2, 1].plot(gaussian_filter1d(tx, sigma=5), c='r', label='tx')
# ax[2, 1].set_xlabel('epoch')
# ax[2, 1].set_ylabel('tx')
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1)

# plt.savefig(os.path.join(path, './0.1.png'))
# plt.show()
