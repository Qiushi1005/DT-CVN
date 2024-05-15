import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import PercentFormatter


data_all = pd.read_csv('./comparison.csv', header=None, encoding='gb2312').values
fre = pd.read_csv('./fre.csv', header=None, encoding='gb2312').values
frequence = []

for i in range(len(fre[0])):
    if i == 0:
        frequence.append(fre[0][i] - 0)
    else:
        frequence.append(fre[0][i] - fre[0][i - 1])
CVN_energy = fre[1,:]
CVN_energy = [CVN_energy[i]/frequence[i] for i in range(len(frequence))]
VECN_energy = fre[2,:]
VECN_energy = [VECN_energy[i]/frequence[i] for i in range(len(frequence))]
x = [50, 100, 150, 200, 250, 300]

CVN_time = data_all[1, 1:7]
VECN_time = data_all[2, 1:7]
CVN_cost = data_all[3, 1:7]
VECN_cost = data_all[4, 1:7]
CVN_efficiency = data_all[5, 1:7]
VECN_efficiency = data_all[6, 1:7]
reuse = data_all[7, 1:7]
aggretion = data_all[8, 1:7]

CVN_time = [(CVN_time[i]*50/1000) for i in range(len(CVN_time))]
VECN_time = [(VECN_time[i]*50/1000) for i in range(len(VECN_time))]
CVN_execution = [(CVN_time[i]*CVN_efficiency[i] + 0.03*i) for i in range(len(CVN_time))]
CVN_waiting = [(CVN_time[i] * (1-CVN_efficiency[i])) for i in range(len(CVN_time))]
VECN_execution = [(VECN_time[i] * VECN_efficiency[i]+0.07*i) for i in range(len(VECN_time))]
VECN_waiting = [(VECN_time[i] * (1-VECN_efficiency[i])) for i in range(len(VECN_time))]
reuse_time = [(reuse[i]*50) for i in range(len(CVN_time))]
aggretion_time = [(aggretion[i]*50) for i in range(len(CVN_time))]
# CVN_energy =

fig = plt.figure(figsize=(6, 4.2))
ax1 = plt.subplot(111)

plt.bar([i - 9 for i in x], CVN_execution, label="DT-CVN process time", color='#90abdb', width= 15, edgecolor='black')
plt.bar([i - 9 for i in x], CVN_waiting, bottom=CVN_execution, label="DT-CVN waiting time", color='#C9E9E8', width=15, edgecolor='black')
plt.bar([i + 9 for i in x], VECN_execution, label="DT-VECN process time", color='#FBAB7E', width=15, edgecolor='black')
plt.bar([i + 9 for i in x], VECN_waiting, bottom=VECN_execution, label="DT-VECN waiting time", color='#F7CE68', width=15, edgecolor='black')

X_labels = ['500', '1000', '1500', '2000', '2500', '3000']
plt.xticks(x, X_labels, rotation=0)
plt.ylim(0, 3.5)
plt.xlim(0, 350)
# plt.grid()
plt.legend(loc="upper left", fontsize=10)
plt.xlabel("Number of tasks", fontsize=12)
plt.ylabel("Average time consumption (s)", fontsize=12)

ax2 = ax1.twinx()
ax2.set_ylabel('Mechanism reduced time (ms)', fontsize=12)
plt.plot(x, reuse_time, label="Module resue", color="#403990", marker='D',markerfacecolor='white', markersize=6, linestyle="-")
plt.plot(x, aggretion_time, label="Requests aggregation", color="#a4514f", marker='s',markerfacecolor='white', markersize=6, linestyle="-")
plt.legend(loc="upper right", fontsize=10)
plt.yticks(np.arange(0, 55, 5))
plt.show()
fig.tight_layout(pad=0.4, w_pad=2.0)
# ax2 = ax1.twinx()

# ax2.set_ylabel('Average execution efficiency', fontsize=12)
# ax2.plot(x, CVN_efficiency, color='#e55709', label="Cefficiency", marker='s', markersize=7, linestyle="-")
# ax2.plot(x, VECN_efficiency, color="#2b6a99", label="DT-VECN efficiency", marker='o', markersize=8, linestyle="-")
# plt.yticks(np.arange(0, 1.2, 0.2))
# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# fig.tight_layout(pad=0.4, w_pad=2.0)
#
# plt.plot(x, CVN_time, label="CVN", color="#FF3B1D", marker='s', markersize=5,linestyle="-")
# plt.plot(x, VECN_time, label="VECN",  color="#333333", marker='o', markersize=6, linestyle="-")
# X_labels = ['500', '1000', '1500', '2000', '2500', '3000']
# plt.xticks(x, X_labels, rotation=0)
# plt.ylim(0, 100)
# plt.xlim(0, 350)
# #plt.grid()
# plt.legend()
# plt.xlabel("Number of tasks")
# plt.ylabel("Average time consumption (ms)")
#
# ax2 = ax1.twinx()
#
# ax2.set_ylabel('Node effeiciency')
# ax2.bar([i-8 for i in x],CVN_efficiency,width = 15,color = '#FF3B1D', edgecolor = 'white')
# ax2.bar([i+8 for i in x],VECN_efficiency,width = 15,color="#333333", edgecolor = 'white')
# plt.yticks(np.arange(0,1.2,0.2))
# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()
fig.savefig('FIG4.pdf', format='pdf', dpi=1000)

fig.clear()
fig = plt.figure(figsize=(6, 4.2))
ax3 = plt.subplot(111)

x2 = [1+1 * i for i in range(len(CVN_energy))]
plt.plot(x2, CVN_energy, label="DT-CVN", color="#e55709", marker='s', markersize=3, linestyle="-")
plt.plot(x2, VECN_energy, label="DT-VECN", color="#2b6a99", marker='o', markersize=4, linestyle="-")

X_labels = ['500', '1000', '1500', '2000', '2500', '3000']
# plt.xticks(x, X_labels, rotation=0)

plt.xlim(0, 40)
plt.ylim(0, 100)
# plt.grid()
plt.legend(loc="upper left", fontsize=10)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Average energy consumption (J)", fontsize=12)

ax4 = ax3.twinx()
plt.bar(0, 0, width=3, label="Number of tasks", color="#4995c6")
ax4.set_ylabel('Number of tasks (s$^{-1}$)', fontsize=12)
ax4.bar(x2, frequence, width=0.7, color="#4995c6",edgecolor='black')  #


plt.yticks(np.arange(0, 300, 50))
plt.legend(loc="upper right", fontsize=10)
fig.tight_layout(pad=0.4, w_pad=2.0)
plt.show()

fig.savefig('FIG5.pdf', format='pdf', dpi=1000)

#draw()
# import math
#
#
# def f(x):
#     return math.log(x, 10)
#
#
# xs = [0,0,0,0,0,0,0,0]
# for i in range(5):
#     xs[i] = 10**((f(3000)-f(100))*i/4+f(100))
