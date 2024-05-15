import numpy as np
import pandas as pd
from collections import OrderedDict
import csv
from km import KM_Algorithm


def main(path,way):
    alg = 'Greedy'
    expert_ori = pd.read_csv('./process_time_matrix2.csv', header=None).drop([0]).values
    job_ori = pd.read_csv('./work_order2.csv', header=None).values
    n = np.where(job_ori[:, 4] == int(path))[0][-1]
    expert = expert_ori[:n]
    job = job_ori[:n]
    expert_num = expert.shape[0]
    task_num = np.max(job[:, 4])
    job_task = job[:, 4].tolist()

    task = [[i + 1] for i in range(task_num)]
    # 提取一张子任务表任务表，第一列为任务序号，第二列为开始时间，
    # 第三列为包含子任务的个数，第四列为一个列表，包含子任务的种类
    for i in range(len(task)):
        task[i].append(job[np.where(job[:, 4] == i + 1)[0], 1][0])  # 任务开始的时间
        task[i].append(len(np.where(job[:, 4] == i + 1)[0]))  # 子任务的个数
        task[i].append(job[np.where(job[:, 4] == i + 1), 2][0].tolist())  # 子任务的种类
    # 建立状态表
    limit = [6 for i in range(20)] + [17 for i in range(4)] + [24]
    cost = [1 for i in range(20)] + [2 for i in range(4)] + [3]
    trans_delay = [2 for i in range(20)] + [1 for i in range(4)] + [3]

    class CVN(object):
        def __init__(self):
            self.task_processing = [1]  # 当前正在处理的任务序号
            self.task_subtask = [[] for i in range(task_num)]  # 所有正在处理的任务包含的子任务的序号
            self.subtask_process = [[] for i in range(job.shape[0])]  # 子任务对应的处理时间 [[子任务序号,处理节点,执行时间],[],...]
            self.subtask_end = [[] for i in range(job.shape[0])]  # 子任务对应的结束时间 [[子任务序号,处理节点,结束时间],[],...]
            self.subtask_cost = [[],[]]  # 子任务的能量消耗
            self.expert_task_processing = [[] for i in range(expert_num)]  # 节点正在处理的子任务序号 节点i:[子任务序号1，2，3...]
            self.expert_end_time = [[] for i in range(expert_num)]  # 节点的子任务结束时间
            self.time = 480
            self.task_end_time = [0 for i in range(task_num)]  # 记录任务的完成时间
            self.id = 1  # 子任务的序号
            self.mean = 0  # 平均执行时间
            self.mean2 = 0  # 节点平均负载
            self.mean3 = 0
            self.chosen = 1
            self._id_parent = 1  # 大的任务的序号
            self.best_fitness = 999999999
            self.buffer = []
            self.module_reduce = 0
            self.request_aggretion = 0
            self.count = 490
            self.frequence = []
            self.energy = [0]

        def step(self, action):  # 动作就是选取的节点，是一个1-25之间的数值
            _time_, kind, _id_parent = job[self.id - 1][1], job[self.id - 1][2], job[self.id - 1][4]
            location = job[self.id - 1][5]

            self.time = _time_

            if self._id_parent != _id_parent:  # 进行下一个子任务的时候，倘若大的任务序号变化，这个时候需要更新状态表
                if path == '3000':
                    if _time_ >= self.count:
                        self.count += 20
                        self.energy.append(0)
                        self.frequence.append(self._id_parent)
                # if way == 'VECN':
                #     start_id = np.where(job[:, 4] == self._id_parent)[0][0]
                #     order = self.buffer.index(max(self.buffer))
                #     transmition = (0 if self.subtask_end[start_id][1]-1 == job[start_id][5]
                #                    else trans_delay[self.subtask_end[start_id][1]-1])
                #     self.task_end_time[self._id_parent-1] += transmition
                #     self.expert_end_time[self.subtask_end[start_id][1]-1][self.expert_end_time
                #     [self.subtask_end[start_id][1]-1].index(self.subtask_end[start_id + order][2])] += transmition
                #     self.subtask_end[start_id + order][2] += transmition

                self._id_parent = _id_parent
                self.buffer = []
                for i in self.task_processing:  # 遍历所有正在执行的任务
                    for j in range(len(self.task_subtask[i - 1]) - 1, -1, -1):  # 遍历所有正在执行的子任务
                        if self.time > self.subtask_end[self.task_subtask[i - 1][j] - 1][2]:  # 根据截止时间判断子任务j是否完成
                            # 更新节点
                            self.expert_task_processing[
                                self.subtask_end[self.task_subtask[i - 1][j] - 1][1] - 1].remove(
                                self.task_subtask[i - 1][j])
                            self.expert_end_time[self.subtask_end[self.task_subtask[i - 1][j] - 1][1] - 1].remove(
                                self.subtask_end[self.task_subtask[i - 1][j] - 1][2])
                            self.task_subtask[i - 1].remove(self.task_subtask[i - 1][j])
                            if len(self.task_subtask[i - 1]) == 0:  # 判断，如果任务i没有子任务了，认为它执行完成了
                                self.task_processing.remove(i)
                self.task_processing.append(_id_parent)

            # 根据action写入状态表
            # 首先寻找action节点对应的处理时间
            trans = (0 if (action - 1) == location else trans_delay[action - 1])
            # if way == 'CVN' else 0
            if len(self.expert_task_processing[action - 1]) < limit[action - 1]:  # 选择的节点暂时没有满负载
                start = self.time

                end = self.time + expert[action - 1][kind] + trans
                       # 任务的完成时间
                if way == 'CVN':
                    for i in self.expert_task_processing[action - 1]:

                        if kind == job[i - 1][2]:
                            end -= 1
                            self.module_reduce += 1

                            break

            else:
                self.expert_end_time[action - 1].sort()
                start = self.expert_end_time[action - 1][-limit[action - 1]] + 1
                end = start + expert[action - 1][kind] + trans
                if way == 'CVN':
                    for i in self.expert_task_processing[action - 1][-limit[action - 1]:]:

                        # if kind == job[i - 1][2]:
                        #     end = self.subtask_end[i - 1][2]
                        #     break
                        self.request_aggretion += trans
                        if kind == job[i - 1][2]:
                            end -= 1
                            self.module_reduce += 1

                            break
            self.energy[-1] += (end - start - trans)*cost[action-1] +trans

            self.subtask_process[self.id - 1] = [self.id, action, start]
            self.subtask_end[self.id - 1] = [self.id, action, end]
            self.expert_end_time[action - 1].append(end)
            self.buffer.append(end)
            self.task_end_time[self._id_parent - 1] = max(self.task_end_time[self._id_parent - 1], end)
            self.expert_task_processing[action - 1].append(self.id)
            self.task_subtask[_id_parent - 1].append(self.id)
            self.id += 1

        def reset(self):
            self.task_processing = []  # 当前正在处理的任务序号
            self.task_subtask = [[] for i in range(task_num)]  # 所有正在处理的任务包含的子任务的序号
            self.subtask_process = [[] for i in range(job.shape[0])]  # 子任务对应的处理时间 [[子任务序号,处理节点,执行时间],[],...]
            self.subtask_end = [[] for i in range(job.shape[0])]  # 子任务对应的结束时间 [[子任务序号,处理节点,结束时间],[],...]
            self.subtask_cost = [[],[]]
            self.expert_task_processing = [[] for i in range(expert_num)]  # 节点正在处理的子任务序号 节点i:[子任务序号1，2，3...]
            self.expert_end_time = [[] for i in range(expert_num)]  # 节点的子任务结束时间
            self.time = 480
            self.task_end_time = [0 for i in range(task_num)]  # 记录任务的完成时间
            self.id = 1  # 子任务的序号
            self.mean = 0  # 平均处理时间
            self.mean2 = 0  # 节点平均负载
            self.mean3 = 0  # 平均执行效率
            self.count = 490
            self._id_parent = 0  # 大的任务的序号
            self.buffer = []
            self.frequence = []
            self.energy = [0]
            self.module_reduce = 0
            self.request_aggretion = 0

        def eval(self):
            sum_waiting = 0
            sum_process = 0
            for i in range(len(self.task_end_time)):
                self.mean += (self.task_end_time[i] - task[i][1])
            self.mean /= task_num
            for j in range(job.shape[0]):  # 本地的成本为1,边缘的成本为1.5，云端的成本为2
                self.mean2 += cost[self.subtask_process[j][1] - 1] * (
                            self.subtask_end[j][2] - self.subtask_process[j][2])
            self.mean2 /= (job[-1][1]-480)
            for i in range(job.shape[0]):
                sum_waiting += self.subtask_process[i][2] - job[i][1]
                sum_process += self.subtask_end[i][2] - self.subtask_process[i][2]
                self.mean3 = sum_process/(sum_process + sum_waiting)
            self.module_reduce /= job.shape[0]
            self.module_reduce -= 0.25
            self.request_aggretion /= (15*job.shape[0])
            self.request_aggretion -= 0.25


        def if_step(self):
            _time_, kind, _id_parent = job[self.id - 1][1], job[self.id - 1][2], job[self.id - 1][4]
            location = job[self.id - 1][5]
            expect_end_time = [0 for i in range(expert_num)]

            for i in range(expert_num):
                moudle_reuse = 0
                for j in self.expert_task_processing[i]:
                    if kind == job[j - 1][2]:
                        moudle_reuse = 1
                        break
                    else:
                        moudle_reuse = 0
                if len(self.expert_task_processing[i]) < limit[i]:
                    expect_end_time[i] = expert[i][kind] + (0 if i == location else trans_delay[i]) - moudle_reuse

                else:
                    self.expert_end_time[i].sort()
                    expect_end_time[i] = (self.expert_end_time[i][-limit[i]] + 1 - _time_)+ 4 + expert[i][kind] + \
                                         (0 if i == location else trans_delay[i]) - moudle_reuse
                self.chosen = expect_end_time.index(min(expect_end_time)) + 1
            return self.chosen

        def if_choose_node(self, id_p):
            start = np.where(job[:, 4] == id_p)[0][0]
            end = np.where(job[:, 4] == id_p)[0][-1]
            _time_ = job[start][1]
            location = job[self.id - 1][5]
            expect_end_time = [0 for i in range(expert_num)]
            for i in range(expert_num):
                expert_time = self.expert_end_time[i][:]  # 将实际的节点状态传入，用以进行假设操作
                expect_time_task = []
                for jj in range(end - start + 1):
                    kind = job[start + jj][2]
                    if len(expert_time) < limit[i]:
                        expert_time.append(expert[i][kind])
                        expect_time_task.append(expert[i][kind])
                    else:
                        expert_time.sort()
                        expert_time.append((expert_time[-limit[i]] + 1-_time_)*2 + 5 + expert[i][kind])
                        expect_time_task.append(expert_time[-limit[i]] + 1 + expert[i][kind] - _time_)
                expect_end_time[i] = - (max(expect_time_task) + (0 if i == location else trans_delay[i]))
            return expect_end_time

        def VECN_time_chose(self):
            time_next = job[self.id - 1][1]
            bipartite_graph = [[0 for i in range(expert_num)] for j in range(expert_num)]
            id_time = np.where(job[:, 1] == time_next)[0]
            task_parent = job[id_time[0]:id_time[-1] + 1, :]

            actions_choose = []
            task_parent_id = list(OrderedDict.fromkeys(task_parent[:, 4]))
            for i in range(len(task_parent_id)):
                bipartite_graph[i] = self.if_choose_node(task_parent_id[i])
            bipartite_graph = [list(col) for col in zip(*bipartite_graph)]
            km = KM_Algorithm(bipartite_graph)
            km.Kuh_Munkras()
            actions_choose1 = km.getResult()
            for i in range(len(task_parent_id)):
                actions_choose += [actions_choose1[i]] * job_task.count(task_parent_id[i])
            return actions_choose

    a = CVN()
    if way == 'CVN':
        for i in range(job.shape[0]):
            action = a.if_step()
            a.step(action)
            print("选择第", i + 1, "个动作:", action)
    else:
        while 1:
            actions_choose = a.VECN_time_chose()
            for i in range(len(actions_choose)):
                a.step(actions_choose[i] + 1)
                print("选择第", i + 1, "个动作:", actions_choose[i])
            if a.id > job.shape[0]:
                break

    a.eval()
    with open('./' + path + "/result_" + path + '_' + alg + way + ".csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(a.subtask_process)
    return [a.mean, a.mean2, a.mean3, a.frequence, a.energy, a.module_reduce, a.request_aggretion]



if __name__ == '__main__':
    #_paths = [0, 100, 235, 548, 1282, 3000]
    _paths = [0, 500, 1000, 1500, 2000, 2500, 3000]
    paths = [str(x) for x in _paths[1:]]
    ways = ['CVN', 'VECN']
    means = [['CVN',0,0,0,0,0,0],['VECN',0,0,0,0,0,0]]
    means2 = [['CVN', 0, 0, 0, 0, 0, 0], ['VECN', 0, 0, 0, 0, 0, 0]]
    means3 = [['CVN', 0, 0, 0, 0, 0, 0], ['VECN', 0, 0, 0, 0, 0, 0]]
    means4 = [['CVN', 0, 0, 0, 0, 0, 0], ['VECN', 0, 0, 0, 0, 0, 0]]
    frequence = []
    energy = [[],[]]
    back = []
    for i in range(len(paths)):
        for j in range(len(ways)):
            back = main(paths[i], ways[j])
            means[j][i + 1] = back[0]
            means2[j][i + 1] = back[1]
            means3[j][i + 1] = back[2]
            energy[j] = back[4]
            if j == 0:
                means4[j][i + 1] = back[5]
                means4[j+1][i + 1] = back[6]
    frequence = back[3]
    frequence.append(3000)

    with open('./' + "comparison.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(_paths)
        writer.writerows(means)
        writer.writerows(means2)
        writer.writerows(means3)
        writer.writerows(means4)
    with open('./' + "fre.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(frequence)
        writer.writerow(energy[0])
        writer.writerow(energy[1])
import draw
# with open('./' + path + "/result_" + path + '_' + alg + way + ".csv", "w", newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(a.subtask_process)

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