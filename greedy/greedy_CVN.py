import numpy as np
import pandas as pd
import csv


def main(path,way):
    # 导入数据集
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
    limit = [5 for i in range(20)] + [10 for i in range(4)] + [20]
    cost = [1 for i in range(20)] + [1.5 for i in range(4)] + [2]
    trans_delay = [3 for i in range(20)] + [1.5 for i in range(4)] + [5]

    class CVN(object):
        def __init__(self):
            self.task_processing = []  # 当前正在处理的任务序号
            self.task_subtask = [[] for i in range(task_num)]  # 所有正在处理的任务包含的子任务的序号
            self.subtask_process = [[] for i in range(job.shape[0])]  # 子任务对应的处理时间 [[子任务序号,处理节点,执行时间],[],...]
            self.subtask_end = [[] for i in range(job.shape[0])]  # 子任务对应的结束时间 [[子任务序号,处理节点,结束时间],[],...]
            self.expert_task_processing = [[] for i in range(expert_num)]  # 节点正在处理的子任务序号 节点i:[子任务序号1，2，3...]
            self.expert_end_time = [[] for i in range(expert_num)]  # 节点的子任务结束时间
            self.time = 480
            self.task_end_time = [0 for i in range(task_num)]  # 记录任务的完成时间
            self.id = 1  # 子任务的序号
            self.mean = 0  # 平均执行时间
            self.mean2 = 0  # 节点平均负载
            self.chosen = 1
            self._id_parent = 0  # 大的任务的序号
            self.best_fitness = 999999999

        def step(self, action):  # 动作就是选取的节点，是一个1-25之间的数值
            _time_, kind, _id_parent = job[self.id - 1][1], job[self.id - 1][2], job[self.id - 1][4]
            location = job[self.id - 1][5]
            self.time = _time_
            if self._id_parent != _id_parent:  # 进行下一个子任务的时候，倘若大的任务序号变化，这个时候需要更新状态表
                self._id_parent = _id_parent

                for i in self.task_processing:  # 遍历所有正在执行的任务
                    for j in self.task_subtask[i - 1]:  # 遍历所有正在执行的子任务
                        if self.time > self.subtask_end[j - 1][2]:  # 根据截止时间判断子任务j是否完成
                            self.task_subtask[i - 1].remove(j)
                            if len(self.task_subtask[i - 1]) == 0:  # 判断，如果任务i没有子任务了，认为它执行完成了
                                self.task_processing.remove(i)
                            # 更新节点
                            self.expert_task_processing[self.subtask_end[j - 1][1] - 1].remove(j)
                            self.expert_end_time[self.subtask_end[j - 1][1] - 1].remove(self.subtask_end[j - 1][2])

                self.task_processing.append(_id_parent)

            # 根据action写入状态表
            # 首先寻找action节点对应的处理时间
            if len(self.expert_task_processing[action - 1]) < limit[action - 1]:  # 选择的节点暂时没有满负载
                end = self.time + expert[action - 1][kind] +(0 if (action - 1) == location else trans_delay[action - 1])  # 任务的完成时间
                self.subtask_process[self.id - 1] = [self.id, action, self.time]
                self.subtask_end[self.id - 1] = [self.id, action, end]
                self.expert_end_time[action - 1].append(end)
            else:
                self.expert_end_time[action - 1].sort()
                start = self.expert_end_time[action - 1][-limit[action - 1]] + 1
                end = start + expert[action - 1][kind] + (0 if (action - 1) == location else trans_delay[action - 1])
                self.subtask_process[self.id - 1] = [self.id, action, start + 1]
                self.subtask_end[self.id - 1] = [self.id, action, end]
                self.expert_end_time[action - 1].append(end)

            self.task_end_time[self._id_parent - 1] = max(self.task_end_time[self._id_parent - 1], end)
            self.expert_task_processing[action - 1].append(self.id)
            self.task_subtask[_id_parent - 1].append(self.id)
            self.id += 1

        def reset(self):
            self.task_processing = []  # 当前正在处理的任务序号
            self.task_subtask = [[] for i in range(task_num)]  # 所有正在处理的任务包含的子任务的序号
            self.subtask_process = [[] for i in range(job.shape[0])]  # 子任务对应的处理时间 [[子任务序号,处理节点,执行时间],[],...]
            self.subtask_end = [[] for i in range(job.shape[0])]  # 子任务对应的结束时间 [[子任务序号,处理节点,结束时间],[],...]
            self.expert_task_processing = [[] for i in range(expert_num)]  # 节点正在处理的子任务序号 节点i:[子任务序号1，2，3...]
            self.expert_end_time = [[] for i in range(expert_num)]  # 节点的子任务结束时间
            self.time = 480
            self.task_end_time = [0 for i in range(task_num)]  # 记录任务的完成时间
            self.id = 1  # 子任务的序号
            self.mean = 0 # 平均处理时间
            self.mean2 = 0  # 节点平均负载
            self._id_parent = 0  # 大的任务的序号

        def eval(self):
            for i in range(len(self.task_end_time)):
                self.mean += (self.task_end_time[i] - task[i][1])
            self.mean /= task_num

            for j in range(job.shape[0]):  # 本地的成本为1,边缘的成本为1.5，云端的成本为2
                self.mean2 += cost[self.subtask_process[j][1]-1] * (self.subtask_end[j][2]-self.subtask_process[j][2])
            self.mean2 /= job.shape[0]

        def if_step(self):
            _time_, kind, _id_parent = job[self.id - 1][1], job[self.id - 1][2], job[self.id - 1][4]
            location = job[self.id - 1][5]
            expect_end_time = [0 for i in range(expert_num)]
            if way == 'CVN':
                for i in range(expert_num):
                    if len(self.expert_task_processing[i]) < limit[i]:
                        expect_end_time[i] = _time_ + expert[i][kind] + (0 if i == location else trans_delay[i])
                    else:
                        self.expert_end_time[i].sort()
                        expect_end_time[i] = self.expert_end_time[i][-limit[i]] + 1 + expert[i][kind] + (0 if i == location else trans_delay[i])
                    self.chosen = expect_end_time.index(min(expect_end_time)) + 1
            else:  # VECN选取最大完成时间，要按照整体task的执行时间来决定
                if self._id_parent != _id_parent or _id_parent == 1:
                    print("新任务")
                    num = job_task.count(_id_parent)
                    for i in range(expert_num):
                        expert_time = self.expert_end_time[i][:]  # 将实际的节点状态传入，用以进行假设操作
                        expect_time_task = []
                        for j in range(num):
                            kind = job[self.id - 1 + j][2]
                            if len(expert_time) < limit[i]:
                                expert_time.append(_time_ + expert[i][kind] + (0 if i == location else trans_delay[i]))
                                expect_time_task.append(_time_ + expert[i][kind]+ (0 if i == location else trans_delay[i]))
                            else:
                                expert_time.sort()
                                expert_time.append(expert_time[-limit[i]] + 1 + expert[i][kind]+ (0 if i == location else trans_delay[i]))
                                expect_time_task.append(expert_time[-limit[i]] + 1 + expert[i][kind] +(0 if i == location else trans_delay[i]))
                        expect_end_time[i] = max(expect_time_task)
                    self.chosen = expect_end_time.index(min(expect_end_time)) + 1

            return self.chosen

    a = CVN()
    for i in range(job.shape[0]):
        action = a.if_step()
        a.step(action)
        print("选择第", i + 1, "个动作:", action)
    a.eval()

    with open('./' + path + "/result_" + path + '_' + alg + way + ".csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(a.subtask_process)
    return [a.mean, a.mean2]


if __name__ == '__main__':
    _paths = [0, 100, 235, 548, 1282, 3000]
    paths = [str(x) for x in _paths[1:]]
    ways = ['CVN', 'VECN']
    means = [['CVN',0,0,0,0,0],['VECN',0,0,0,0,0]]
    means2 = [['CVN', 0, 0, 0, 0, 0], ['VECN', 0, 0, 0, 0, 0]]
    for i in range(len(paths)):
        for j in range(len(ways)):
            means[j][i + 1] = main(paths[i], ways[j])[0]
            means2[j][i + 1] = main(paths[i], ways[j])[1]

    main(paths[4], ways[0])
    with open('./' + "comparison.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(_paths)
        writer.writerows(means)
        writer.writerows(means2)