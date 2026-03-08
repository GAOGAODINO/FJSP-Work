"""
禁忌搜索（TS）- 局部搜索优化模型
"""
import random
import numpy as np

class TabuSearch:
    def __init__(self, jobs, num_machines, max_iterations=100, tabu_size=20):
        self.jobs = jobs
        self.num_machines = num_machines
        self.max_iterations = max_iterations
        self.tabu_size = tabu_size

        # 所有工序的扁平列表
        self.all_ops = []
        for job in jobs:
            for op in job:
                self.all_ops.append(op)

        # 工序索引映射
        self.op_to_index = {}
        idx = 0
        for job_id, job in enumerate(jobs):
            for op_idx, _ in enumerate(job):
                self.op_to_index[(job_id, op_idx)] = idx
                idx += 1

        self.total_ops = sum(len(job) for job in jobs)

    def generate_initial_solution(self):
        """生成初始解（使用SPT规则）"""
        all_ops = []
        for job_id, job in enumerate(self.jobs):
            for op_idx, op in enumerate(job):
                all_ops.append({
                    'job_id': job_id,
                    'op_idx': op_idx,
                    'machines': op['machines'],
                    'times': op['times'],
                    'min_time': min(op['times'])
                })

        all_ops.sort(key=lambda x: x['min_time'])

        machine_seq = []
        op_seq = []

        for op in all_ops:
            fastest_idx = np.argmin(op['times'])
            machine_seq.append(op['machines'][fastest_idx])
            op_seq.append(op['job_id'])

        return {'machine_seq': machine_seq, 'op_seq': op_seq}

    def decode(self, individual):
        """解码为调度方案"""
        machine_seq = individual['machine_seq']
        op_seq = individual['op_seq']

        machine_available = [0] * self.num_machines
        job_next_available = [0] * len(self.jobs)
        job_current_op = [0] * len(self.jobs)

        schedule = []
        ops_scheduled = 0

        for job_id in op_seq:
            op_idx = job_current_op[job_id]

            if op_idx >= len(self.jobs[job_id]):
                continue

            global_op_idx = self.op_to_index[(job_id, op_idx)]
            machine = machine_seq[global_op_idx]

            op_data = self.jobs[job_id][op_idx]

            if machine not in op_data['machines']:
                machine = random.choice(op_data['machines'])
                machine_seq[global_op_idx] = machine

            time_idx = op_data['machines'].index(machine)
            proc_time = op_data['times'][time_idx]

            start_time = max(machine_available[machine], job_next_available[job_id])
            end_time = start_time + proc_time

            schedule.append((job_id, op_idx, machine, start_time, end_time))

            machine_available[machine] = end_time
            job_next_available[job_id] = end_time
            job_current_op[job_id] += 1
            ops_scheduled += 1

        if ops_scheduled != self.total_ops:
            return float('inf'), None

        return max(job_next_available), schedule

    def generate_neighbors(self, individual):
        """生成邻域解"""
        neighbors = []
        machine_seq = individual['machine_seq']
        op_seq = individual['op_seq']

        # 邻域操作1：改变一道工序的机器分配
        for i in range(len(machine_seq)):
            op = self.all_ops[i]
            current_machine = machine_seq[i]
            other_machines = [m for m in op['machines'] if m != current_machine]
            if other_machines:
                new_machine_seq = machine_seq.copy()
                new_machine_seq[i] = random.choice(other_machines)
                neighbors.append({
                    'machine_seq': new_machine_seq,
                    'op_seq': op_seq.copy()
                })

        # 邻域操作2：交换两道工序的位置
        for _ in range(min(20, len(op_seq))):
            idx1, idx2 = random.sample(range(len(op_seq)), 2)
            if op_seq[idx1] != op_seq[idx2]:
                new_op_seq = op_seq.copy()
                new_op_seq[idx1], new_op_seq[idx2] = new_op_seq[idx2], new_op_seq[idx1]
                neighbors.append({
                    'machine_seq': machine_seq.copy(),
                    'op_seq': new_op_seq
                })

        return neighbors

    def solve(self):
        """禁忌搜索主过程"""
        current = self.generate_initial_solution()
        current_makespan, _ = self.decode(current)

        best = current.copy()
        best_makespan = current_makespan

        tabu_list = []
        history = [best_makespan]

        print(f"  TS初始解: {best_makespan}")

        for iteration in range(self.max_iterations):
            neighbors = self.generate_neighbors(current)

            best_neighbor = None
            best_neighbor_makespan = float('inf')

            for neighbor in neighbors:
                makespan, _ = self.decode(neighbor)

                is_tabu = False
                for tabu in tabu_list:
                    if (neighbor['machine_seq'] == tabu['machine_seq'] and
                            neighbor['op_seq'] == tabu['op_seq']):
                        is_tabu = True
                        break

                if makespan < best_makespan:
                    best_neighbor = neighbor
                    best_neighbor_makespan = makespan
                    break

                if not is_tabu and makespan < best_neighbor_makespan:
                    best_neighbor = neighbor
                    best_neighbor_makespan = makespan

            if best_neighbor is None:
                break

            current = best_neighbor
            current_makespan = best_neighbor_makespan

            if current_makespan < best_makespan:
                best = current.copy()
                best_makespan = current_makespan

            tabu_list.append(current.copy())
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0)

            history.append(best_makespan)

            if iteration % 20 == 0:
                print(f"    TS第{iteration}代，当前最优: {best_makespan}")

        _, best_schedule = self.decode(best)
        return best, best_makespan, best_schedule, history
