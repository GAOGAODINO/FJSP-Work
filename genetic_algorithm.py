"""
遗传算法（GA）- 全局搜索模型
"""
import random
import numpy as np

class FJSP_GA:
    def __init__(self, jobs, num_machines, pop_size=100, generations=500,
                 crossover_rate=0.8, mutation_rate=0.1):
        self.jobs = jobs
        self.num_machines = num_machines
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

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

        self.num_operations = len(self.all_ops)
        self.total_ops = sum(len(job) for job in jobs)

    def generate_valid_op_seq(self):
        """生成有效的工序序列（每个作业的工序按顺序出现）"""
        op_seq = []
        for job_id, job in enumerate(self.jobs):
            for _ in job:
                op_seq.append(job_id)
        random.shuffle(op_seq)
        return op_seq

    def initialize_population(self):
        """初始化种群 - 确保所有个体都有效"""
        population = []
        for _ in range(self.pop_size):
            # 机器分配
            machine_seq = []
            for op in self.all_ops:
                machine = random.choice(op['machines'])
                machine_seq.append(machine)

            # 工序顺序
            operation_seq = self.generate_valid_op_seq()

            individual = {
                'machine_seq': machine_seq.copy(),
                'op_seq': operation_seq.copy()
            }
            population.append(individual)

        return population

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

            # 确保机器在可选列表中
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
            return None, None

        makespan = max(job_next_available)
        return makespan, schedule

    def fitness(self, individual):
        """适应度函数"""
        makespan, _ = self.decode(individual)
        if makespan is None or makespan <= 0:
            return 0
        return 1.0 / makespan

    def tournament_selection(self, population, fitnesses, k=3):
        """锦标赛选择"""
        selected = []
        pop_size = len(population)

        for _ in range(pop_size):
            indices = random.sample(range(pop_size), min(k, pop_size))
            best_idx = indices[0]
            best_fitness = fitnesses[best_idx]

            for idx in indices[1:]:
                if fitnesses[idx] > best_fitness:
                    best_fitness = fitnesses[idx]
                    best_idx = idx

            selected.append({
                'machine_seq': population[best_idx]['machine_seq'].copy(),
                'op_seq': population[best_idx]['op_seq'].copy()
            })

        return selected

    def crossover(self, parent1, parent2):
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = {
            'machine_seq': parent1['machine_seq'].copy(),
            'op_seq': parent1['op_seq'].copy()
        }
        child2 = {
            'machine_seq': parent2['machine_seq'].copy(),
            'op_seq': parent2['op_seq'].copy()
        }

        # 机器序列交叉（两点交叉）
        size = len(parent1['machine_seq'])
        if size > 1:
            point1, point2 = sorted(random.sample(range(size), 2))
            child1['machine_seq'] = (parent1['machine_seq'][:point1] +
                                     parent2['machine_seq'][point1:point2] +
                                     parent1['machine_seq'][point2:])
            child2['machine_seq'] = (parent2['machine_seq'][:point1] +
                                     parent1['machine_seq'][point1:point2] +
                                     parent2['machine_seq'][point2:])

        # 工序序列交叉
        if len(parent1['op_seq']) > 1:
            point1, point2 = sorted(random.sample(range(len(parent1['op_seq'])), 2))
            child1['op_seq'] = (parent1['op_seq'][:point1] +
                                parent2['op_seq'][point1:point2] +
                                parent1['op_seq'][point2:])
            child2['op_seq'] = (parent2['op_seq'][:point1] +
                                parent1['op_seq'][point1:point2] +
                                parent2['op_seq'][point2:])

        return child1, child2

    def mutate(self, individual):
        """变异操作"""
        if random.random() > self.mutation_rate:
            return individual

        mutated = {
            'machine_seq': individual['machine_seq'].copy(),
            'op_seq': individual['op_seq'].copy()
        }

        if random.random() < 0.5:  # 机器变异
            idx = random.randint(0, len(mutated['machine_seq']) - 1)
            op = self.all_ops[idx]
            current = mutated['machine_seq'][idx]
            other_machines = [m for m in op['machines'] if m != current]
            if other_machines:
                mutated['machine_seq'][idx] = random.choice(other_machines)
        else:  # 顺序变异
            idx1, idx2 = random.sample(range(len(mutated['op_seq'])), 2)
            if mutated['op_seq'][idx1] != mutated['op_seq'][idx2]:
                mutated['op_seq'][idx1], mutated['op_seq'][idx2] = \
                    mutated['op_seq'][idx2], mutated['op_seq'][idx1]

        return mutated

    def evolve(self):
        """主进化过程"""
        population = self.initialize_population()
        best_individual = None
        best_makespan = float('inf')
        best_schedule = None
        history = []

        for gen in range(self.generations):
            fitnesses = []
            valid_count = 0

            for ind in population:
                f = self.fitness(ind)
                fitnesses.append(f)
                if f > 0:
                    valid_count += 1

            makespans = []
            for ind in population:
                makespan, _ = self.decode(ind)
                if makespan is not None:
                    makespans.append(makespan)

            if makespans:
                gen_best = min(makespans)
                if gen_best < best_makespan:
                    best_makespan = gen_best
                    for ind in population:
                        m, s = self.decode(ind)
                        if m == gen_best:
                            best_individual = {
                                'machine_seq': ind['machine_seq'].copy(),
                                'op_seq': ind['op_seq'].copy()
                            }
                            best_schedule = s
                            break

            history.append(best_makespan if best_makespan != float('inf') else 0)

            if gen % 50 == 0:
                valid_pct = (valid_count / self.pop_size) * 100
                print(f"  第{gen}代，有效个体: {valid_pct:.1f}%，当前最优: {best_makespan if best_makespan != float('inf') else 'N/A'}")

            if valid_count == 0:
                print(f"  警告：第{gen}代没有有效个体，重新初始化")
                population = self.initialize_population()
                continue

            selected = self.tournament_selection(population, fitnesses)

            next_population = []
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size:
                    c1, c2 = self.crossover(selected[i], selected[i + 1])
                    next_population.append(c1)
                    next_population.append(c2)
                else:
                    next_population.append(selected[i].copy())

            next_population = [self.mutate(ind) for ind in next_population]

            if best_individual is not None:
                idx = random.randint(0, self.pop_size - 1)
                next_population[idx] = {
                    'machine_seq': best_individual['machine_seq'].copy(),
                    'op_seq': best_individual['op_seq'].copy()
                }

            population = next_population

        return best_individual, best_makespan, best_schedule, history
