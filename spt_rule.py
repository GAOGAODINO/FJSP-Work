"""
SPT规则（基准算法）- 最短加工时间优先
"""
import numpy as np
from genetic_algorithm import FJSP_GA  # 借用GA的解码函数

def priority_rule_schedule(jobs, num_machines):
    """用SPT规则生成基准解"""
    # 收集所有工序
    all_ops = []
    for job_id, job in enumerate(jobs):
        for op_idx, op in enumerate(job):
            all_ops.append({
                'job_id': job_id,
                'op_idx': op_idx,
                'machines': op['machines'],
                'times': op['times'],
                'min_time': min(op['times'])
            })

    # 按最短加工时间排序
    all_ops.sort(key=lambda x: x['min_time'])

    # 构建个体
    machine_seq = []
    op_seq = []

    # 创建临时GA对象用于解码
    temp_ga = FJSP_GA(jobs, num_machines)

    for op in all_ops:
        # 选最快机器
        fastest_idx = np.argmin(op['times'])
        machine_seq.append(op['machines'][fastest_idx])
        op_seq.append(op['job_id'])

    individual = {'machine_seq': machine_seq, 'op_seq': op_seq}
    makespan, schedule = temp_ga.decode(individual)

    return makespan, schedule
