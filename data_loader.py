"""
数据加载模块 - 负责加载FJSP实例
"""
import json

def load_fjsp_instance(filename):
    """加载FJSP的JSON格式实例"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_machines = data['machines']
    jobs = []

    for job_id, job_data in enumerate(data['jobs']):
        operations = []
        for op_id, op_data in enumerate(job_data):
            machines = [item['machine'] for item in op_data]
            times = [item['processing'] for item in op_data]
            operations.append({
                'job_id': job_id,
                'op_idx': op_id,
                'machines': machines,
                'times': times,
                'num_options': len(machines)
            })
        jobs.append(operations)

    total_ops = sum(len(job) for job in jobs)
    print(f"  加载完成：{len(jobs)}个作业，{num_machines}台机器，{total_ops}道工序")
    return jobs, num_machines
