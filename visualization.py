"""
可视化模块 - 绘制甘特图和收敛曲线
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_gantt(schedule, num_machines, filename, title="Gantt Chart"):
    """画甘特图并保存"""
    if schedule is None or len(schedule) == 0:
        print(f"  警告：没有有效的调度方案，无法画甘特图")
        return

    fig, ax = plt.subplots(figsize=(15, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # 按机器分组
    schedule_by_machine = {m: [] for m in range(num_machines)}
    for item in schedule:
        job_id, op_idx, machine, start, end = item
        if machine in schedule_by_machine:
            schedule_by_machine[machine].append(item)

    # 绘制
    for machine in range(num_machines):
        items = schedule_by_machine[machine]
        items.sort(key=lambda x: x[3])

        for job_id, op_idx, m, start, end in items:
            color = colors[job_id % 20]
            ax.barh(m, end - start, left=start, height=0.8,
                    color=color, edgecolor='black', linewidth=1)

            # 显示标签
            width = end - start
            if width > 2:
                ax.text(start + width / 2, m, f'J{job_id}O{op_idx}',
                        ha='center', va='center', fontsize=8)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Machine', fontsize=12)
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'M{i}' for i in range(num_machines)])
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)

    makespan = max(end for _, _, _, _, end in schedule)
    ax.axvline(x=makespan, color='red', linestyle='--', label=f'Makespan={makespan}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ 甘特图已保存: {filename}")

def plot_convergence(history, filename, title="Convergence Curve"):
    """画收敛曲线并保存"""
    plt.figure(figsize=(10, 6))

    valid_history = [h for h in history if h > 0]
    if not valid_history:
        print(f"  警告：没有有效的收敛数据")
        plt.close()
        return

    plt.plot(range(len(history)), history, 'b-', linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Makespan', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    best_value = min(valid_history)
    best_gen = history.index(best_value)
    plt.plot(best_gen, best_value, 'r*', markersize=15, label=f'Best: {best_value}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 收敛曲线已保存: {filename}")
