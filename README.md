markdown
# FJSP 三种算法对比求解

本项目针对柔性作业车间调度问题（FJSP），实现了三种算法进行对比实验：
- **SPT规则**：基准算法，最短加工时间优先
- **遗传算法（GA）**：全局搜索模型
- **禁忌搜索（TS）**：局部搜索优化模型

## 文件结构
FJSP_Project/
├── main.py # 主程序入口
├── data_loader.py # 数据加载模块
├── genetic_algorithm.py # 遗传算法（GA）
├── tabu_search.py # 禁忌搜索（TS）
├── spt_rule.py # SPT规则（基准算法）
├── visualization.py # 可视化模块
├── requirements.txt # 依赖包
├── README.md # 项目说明
└── results/ # 结果保存文件夹（自动创建）

text

## 安装依赖

```bash
pip install -r requirements.txt
运行实验
bash
python main.py
