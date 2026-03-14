# Flight Departure Scheduler

基于NSGA-II多目标遗传算法的民航航班起飞排序优化系统

## 项目简介

本项目实现了航班起飞排序的智能优化算法，旨在帮助机场在多跑道环境下高效调度航班起飞。

## 主要特性

- **双算法对比**：FCFS（先到先服务）基准算法 vs NSGA-II多目标优化算法
- **多目标优化**：同时优化总延误时间、总燃油消耗、跑道利用率
- **航班类型支持**：普通航班、VIP航班、紧急航班（带优先级权重）
- **尾流间隔约束**：考虑A320、B737、A350等不同机型的尾流间隔要求
- **可视化分析**：生成3D/2D帕累托前沿图、适应度进化曲线、延误分布图
- **详细报告**：输出跑道利用率、延误分布等详细统计信息

## 技术栈

- Python 3.x
- DEAP (遗传算法框架)
- NumPy & Pandas (数据处理)
- Matplotlib (可视化)

## 安装依赖

```bash
pip install deap numpy pandas matplotlib
```

## 使用方法

### 1. 生成测试数据

```bash
python generate_data.py
```

这将生成150条航班测试数据到 `flight_data.csv`。

### 2. 运行调度算法

```bash
python main.py
```

程序将：
1. 加载航班数据
2. 运行FCFS基准算法
3. 运行NSGA-II优化算法（默认1000代）
4. 输出详细对比结果和可视化图表
5. 保存调度结果到 `results/` 目录

## 输出说明

- `results/fitness_evolution.png` - 适应度进化曲线
- `results/3d_pareto_front.png` - 3D帕累托前沿
- `results/2d_pareto_fronts.png` - 2D帕累托前沿投影
- `results/*_schedule_*.csv` - 调度结果CSV文件

## 算法说明

### FCFS (First Come First Served)

按航班实际到达时间排序，依次分配跑道。

### NSGA-II (Non-dominated Sorting Genetic Algorithm II)

多目标遗传算法，使用以下三个适应度函数：
1. **总延误时间**（权重-1.0，最小化）
2. **总燃油消耗**（权重-1.0，最小化）
3. **跑道利用率**（权重+1.2，最大化）

采用时间感知交叉算子和邻域变异算子，提高搜索效率。

## 项目结构

```
.
├── main.py              # 主程序入口
├── scheduler.py         # NSGA-II算法实现
├── generate_data.py     # 测试数据生成器
├── flight_data.csv      # 航班数据（可重新生成）
└── results/             # 输出结果目录
```

## 许可证

MIT License
