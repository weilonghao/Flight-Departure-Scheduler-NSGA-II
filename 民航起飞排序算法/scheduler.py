from deap import base, creator, tools
import random
import numpy as np

# 多目标适应度定义
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.2))
creator.create("Individual", list, fitness=creator.FitnessMulti)


def neighbor_mutation(individual, flights, indpb):
    """邻域变异：优先交换时间相近的航班"""
    size = len(individual)
    flight_times = [flights[i]['actual'] for i in individual]

    for i in range(size):
        if random.random() < indpb:
            # 寻找时间最接近的3个邻域
            nearest = np.argsort(np.abs(np.array(flight_times) - flight_times[i]))[1:4]
            j = random.choice(nearest)
            individual[i], individual[j] = individual[j], individual[i]
    return individual,


def time_aware_crossover(ind1, ind2, flights):
    """时间感知交叉：优先保留时间相近的航班块"""
    size = len(ind1)
    # 选择交叉点（考虑时间连续性）
    flight_times = [flights[i]['actual'] for i in ind1]
    cxpoint = sorted(random.sample(range(size), 2))

    # 执行顺序交叉
    tools.cxOrdered(ind1, ind2)
    return ind1, ind2


def create_toolbox(flights, n_runways=3):
    toolbox = base.Toolbox()
    n_flights = len(flights)

    # 生成按实际时间排序的基准序列
    sorted_indices = sorted(range(n_flights), key=lambda x: flights[x]['actual'])

    def init_indices():
        if random.random() < 0.7:  # 70%概率生成基准相关序列
            indices = sorted_indices.copy()
            # 随机交换部分位置（保持大体顺序）
            swaps = min(int(n_flights * 0.15), 20)  # 最多交换15%或20个
            for _ in range(swaps):
                i = random.randint(0, n_flights - 2)
                if abs(flights[indices[i]]['actual'] - flights[indices[i + 1]]['actual']) < 30:  # 只交换时间接近的
                    indices[i], indices[i + 1] = indices[i + 1], indices[i]
            return indices
        return random.sample(range(n_flights), n_flights)

    # 注册遗传算法组件
    toolbox.register("indices", init_indices)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 改进遗传算子
    toolbox.register("mate", time_aware_crossover, flights=flights)
    toolbox.register("mutate", neighbor_mutation, flights=flights, indpb=0.15)
    toolbox.register("select", tools.selNSGA2)

    # 评估函数
    def evaluate(individual, flights, n_runways):
        runways = [{'last_time': 0, 'vortex': None, 'first': None, 'last': None}
                   for _ in range(n_runways)]

        total_delay = 0
        total_fuel = 0
        scheduled = []
        delays = []

        for idx in individual:
            flight = flights[idx]
            actual = flight['actual']
            min_time = float('inf')
            selected = -1

            # 跑道选择策略：考虑尾流间隔和当前空闲时间
            for r in range(n_runways):
                runway = runways[r]
                candidate = max(actual, runway['last_time'])

                # 添加尾流间隔惩罚
                if runway['vortex'] is not None:
                    if flight['weight'] > runway['vortex'] * 1.2:  # 重型飞机需要更长间隔
                        candidate += 1

                if candidate < min_time:
                    min_time = candidate
                    selected = r

            # 计算实际起飞时间
            departure = min_time
            delay = departure - actual
            delays.append(delay)

            # 更新跑道状态
            runways[selected]['last_time'] = departure + flight['vortex']
            runways[selected]['vortex'] = flight['vortex']
            if runways[selected]['first'] is None:
                runways[selected]['first'] = departure
            runways[selected]['last'] = departure

            # 累计指标（考虑航班优先级）
            total_delay += delay * flight['weight']
            total_fuel += flight['fuel'] * delay / 60
            scheduled.append(departure)

        # 计算总时间和利用率
        if not scheduled:
            return (float('inf'), float('inf'), 0)

        total_time = max(scheduled) - min(scheduled)
        usage = sum(r['last'] - r['first'] for r in runways if r['first'] is not None)
        utilization = (usage / (n_runways * total_time)) * 100 if total_time > 0 else 0

        # 添加公平性惩罚项（防止极端延迟）
        fairness_penalty = np.std(delays) * 0.1
        total_delay += fairness_penalty

        return (total_delay, total_fuel, utilization)

    toolbox.register("evaluate", evaluate, flights=flights, n_runways=n_runways)
    return toolbox