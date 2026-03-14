import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms, tools
from scheduler import create_toolbox
import time
import json
from collections import defaultdict
import os
from mpl_toolkits.mplot3d import Axes3D

# 常量定义
N_RUNWAYS = 3
TIME_WINDOW = 180  # 3小时时间窗
RESULTS_DIR = "results"


def load_flight_data(filename="flight_data.csv"):
    """加载并预处理航班数据"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(filename)

    # 数据预处理
    numeric_cols = ['planned', 'actual', 'vortex', 'fuel', 'weight']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 移除异常数据
    df = df[(df['actual'] >= 0) & (df['actual'] <= TIME_WINDOW)]
    return df.to_dict('records')


def run_nsga2(flights, n_gen=1000):
    """运行NSGA-II算法"""
    toolbox = create_toolbox(flights, N_RUNWAYS)

    # 统计设置
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("time", lambda x: time.strftime("%H:%M:%S"))

    # 算法参数
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(50)
    pareto = tools.ParetoFront()
    logbook = tools.Logbook()

    start_time = time.time()

    # 运行算法
    pop, logbook = algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=200, lambda_=300,
        cxpb=0.8, mutpb=0.2,
        ngen=n_gen, stats=stats,
        halloffame=hof, verbose=True
    )

    # 绘制进化曲线和帕累托前沿
    plot_fitness_curves(logbook)
    plot_pareto_front(hof, flights)

    return {
        'pareto_front': [{'solution': ind, 'fitness': ind.fitness.values} for ind in hof],
        'compute_time': time.time() - start_time,
        'logbook': logbook
    }


def plot_fitness_curves(logbook):
    """绘制适应度进化曲线"""
    gen = logbook.select("gen")
    avg = logbook.select("avg")
    min_ = logbook.select("min")
    max_ = logbook.select("max")

    plt.figure(figsize=(15, 10))

    # 三个目标的适应度曲线
    objectives = ['Total Delay (min)', 'Total Fuel (tons)', 'Utilization (%)']
    colors = ['red', 'green', 'blue']

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(gen, [a[i] for a in avg], 'b-', label="Average")
        plt.plot(gen, [m[i] for m in min_], 'r-', label="Minimum")
        plt.plot(gen, [m[i] for m in max_], 'g--', label="Maximum")
        plt.xlabel("Generation")
        plt.ylabel(objectives[i])
        plt.title(f"{objectives[i]} Evolution")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fitness_evolution.png'))
    plt.show()


def plot_pareto_front(hof, flights):
    """绘制帕累托前沿"""
    if len(hof) == 0:
        return

    objectives = np.array([ind.fitness.values for ind in hof])

    # 3D帕累托前沿
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = objectives[:, 0]  # 总延误
    y = objectives[:, 1]  # 总燃油
    z = objectives[:, 2]  # 跑道利用率

    sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=50)
    ax.set_xlabel('Total Delay (min)')
    ax.set_ylabel('Total Fuel (tons)')
    ax.set_zlabel('Utilization (%)')
    ax.set_title('3D Pareto Front')
    fig.colorbar(sc, label='Utilization (%)')

    plt.savefig(os.path.join(RESULTS_DIR, '3d_pareto_front.png'))
    plt.close()

    # 2D投影图
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(x, y, c=z, cmap='viridis')
    plt.xlabel('Total Delay (min)')
    plt.ylabel('Total Fuel (tons)')
    plt.title('Delay vs Fuel')
    plt.colorbar(label='Utilization (%)')

    plt.subplot(1, 3, 2)
    plt.scatter(x, z, c=y, cmap='viridis')
    plt.xlabel('Total Delay (min)')
    plt.ylabel('Utilization (%)')
    plt.title('Delay vs Utilization')
    plt.colorbar(label='Fuel (tons)')

    plt.subplot(1, 3, 3)
    plt.scatter(y, z, c=x, cmap='viridis')
    plt.xlabel('Total Fuel (tons)')
    plt.ylabel('Utilization (%)')
    plt.title('Fuel vs Utilization')
    plt.colorbar(label='Delay (min)')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '2d_pareto_fronts.png'))
    plt.show()


def analyze_solution(solution, flights, algorithm_name="Unknown"):
    """分析调度解决方案"""
    runways = {r: {'last_time': 0, 'first': None, 'last': None, 'flights': []}
               for r in range(N_RUNWAYS)}

    metrics = {
        'algorithm': algorithm_name,
        'total_delay': 0,
        'total_fuel': 0,
        'utilization': 0,
        'schedule': [],
        'runway_stats': defaultdict(dict)
    }
    delays = []

    for idx in solution:
        flight = flights[idx]
        actual = flight['actual']

        selected = min(runways.keys(),
                       key=lambda r: max(actual, runways[r]['last_time']))

        departure = max(actual, runways[selected]['last_time'])
        delay = departure - actual
        delays.append(delay)

        runways[selected]['last_time'] = departure + flight['vortex']
        if runways[selected]['first'] is None:
            runways[selected]['first'] = departure
        runways[selected]['last'] = departure
        runways[selected]['flights'].append({
            'flight_id': flight.get('id', idx),
            'departure': departure,
            'delay': delay
        })

        metrics['schedule'].append({
            'flight_id': flight.get('id', idx),
            'runway': selected + 1,
            'planned': flight.get('planned', 0),
            'actual': actual,
            'departure': departure,
            'delay': delay,
            'weight': flight['weight'],
            'fuel': flight['fuel']
        })

        metrics['total_delay'] += delay * flight['weight']
        metrics['total_fuel'] += flight['fuel'] * delay / 60

    if delays:
        start_times = [r['first'] for r in runways.values() if r['first'] is not None]
        end_times = [r['last'] for r in runways.values() if r['last'] is not None]

        if start_times and end_times:
            total_time = max(end_times) - min(start_times)
            usage = sum(r['last'] - r['first'] for r in runways.values()
                        if r['first'] is not None and r['last'] is not None)
            metrics['utilization'] = (usage / (N_RUNWAYS * total_time)) * 100 if total_time > 0 else 0

    for r in runways:
        runway_flights = runways[r]['flights']
        if runway_flights:
            delays = [f['delay'] for f in runway_flights]
            metrics['runway_stats'][r + 1] = {
                'flight_count': len(runway_flights),
                'avg_delay': np.mean(delays),
                'utilization': (runways[r]['last'] - runways[r]['first']) / total_time if total_time > 0 else 0
            }

    return metrics


def print_detailed_comparison(fcfs_metrics, nsga_metrics):
    """打印详细的指标对比"""
    print("\n" + "=" * 80)
    print(" " * 30 + "DETAILED COMPARISON RESULTS" + " " * 30)
    print("=" * 80)

    # 基本指标对比
    print("\n" + "-" * 40 + " Key Metrics Comparison " + "-" * 40)
    print(f"{'Metric':<25} | {'FCFS':>15} | {'NSGA-II':>15} | {'Improvement':>15}")
    print("-" * 85)

    metrics = [
        ('Total Delay (min)', 'total_delay', '{:.1f}', '↓'),
        ('Total Fuel (tons)', 'total_fuel', '{:.2f}', '↓'),
        ('Utilization (%)', 'utilization', '{:.1f}', '↑'),
    ]

    for name, key, fmt, direction in metrics:
        fcfs_val = fcfs_metrics[key]
        nsga_val = nsga_metrics[key]

        if direction == '↑':
            improvement = (nsga_val - fcfs_val) / abs(fcfs_val) if fcfs_val != 0 else 0
        else:
            improvement = (fcfs_val - nsga_val) / abs(fcfs_val) if fcfs_val != 0 else 0

        print(f"{name:<25} | {fmt.format(fcfs_val):>15} | {fmt.format(nsga_val):>15} | "
              f"{improvement:>+15.1%}")

    # 延误分布统计（简化版）
    print("\n" + "-" * 40 + " Delay Distribution " + "-" * 40)
    fcfs_delays = [f['delay'] for f in fcfs_metrics['schedule']]
    nsga_delays = [f['delay'] for f in nsga_metrics['schedule']]

    delay_stats = [
        ('Average Delay', np.mean, '{:.1f} min'),
        ('Median Delay', np.median, '{:.1f} min'),
        ('90th Percentile', lambda x: np.percentile(x, 90), '{:.1f} min'),
    ]

    print(f"{'Statistic':<20} | {'FCFS':>15} | {'NSGA-II':>15} | {'Improvement':>15}")
    print("-" * 65)
    for name, func, fmt in delay_stats:
        fcfs_stat = func(fcfs_delays) if fcfs_delays else 0
        nsga_stat = func(nsga_delays) if nsga_delays else 0
        improvement = (fcfs_stat - nsga_stat) / fcfs_stat if fcfs_stat != 0 else 0
        print(f"{name:<20} | {fmt.format(fcfs_stat):>15} | {fmt.format(nsga_stat):>15} | "
              f"{improvement:>+15.1%}")

    # 跑道使用情况
    print("\n" + "-" * 40 + " Runway Utilization " + "-" * 40)
    print("\nFlights per Runway:")
    print(f"{'Runway':<10} | {'Flights':>10} | {'Avg Delay':>12} | {'Utilization':>12}")
    print("-" * 65)

    for r in range(1, N_RUNWAYS + 1):
        for algo, metrics in [('FCFS', fcfs_metrics), ('NSGA-II', nsga_metrics)]:
            stats = metrics['runway_stats'].get(r, {})
            print(f"{algo + ' R' + str(r):<10} | "
                  f"{stats.get('flight_count', 0):>10} | "
                  f"{stats.get('avg_delay', 0):>12.1f} | "
                  f"{stats.get('utilization', 0) * 100:>11.1f}%")
        print("-" * 65)

    # 典型航班调度对比
    print("\n" + "-" * 40 + " Sample Flight Scheduling " + "-" * 40)
    print("\nFirst 10 Flights Comparison:")
    print(
        f"{'ID':<8} | {'Algorithm':<10} | {'Runway':<8} | {'Planned':<8} | {'Actual':<8} | {'Departure':<10} | {'Delay':<6} | {'Weight':<6}")
    print("-" * 90)

    for i in range(10):
        if i < len(fcfs_metrics['schedule']):
            fcfs = fcfs_metrics['schedule'][i]
            print(f"{fcfs['flight_id']:<8} | {'FCFS':<10} | "
                  f"{'R' + str(fcfs['runway']):<8} | "
                  f"{fcfs['planned']:<8} | "
                  f"{fcfs['actual']:<8} | "
                  f"{fcfs['departure']:<10} | "
                  f"{fcfs['delay']:<6.1f} | "
                  f"{fcfs['weight']:<6.1f}")

        if i < len(nsga_metrics['schedule']):
            nsga = nsga_metrics['schedule'][i]
            print(f"{nsga['flight_id']:<8} | {'NSGA-II':<10} | "
                  f"{'R' + str(nsga['runway']):<8} | "
                  f"{nsga['planned']:<8} | "
                  f"{nsga['actual']:<8} | "
                  f"{nsga['departure']:<10} | "
                  f"{nsga['delay']:<6.1f} | "
                  f"{nsga['weight']:<6.1f}")
        print("-" * 90)


def visualize_results(fcfs_metrics, nsga_metrics):
    """可视化对比结果"""
    plt.figure(figsize=(18, 12))

    # 指标对比柱状图（简化）
    plt.subplot(2, 2, 1)
    metrics = ['Total Delay', 'Total Fuel', 'Utilization']
    fcfs_values = [fcfs_metrics['total_delay'], fcfs_metrics['total_fuel'], fcfs_metrics['utilization']]
    nsga_values = [nsga_metrics['total_delay'], nsga_metrics['total_fuel'], nsga_metrics['utilization']]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = plt.bar(x - width / 2, fcfs_values, width, label='FCFS')
    bars2 = plt.bar(x + width / 2, nsga_values, width, label='NSGA-II')
    plt.xticks(x, metrics)
    plt.ylabel('Value')
    plt.title('Performance Comparison')
    plt.legend()

    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom')

    # 延误分布箱线图
    plt.subplot(2, 2, 2)
    fcfs_delays = [f['delay'] for f in fcfs_metrics['schedule']]
    nsga_delays = [f['delay'] for f in nsga_metrics['schedule']]
    plt.boxplot([fcfs_delays, nsga_delays],
                labels=['FCFS', 'NSGA-II'],
                showmeans=True)
    plt.ylabel('Delay (minutes)')
    plt.title('Delay Distribution')

    # 跑道利用率
    plt.subplot(2, 2, 3)
    runway_labels = [f'Runway {i + 1}' for i in range(N_RUNWAYS)]
    fcfs_util = [fcfs_metrics['runway_stats'][i + 1]['utilization'] * 100 for i in range(N_RUNWAYS)]
    nsga_util = [nsga_metrics['runway_stats'][i + 1]['utilization'] * 100 for i in range(N_RUNWAYS)]

    x = np.arange(len(runway_labels))
    width = 0.35
    plt.bar(x - width / 2, fcfs_util, width, label='FCFS')
    plt.bar(x + width / 2, nsga_util, width, label='NSGA-II')
    plt.xticks(x, runway_labels)
    plt.ylabel('Utilization (%)')
    plt.title('Runway Utilization')
    plt.legend()




def save_results(fcfs_metrics, nsga_metrics):
    """保存结果到文件"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 保存指标结果（简化）
    results = {
        'timestamp': timestamp,
        'fcfs': fcfs_metrics,
        'nsga2': nsga_metrics,
        'comparison': {
            'total_delay_improvement': (fcfs_metrics['total_delay'] - nsga_metrics['total_delay']) / fcfs_metrics[
                'total_delay'],
            'total_fuel_improvement': (fcfs_metrics['total_fuel'] - nsga_metrics['total_fuel']) / fcfs_metrics[
                'total_fuel'],
            'utilization_improvement': nsga_metrics['utilization'] - fcfs_metrics['utilization']
        }
    }

    # 保存调度表
    def save_schedule(metrics, prefix):
        df = pd.DataFrame(metrics['schedule'])
        df.to_csv(os.path.join(RESULTS_DIR, f'{prefix}_schedule_{timestamp}.csv'), index=False)

    save_schedule(fcfs_metrics, 'fcfs')
    save_schedule(nsga_metrics, 'nsga2')


def main():
    # 加载数据
    flights = load_flight_data()
    print(f"\nLoaded {len(flights)} flights for scheduling")

    # 运行FCFS算法
    print("\nRunning FCFS algorithm...")
    fcfs_solution = sorted(range(len(flights)), key=lambda x: flights[x]['actual'])
    fcfs_metrics = analyze_solution(fcfs_solution, flights, "FCFS")

    # 运行NSGA-II算法
    print("\nRunning NSGA-II algorithm...")
    nsga_result = run_nsga2(flights)

    if not nsga_result['pareto_front']:
        print("Error: No solutions found in Pareto front!")
        return

    # 分析最优解
    best_solution = min(nsga_result['pareto_front'],
                        key=lambda x: 0.7 * x['fitness'][0] + 0.3 * x['fitness'][1])
    nsga_metrics = analyze_solution(best_solution['solution'], flights, "NSGA-II")

    # 结果展示
    print("\n" + "=" * 60)
    print(" " * 20 + "SCHEDULING ALGORITHM COMPARISON" + " " * 20)
    print("=" * 60)

    print_detailed_comparison(fcfs_metrics, nsga_metrics)
    visualize_results(fcfs_metrics, nsga_metrics)
    save_results(fcfs_metrics, nsga_metrics)

    print("\nResults saved to 'results' directory")


if __name__ == "__main__":
    main()