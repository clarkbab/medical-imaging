from matplotlib import pyplot as plt
import pandas as pd

def plot_gpu_usage(name: str) -> None:
    filepath = f'{name}.csv'
    df = pd.read_csv(filepath)
    x = df['time']
    n_gpus = len(df.columns) - 1
    for i in range(n_gpus):
        device_name = f'cuda:{i}'
        key = f'{device_name}-usage'
        y = df[key]
        plt.plot(x, y, label=key)
    plt.title(name)
    plt.xlabel('time [seconds]')
    plt.ylabel('GPU usage [MB]')
    plt.show()
