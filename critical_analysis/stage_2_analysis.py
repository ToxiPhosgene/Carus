import torch
import powerlaw
from matplotlib import pyplot as plt
from tqdm import tqdm
import warnings
import numpy as np
import time
from scipy import stats

from critical_analysis.stage_2_analysis_libs import library

# # Подавляем warnings
# warnings.filterwarnings('ignore')


def split_signal_2_frames(signal: torch.Tensor, frame_length: int):
    frames = list()
    for i in range(0, int(signal.size(-1) - frame_length + 1)):
        frames.append(signal[:, i:i+frame_length].unsqueeze(dim=0))
    frames = torch.concatenate(frames, dim=0).transpose(dim0=1, dim1=0)

    return frames


def padding_signal(signal: torch.Tensor, target_signal: torch.Tensor):
    """Векторизованное дополнение сигнала"""
    if signal.size(-1) >= target_signal.size(-1):
        return signal

    pad_size = target_signal.size(-1) - signal.size(-1)
    last_values = signal[:, -1:].expand(-1, pad_size)
    return torch.cat([signal, last_values], dim=-1)


def capek_ribeiro_segmentation(signals: list[torch.Tensor], frame_length: int,
                               local_window_length: int, sensitive_coef: float):
    actions_list = list()
    for signal in signals:
        frames = split_signal_2_frames(signal, frame_length)
        frames_energy = torch.sum(frames ** 2, dim=-1)
        frames_energy = padding_signal(frames_energy, signal)

        frames = split_signal_2_frames(signal, local_window_length)
        frames_mean = frames.mean(dim=-1)
        frames_std = frames.std(dim=-1)

        frames_mean = padding_signal(frames_mean, signal)
        frames_std = padding_signal(frames_std, signal)

        signal_energy = frames_mean + frames_std * sensitive_coef
        actions = signal_energy > frames_energy
        actions_list.append(actions)

    return actions_list


def analyze_crackling_noise(actions: list[torch.Tensor]):
    """

    :param actions: Список сырых сигналов с ЭЭГ в размерности [электроды, время]
    :return:
    """
    ks = list(range(1, 50))
    xpt = 10
    sizes_per_k = [[] for _ in range(len(ks))]
    durations_per_k = [[] for _ in range(len(ks))]
    min_distance = 10e9

    best_k = None
    best_slope = None
    best_ratio = None
    best_slope_err = None
    best_ratio_err = None

    slopes = dict()
    size_slopes = dict()
    duration_slopes = dict()
    correct_slopes = dict()

    for action in actions:
        if type(action) == torch.Tensor:
            action = action.numpy()

        cleaned_array = library.get_removed_array(action)

        sizes, durations = library.get_avalanches_per_k(
            cleaned_array,
            ks=ks,  # масштабы от 1 до 49
            bootstrap=False
        )

        sizes_per_k = [sizes_per_k[kid] + sizes[kid] for kid in range(len(ks))]
        durations_per_k = [durations_per_k[kid] + durations[kid] for kid in range(len(ks))]

    for kid, (sizes_k, durations_k) in enumerate(zip(sizes_per_k, durations_per_k)):
        # Определяем гамму
        xs, ys = library.get_scaling_xs_ys(durations_k, sizes_k)
        slope_, xs_fit, ys_fit, err = library.get_scaling_exponent(xs, ys, xpt=xpt)
        slopes[kid] = {"slope": slope_, "std": err}

        # Определяем тау
        size_slope_, _, _, size_err = library.get_pl_exponent(sizes_k)
        size_slopes[kid] = {"slope": size_slope_, "std": size_err}

        # Определяем тау_t
        duration_slope_, _, _, duration_err = library.get_pl_exponent(durations_k)
        duration_slopes[kid] = {"slope": duration_slope_, "std": duration_err}

        ratio = (duration_slope_ - 1) / (size_slope_ - 1)
        ratio_err = size_err / size_slope_ + duration_err / duration_slope_

        if min_distance > np.abs(ratio - slope_):
            min_distance = np.abs(ratio - slope_)
            best_ratio = ratio
            best_ratio_err = ratio_err
            best_slope = slope_
            best_slope_err = err
            best_k = kid

        if (slope_ - err <= ratio <= slope_ + err) or (slope_ - err <= ratio - ratio_err <= slope_ + err) or (slope_ - err <= ratio + ratio_err <= slope_ + err):
            correct_slopes[kid] = {
                "slope": ratio,
                "std": ratio_err,
                "gamma": slope_,
                "tau": size_slope_,
                "tau_t": duration_slope_
            }

    return {"gamma": slopes, "tau": size_slopes, "tau_t": duration_slopes, "ratio": correct_slopes,
            "best": {
                "k": best_k,
                "slope": best_slope,
                "ratio": best_ratio,
                "slope_err": best_slope_err,
                "ratio_err": best_ratio_err
            }}


def main():
    dt = 0.01
    n_samples = int(60 * 5 / dt)
    frame_length = 3
    local_window_length = 3
    sensitive_coef = 3
    n_signals = 5  # Количество сигналов для обработки

    print(f"Генерация {n_signals} сигналов...")
    signals = []
    for _ in range(n_signals):
        # Генерируем на GPU если доступен
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        signal = torch.randn((1000, n_samples), device=device)
        signals.append(signal)

    print("Выполнение сегментации...")
    start_time = time.time()
    actions = capek_ribeiro_segmentation(signals, frame_length, local_window_length, sensitive_coef)

    seg_time = time.time() - start_time
    print(f"Сегментация: {seg_time:.2f} сек")

    print("\nАнализ crackling noise...")
    start_time = time.time()

    analyze_crackling_noise(actions=[action.cpu().numpy() for action in actions])

    analysis_time = time.time() - start_time
    print(f"Анализ: {analysis_time:.2f} сек")
    print(f"Общее время: {seg_time + analysis_time:.2f} сек")


if __name__ == "__main__":
    main()