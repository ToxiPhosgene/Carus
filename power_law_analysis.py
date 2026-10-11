from core_frame import *
from soc_system import *
from critical_analysis.stage_1_analysis import *
from critical_analysis.stage_2_analysis import *

import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import stats
import warnings

# Подавляем только это конкретное предупреждение
warnings.filterwarnings("ignore", message=".*Initial guess is not within the specified bounds.*")
warnings.filterwarnings('ignore', module='powerlaw')
warnings.filterwarnings('ignore', module='scipy.optimize')
warnings.filterwarnings('ignore', module='library')


# torch.sparse.check_sparse_tensor_invariants.disable()


def run_cycle(name, oscillator, n, disable_tqdm, iteration, transient):
    """
    Главный цикл

    Здесь используется список с сигналами, для оценки доверительного интервала

    :param name:
    :param oscillator:
    :param n:
    :param disable_tqdm:
    :param iteration:
    :param transient:
    :return:
    """

    print("\n" + "=" * 5 + f"Для {name}" + "=" * 5)

    p_bar = tqdm(range(n), disable=(not disable_tqdm))
    signals = list()

    dtype = oscillator.dtype
    device = oscillator.device
    for _ in p_bar:
        # Для установления равновесия осцилляторов
        oscillator.burn_transient(int(transient / oscillator.dt), disable_tqdm=disable_tqdm)

        signal = []

        for i in tqdm(range(int(iteration / oscillator.dt)), desc=f"Collecting data - {name}",
                      disable=disable_tqdm):
            if i % int(20 / oscillator.dt):  # Генерируем импульс каждые 20 секунд
                x = torch.zeros_like(oscillator.alpha)
                noise = (torch.rand(size=(x.size(0), min(50, x.size(1))),
                                    dtype=dtype, device=device) * 2 - 1) * 0.1
                x[:, :noise.size(1)] += noise

            else:
                x = torch.zeros(size=oscillator.alpha.size(), dtype=dtype,
                                device=device)

            x = oscillator(x)
            signal.append(
                x.clone().detach().cpu()
            )
        # signal = torch.tensor(
        #     data=[
        #         signal
        #     ],
        #     dtype=torch.float32
        # )
        signal = torch.concatenate(
            tensors=signal,
            dim=0
        )
        signals.append(signal.to(dtype=torch.float32))

    return signals


@torch.no_grad()
def main():
    volume = 1000
    device = "cuda"
    n = 100

    iteration = 60 * 10
    transient = iteration * 4

    oscillators = {
        "volume: 15, float64":
            Neuromorphic_Oscillator_Network(
                volume=15,

                transient=iteration * 4,
                device=device,
                dtype=torch.float64
            ),
        "volume: 60, float64":
            Neuromorphic_Oscillator_Network(
                volume=60,

                transient=iteration * 4,
                device=device,
                dtype=torch.float64
            ),
        "volume: 250, float64":
            Neuromorphic_Oscillator_Network(
                volume=250,

                transient=iteration * 4,
                device=device,
                dtype=torch.float64
            ),
        "volume: 1000, float64":
            Neuromorphic_Oscillator_Network(
                volume=1000,

                transient=iteration * 4,
                device=device,
                dtype=torch.float64
            )
    }

    if n == 1:
        disable_tqdm = False
    else:
        disable_tqdm = True


    for name in oscillators:
        results = dict()  # Словарь для анализов

        # Прогоняем цикл и собираем сигнал
        signals = run_cycle(name, oscillators[name], n, disable_tqdm, iteration, transient)  # Сигнал размерностью [время, электроды]

        # Анализ DFA и Херста на фрактальность
        results["dfa_herst"] = dfa_and_herst_analysis([signal.sum(dim=-1).unsqueeze(dim=0) for signal in signals])

        # Анализ трескового шума
        # Параметры для анализа
        frame_length = 135  # 100...130
        local_window_length = 135  # 100...130
        sensitive_coef = 3

        # Выделение активности с последующим анализом
        # actions = capek_ribeiro_segmentation([signal.T for signal in signals], frame_length, local_window_length, sensitive_coef)
        results['crackling_noise'] = analyze_crackling_noise([signal.T for signal in signals])

        # Вывод результатов
        print("\nАНАЛИЗ ПО ХЕРСТУ И DFA")
        print(f"Херст:                {results['dfa_herst']['h']['mean']:.4f}±{results['dfa_herst']['h']['std']: .4f}")
        print(f"DFA:                  {results['dfa_herst']['dfa']['mean']:.4f}±{results['dfa_herst']['dfa']['std']: .4f}")
        print(f"Допустимые по Херсту:{results['dfa_herst']['ratio_h']: .2f}%")
        print(f"Допустимые DFA:      {results['dfa_herst']['ratio_dfa']: .2f}%")
        print()

        print("АНАЛИЗ ТРЕСКОВОГО ШУМА")
        for kid in results['crackling_noise']['ratio']:
            print(f"k = {kid}\n"
                  f"gamma: {results['crackling_noise']['gamma'][kid]['slope']:.3f}±{results['crackling_noise']['gamma'][kid]['std']:.3f}\n"
                  f"tau  : {results['crackling_noise']['tau'][kid]['slope']:.3f}±{results['crackling_noise']['tau'][kid]['std']:.3f}\n"
                  f"tau_t: {results['crackling_noise']['tau_t'][kid]['slope']:.3f}±{results['crackling_noise']['tau_t'][kid]['std']:.3f}\n"
                  f"ratio: {results['crackling_noise']['ratio'][kid]['slope']:.3f}±{results['crackling_noise']['ratio'][kid]['std']:.3f}")
        print(f"Лучший результат при k = {results['crackling_noise']['best']['k']}")
        print(f"gamma = {results['crackling_noise']['best']['slope']:.3f}±{results['crackling_noise']['best']['slope_err']:.3f}\n"
              f"ratio = {results['crackling_noise']['best']['ratio']:.3f}±{results['crackling_noise']['best']['ratio_err']:.3f}")

        with open(f"{name.replace(':', '_').replace(',', '_').replace(' ', '')}.txt", "w") as f:
            f.write("\nАНАЛИЗ ПО ХЕРСТУ И DFA\n")
            f.write(
                f"Херст:                {results['dfa_herst']['h']['mean']:.4f}±{results['dfa_herst']['h']['std']: .4f}\n")
            f.write(
                f"DFA:                  {results['dfa_herst']['dfa']['mean']:.4f}±{results['dfa_herst']['dfa']['std']: .4f}\n")
            f.write(f"Допустимые по Херсту:{results['dfa_herst']['ratio_h']: .2f}%\n")
            f.write(f"Допустимые DFA:      {results['dfa_herst']['ratio_dfa']: .2f}%\n")
            f.write("\n")

            f.write("АНАЛИЗ ТРЕСКОВОГО ШУМА\n")
            for kid in results['crackling_noise']['ratio']:
                f.write(f"k = {kid}\n"
                      f"gamma: {results['crackling_noise']['gamma'][kid]['slope']:.3f}±{results['crackling_noise']['gamma'][kid]['std']:.3f}\n"
                      f"tau  : {results['crackling_noise']['tau'][kid]['slope']:.3f}±{results['crackling_noise']['tau'][kid]['std']:.3f}\n"
                      f"tau_t: {results['crackling_noise']['tau_t'][kid]['slope']:.3f}±{results['crackling_noise']['tau_t'][kid]['std']:.3f}\n"
                      f"ratio: {results['crackling_noise']['ratio'][kid]['slope']:.3f}±{results['crackling_noise']['ratio'][kid]['std']:.3f}\n")
            f.write(f"Лучший результат при k = {results['crackling_noise']['best']['k']}\n")
            f.write(
                f"gamma = {results['crackling_noise']['best']['slope']:.3f}±{results['crackling_noise']['best']['slope_err']:.3f}\n"
                f"ratio = {results['crackling_noise']['best']['ratio']:.3f}±{results['crackling_noise']['best']['ratio_err']:.3f}\n")


if __name__ == "__main__":
    main()
    # print(stats.t.ppf(0.9, df=80))
    # actions = [torch.rand(size=(6354, 678))]
    # print(actions[-1].size())
    # t = torch.full(size=(actions[-1].size(-1), ), fill_value=actions[-1].to(dtype=torch.float32).median())
    # print(t.size())