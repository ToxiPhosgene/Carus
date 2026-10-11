from core_frame import *
from soc_system import *
from critical_analysis.stage_1_analysis import *
from critical_analysis.stage_2_analysis import *

import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings("ignore", message=".*Initial guess is not within the specified bounds.*")

def run_cycle(name, oscillator, n, disable_tqdm, iteration, transient):
    """
    The main cycle

    Here, a list of signals is used to estimate the confidence interval.

    :param name:
    :param oscillator:
    :param n:
    :param disable_tqdm:
    :param iteration:
    :param transient:
    :return:
    """

    print("\n" + "=" * 5 + f"For {name}" + "=" * 5)

    p_bar = tqdm(range(n), disable=(not disable_tqdm))
    signals = list()

    dtype = oscillator.dtype
    device = oscillator.device
    for _ in p_bar:
        # To establish the equilibrium of the oscillators
        oscillator.burn_transient(int(transient / oscillator.dt), disable_tqdm=disable_tqdm)

        signal = []

        for i in tqdm(range(int(iteration / oscillator.dt)), desc=f"Collecting data - {name}",
                      disable=disable_tqdm):
            if i % int(20 / oscillator.dt) == 0:  # We generate a pulse every 20 seconds
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
        signal = torch.concatenate(
            tensors=signal,
            dim=0
        )
        signals.append(signal.to(dtype=torch.float32))

    return signals


@torch.no_grad()
def main():
    device = "cuda"
    n = 4


    iteration = 60 * 0.5
    transient = iteration * 4

    oscillators = {
        "volume: 15, float16":
            Neuromorphic_Oscillator_Network(
                volume=15,

                transient=iteration * 4,
                device=device,
                dtype=torch.float16
            ),
        "volume: 60, float16":
            Neuromorphic_Oscillator_Network(
                volume=60,

                transient=iteration * 4,
                device=device,
                dtype=torch.float16
            ),
        "volume: 250, float16":
            Neuromorphic_Oscillator_Network(
                volume=250,

                transient=iteration * 4,
                device=device,
                dtype=torch.float16
            ),
        "volume: 1000, float16":
            Neuromorphic_Oscillator_Network(
                volume=1000,

                transient=iteration * 4,
                device=device,
                dtype=torch.float16
            ),
        "volume: 15, float32":
            Neuromorphic_Oscillator_Network(
                volume=15,

                transient=iteration * 4,
                device=device,
                dtype=torch.float32
            ),
        "volume: 60, float32":
            Neuromorphic_Oscillator_Network(
                volume=60,

                transient=iteration * 4,
                device=device,
                dtype=torch.float32
            ),
        "volume: 250, float32":
            Neuromorphic_Oscillator_Network(
                volume=250,

                transient=iteration * 4,
                device=device,
                dtype=torch.float32
            ),
        "volume: 1000, float32":
            Neuromorphic_Oscillator_Network(
                volume=1000,

                transient=iteration * 4,
                device=device,
                dtype=torch.float32
            ),
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
        results = dict()  # Dictionary for analysis

        # We run the cycle and collect the signal
        signals = run_cycle(name, oscillators[name], n, disable_tqdm, iteration,
                            transient)  # Signal dimension [time, electrodes]

        # Analysis of DFA and Hurst on fractality
        results["dfa_herst"] = dfa_and_herst_analysis([signal.sum(dim=-1).unsqueeze(dim=0) for signal in signals])

        results['crackling_noise'] = analyze_crackling_noise([signal.T for signal in signals])

        # Output of results
        print("\nHEARST AND DFA ANALYSIS")
        print(f"HEARST:                {results['dfa_herst']['h']['mean']:.4f}±{results['dfa_herst']['h']['std']: .4f}")
        print(f"DFA:                  {results['dfa_herst']['dfa']['mean']:.4f}±{results['dfa_herst']['dfa']['std']: .4f}")
        print(f"Acceptable by Херсту:{results['dfa_herst']['ratio_h']: .2f}%")
        print(f"Acceptable by DFA:      {results['dfa_herst']['ratio_dfa']: .2f}%")
        print()

        print("CRACKLING NOISE ANALYSIS")
        for kid in results['crackling_noise']['ratio']:
            print(f"k = {kid}\n"
                  f"gamma: {results['crackling_noise']['gamma'][kid]['slope']:.3f}±{results['crackling_noise']['gamma'][kid]['std']:.3f}\n"
                  f"tau  : {results['crackling_noise']['tau'][kid]['slope']:.3f}±{results['crackling_noise']['tau'][kid]['std']:.3f}\n"
                  f"tau_t: {results['crackling_noise']['tau_t'][kid]['slope']:.3f}±{results['crackling_noise']['tau_t'][kid]['std']:.3f}\n"
                  f"ratio: {results['crackling_noise']['ratio'][kid]['slope']:.3f}±{results['crackling_noise']['ratio'][kid]['std']:.3f}")
        print(f"Best result with k = {results['crackling_noise']['best']['k']}")
        print(f"gamma = {results['crackling_noise']['best']['slope']:.3f}±{results['crackling_noise']['best']['slope_err']:.3f}\n"
              f"ratio = {results['crackling_noise']['best']['ratio']:.3f}±{results['crackling_noise']['best']['ratio_err']:.3f}")


if __name__ == "__main__":
    main()
