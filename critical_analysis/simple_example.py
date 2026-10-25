from core.soc_system import *
import nolds


def simple_example(volume, iteration, transient, device):
    oscillator = Neuromorphic_Oscillator_Network(
        volume=volume,
        transient=transient,
        device=device,
        dtype=torch.float16
    )
    iteration = int(iteration / oscillator.dt)

    oscillator.burn_transient(transient=int(transient / oscillator.dt))

    signal_from_all_network = []
    signal_from_10_neurons = []
    weights = []
    for i in tqdm(range(iteration)):
        if i % int(20 / oscillator.dt) == 0:
            x = torch.zeros_like(oscillator.alpha)
            noise = (torch.rand(size=(x.size(0), min(50, x.size(1))),
                                dtype=x.dtype, device=x.device) * 2 - 1) * 0.1
            x[:, :noise.size(1)] += noise

            # x = torch.zeros(size=oscillator.alpha.size(), dtype=oscillator.alpha.dtype, device=oscillator.device)
        else:
            x = torch.zeros(size=oscillator.alpha.size(), dtype=oscillator.alpha.dtype, device=oscillator.device)

        x = oscillator(x)
        signal_from_all_network.append(
            x.mean().clone().detach().cpu()
        )
        signal_from_10_neurons.append(
            x[:, -10:].mean().clone().detach().cpu()
        )
        weights.append(
            (oscillator.weight != 0).sum().clone().detach().cpu()
        )

    signal_from_all_network = torch.tensor(
        data=signal_from_all_network,
        dtype=torch.float32,
        device="cpu"
    )
    signal_from_10_neurons = torch.tensor(
        data=signal_from_10_neurons,
        dtype=torch.float32,
        device="cpu"
    )
    weights = torch.tensor(
        data=weights,
        dtype=torch.float32,
        device="cpu"
    )

    print("\n" + "=" * 20 + "For x" + "=" * 20)
    print("For the entire network")
    print(signal_from_all_network)
    print(f"Hurst: {nolds.hurst_rs(signal_from_all_network):.4f}")
    print(f"DFA: {nolds.dfa(signal_from_all_network):.4f}")

    print("\nFor several 10 neurons")
    print(signal_from_10_neurons)
    print(f"Hurst: {nolds.hurst_rs(signal_from_10_neurons):.4f}")
    print(f"DFA: {nolds.dfa(signal_from_10_neurons):.4f}")

    print("\n" + "=" * 20 + "For WEIGHT" + "=" * 20)
    print(weights)
    print(f"Hurst: {nolds.hurst_rs(weights):.4f}")
    print(f"DFA: {nolds.dfa(weights):.4f}")

