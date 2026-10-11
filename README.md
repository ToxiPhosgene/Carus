# Neuromorphic Network on Modified Chua Oscillators
Reproduces self-organized criticality in ensembles of 15–1000
modified Chua oscillators with plastic synapses at float16/float32/float64.

## Requirements

- Python 3.11+
- CUDA 12.0+ (for GPU acceleration)
- tqdm >= 4.67.1
- torch >= 2.11.0 + cu130
- nolds >= 0.6.3
- scipy >= 1.16.3
- matplotlib >= 3.10.7
- powerlaw >= 2.0.0
- h5py >= 3.16.0
- pandas >= 2.3.3

## Project Structure
```
Carus/
├── cpp_files/CUDA_Oscillator_Network/
├── critical_analysis/
│ ├── stage_1_analysis.py # DFA and Hurst analysis
│ └── stage_2_analysis.py # Crackling noise analysis
├── core_frame.py # Base classes
├── soc_system.py # Neormorphic network class
├── power_law_analysis.py # Main file for total analysis
├── requirements.txt
├── LICENSE
└── README.md
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run a single experiment with 100 oscillators at float16:

```bash
python.exe -c "from simple_example import simple_example; simple_example(100, 6000, 6000, 'cuda')"
```

### Expected Output

```bash
Skip transient: 100%|█████████████████████████████████████████████████████████| 600000/600000 [15:15<00:00, 655.63it/s]
100%|█████████████████████████████████████████████████████████████████████████| 600000/600000 [15:08<00:00, 660.39it/s]

====================For x====================
For the entire network
tensor([-0.0006, -0.0006, -0.0006,  ...,  0.0007,  0.0009,  0.0010])
Hurst: 0.7200
DFA: 0.8947

For several 10 neurons
tensor([0.0023, 0.0025, 0.0026,  ..., 0.0049, 0.0050, 0.0050])
Hurst: 0.7675
DFA: 1.0775

====================For WEIGHT====================
tensor([10000., 10000., 10000.,  ..., 10000., 10000., 10000.])
Hurst: nan
DFA: nan
```

## Run all configurations
```bash
python.exe -c "from power_law import main; main()"
```
| Parameter | Value | Description |
|---|---|---|
| `volume` | 15, 60, 250, 1,000 | Network size |
| `dtype` | float16, float32, float64 | Computational precision |
| `simulation time` | 600 s | Simulation time |
| `transient time` | 2,400 s | Skip transient |
| `dt` | 1e-3 s | Integration step |
| `n_runs` | 100 | Number of repetitions |
| `stim_freq` | 0.05 Hz | Stimulation frequency |
| `device` | 'cuda' | Device for performing |

## Creating an instance

For to raise a neuromorphic neural network need.
1. Make instance Neuromorphic_Oscillator_Network(volume, transient, device, dtype). Optional arguments is:

- *distribution_epsilon_weight*,
- *distribution_epsilon_weight_range*,
- *distribution_decay_activity_dendrites*,
- *distribution_amplitude_ltd*,
- *distribution_amplitude_ltp*,
- *distribution_apoptosis_threshold*,
- *distribution_inhibitory_ratio*,
- *distribution_neurogenesis_chance*,
- *distribution_new_connection_strength*,
- *distribution_alpha*,
- *distribution_beta*,
- *distribution_m0*,
- *distribution_m1*.
  
2. With .burn_transient() skip transient for stabilized a trajectory.
3. Passing the input vector X to the network.

## Citation (BiBTeX)

## Contact

Ivan A. Gass — gassivan24@gmail.com
ITMO University, Faculty of Software Engineering and Computer Systems, Kronverksky Pr. 49, St. Petersburg, 197101, Russia
