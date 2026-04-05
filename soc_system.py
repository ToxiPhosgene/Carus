"""
Не делайте мне мозги. Тут НЕОБХОДИМ torch.float16.

08.03.2026. Тип данных не влияет на качество, но влияет на скорость
"""
from core_frame import *

import time
import torch
import nolds
from scipy import stats
from tqdm import tqdm
from matplotlib import pyplot as plt


class Neuromorphic_Oscillator_Network(Neuromorphic_Oscillator_Chua):
    def __init__(
            self,
            # Пластичность
            distribution_epsilon_weight=None,
            distribution_epsilon_weight_range=None,
            distribution_decay_activity_dendrites=None,
            distribution_amplitude_ltd=None,
            distribution_amplitude_ltp=None,

            # Апоптоз и Нейрогенез
            distribution_apoptosis_threshold=None,
            distribution_inhibitory_ratio=None,
            distribution_neurogenesis_chance=None,
            distribution_new_connection_strength=None,

            # Параметры осциллятора
            distribution_alpha=None,
            distribution_beta=None,
            distribution_m0=None,
            distribution_m1=None,

            transient=4800,
            **kwargs):
        super().__init__(**kwargs)

        # ================== ИНИЦИАЛИЗАЦИЯ ================= #
        """
        Это точечное управление. Точечное - потому что есть возможность настраивать на просто отдельные группки 
        синапсов/нейронов, а вплоть до одного конкретного синапса/нейрона. Здесь, в этой структуре:
         [
                {
                    "start_X": 0,  - Начальная координата по X
                    "end_X": 10,  - Конечная координата по X
                    "start_Y": 0,  - Начальная координата по Y
                    "end_Y": 13,  - Конечная координата по Y
                    "start_value": 10.2,  - Начальное значение генерации
                    "end_value": 10.8  - Конечное значение генерации
                },
                {
                    "start_X": 3,  - Да, они могут накладываться для более сложного "рельефа"
                    "end_X": 20,
                    "start_Y": 15,  - Да, они могут быть одинаковыми, чтобы настроить точечно
                    "end_Y": 15,
                    "start_value": -5,  - Да, тут тоже могут быть одинаковы, чтобы заполнить конкретным числом
                    "end_value": -5
                }
        ]

        Для одномерного массива, просто пропадает "start_Y" и "end_Y". 

        ToDo:
            - Может в будущем сделать не чувствительность к этому?..
        """

        # Пластичность
        if distribution_epsilon_weight is None:
            distribution_epsilon_weight = [
                {
                    "start_X": 0,
                    "end_X": self.volume,
                    "start_Y": 0,
                    "end_Y": self.volume,
                    "start_value": 0.3,
                    "end_value": 0.3
                }
            ]
        if distribution_epsilon_weight_range is None:
            distribution_epsilon_weight_range = [
                {
                    "start_X": 0,
                    "end_X": self.volume,
                    "start_Y": 0,
                    "end_Y": self.volume,
                    "start_value": 1,
                    "end_value": 1
                }
            ]
        if distribution_decay_activity_dendrites is None:
            distribution_decay_activity_dendrites = [
                {
                    "start_X": 0,
                    "end_X": self.volume,
                    "start_Y": 0,
                    "end_Y": self.volume,
                    "start_value": 10,
                    "end_value": 10
                }
            ]
        if distribution_amplitude_ltd is None:
            distribution_amplitude_ltd = [
                {
                    "start_X": 0,
                    "end_X": self.volume,
                    "start_Y": 0,
                    "end_Y": self.volume,
                    "start_value": 0.75,
                    "end_value": 0.75
                }
            ]
        if distribution_amplitude_ltp is None:
            distribution_amplitude_ltp = [
                {
                    "start_X": 0,
                    "end_X": self.volume,
                    "start_Y": 0,
                    "end_Y": self.volume,
                    "start_value": 0.1,
                    "end_value": 0.1
                }
            ]

        # Нейрогенез и апоптоз
        if distribution_apoptosis_threshold is None:
            distribution_apoptosis_threshold = [
                {
                    "start_X": 0,
                    "end_X": self.volume,
                    "start_Y": 0,
                    "end_Y": self.volume,
                    "start_value": 1e-3,
                    "end_value": 1e-3
                }
            ]
        if distribution_inhibitory_ratio is None:
            distribution_inhibitory_ratio = [
                {
                    "start_X": 0,
                    "end_X": self.volume,
                    "start_Y": 0,
                    "end_Y": self.volume,
                    "start_value": 0.5,
                    "end_value": 0.5
                }
            ]
        if distribution_neurogenesis_chance is None:
            distribution_neurogenesis_chance = [
                {
                    "start_X": 0,
                    "end_X": self.volume,
                    "start_Y": 0,
                    "end_Y": self.volume,
                    "start_value": 1 * 1e-4,
                    "end_value": 1 * 1e-4
                }
            ]
        if distribution_new_connection_strength is None:
            distribution_new_connection_strength = [
                {
                    "start_X": 0,
                    "end_X": self.volume,
                    "start_Y": 0,
                    "end_Y": self.volume,
                    "start_value": 6 * 1e-2,
                    "end_value": 6 * 1e-2
                }
            ]

        # Параметры осциллятора
        if distribution_alpha is None:
            distribution_alpha = [
                {
                    "start_X": 0,
                    "end_X": self.volume,
                    "start_value": 10.2,
                    "end_value": 10.8
                }
            ]
        if distribution_beta is None:
            distribution_beta = [
                {
                    "start_X": 0,
                    "end_X": self.volume,
                    "start_value": -4e-4,
                    "end_value": 4e-4
                }
            ]
        if distribution_m0 is None:
            distribution_m0 = [
                {
                    "start_X": 0,
                    "end_X": self.volume,
                    "start_value": -0.2,
                    "end_value": 0.2
                }
            ]
        if distribution_m1 is None:
            distribution_m1 = [
                {
                    "start_X": 0,
                    "end_X": self.volume,
                    "start_value": -0.2,
                    "end_value": 0.2
                }
            ]
        # ================================================== #

        # ====== СОЗДАНИЕ МАТРИЦЫ СИНАПТИЧЕСКИХ ВЕСОВ ====== #
        self.weight = torch.zeros(size=(self.volume, self.volume), dtype=self.dtype, device=self.device)
        # ================================================== #

        # ================== ПЛАСТИЧНОСТЬ ================== #
        self.epsilon_weight = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.epsilon_weight_range = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.decay_activity_dendrites = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.amplitude_ltd = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.amplitude_ltp = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.ema_activity_dendrites = torch.zeros(size=(1, self.alpha.size(dim=-1)), dtype=self.dtype,
                                                  device=self.device)

        distribution_epsilon_weight = self.__standardize_distribution_2d(distribution_epsilon_weight)
        distribution_epsilon_weight_range = self.__standardize_distribution_2d(distribution_epsilon_weight_range)
        distribution_decay_activity_dendrites = self.__standardize_distribution_2d(
            distribution_decay_activity_dendrites)
        distribution_amplitude_ltd = self.__standardize_distribution_2d(distribution_amplitude_ltd)
        distribution_amplitude_ltp = self.__standardize_distribution_2d(distribution_amplitude_ltp)

        self.__apply_distribution_2d(distribution_epsilon_weight, self.epsilon_weight)
        self.__apply_distribution_2d(distribution_epsilon_weight_range, self.epsilon_weight_range)
        self.__apply_distribution_2d(distribution_decay_activity_dendrites, self.decay_activity_dendrites)
        self.__apply_distribution_2d(distribution_amplitude_ltd, self.amplitude_ltd)
        self.__apply_distribution_2d(distribution_amplitude_ltp, self.amplitude_ltp)

        self.decay_activity_dendrites = 2 / (
                    1 + self.decay_activity_dendrites / self.dt)  # Можно расширить до случайного значения на каждый элемент системы
        # ================================================== #

        # =============== АПОПТОЗ И НЕЙРОГЕНЕЗ ============= #
        self.apoptosis_threshold = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.inhibitory_ratio = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.neurogenesis_chance = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.new_connection_strength = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)

        distribution_apoptosis_threshold = self.__standardize_distribution_2d(distribution_apoptosis_threshold)
        distribution_inhibitory_ratio = self.__standardize_distribution_2d(distribution_inhibitory_ratio)
        distribution_neurogenesis_chance = self.__standardize_distribution_2d(distribution_neurogenesis_chance)
        distribution_new_connection_strength = self.__standardize_distribution_2d(distribution_new_connection_strength)

        self.__apply_distribution_2d(distribution_apoptosis_threshold, self.apoptosis_threshold)
        self.__apply_distribution_2d(distribution_inhibitory_ratio, self.inhibitory_ratio)
        self.__apply_distribution_2d(distribution_neurogenesis_chance, self.neurogenesis_chance)
        self.__apply_distribution_2d(distribution_new_connection_strength, self.new_connection_strength)
        # ================================================== #

        # ============== ВНУТРЕННЕЕ СОСТОЯНИЕ ============== #
        self.x_inter_weight = torch.zeros(size=(1, self.alpha.size(dim=-1)), dtype=self.dtype, device=self.device)

        self.alpha = torch.zeros_like(self.alpha, dtype=self.dtype, device=self.device)

        distribution_alpha = self.__standardize_distribution_1d(distributions=distribution_alpha)
        distribution_beta = self.__standardize_distribution_1d(distributions=distribution_beta)
        distribution_m0 = self.__standardize_distribution_1d(distributions=distribution_m0)
        distribution_m1 = self.__standardize_distribution_1d(distributions=distribution_m1)

        self.alpha = self.__apply_distribution_1d(distributions=distribution_alpha, value=self.alpha)
        self.beta = self.__apply_distribution_1d(distributions=distribution_beta, value=self.beta)
        self.m0 = self.__apply_distribution_1d(distributions=distribution_m0, value=self.m0)
        self.m1 = self.__apply_distribution_1d(distributions=distribution_m1, value=self.m1)
        # ================================================== #
        # self.burn_transient(transient=int(transient / self.dt))  # Для экспериментов отключен self.burn_transient

    def forward(self, x_input):
        """
        Важное уточнение, что в данном блоке происходит распараллеливание сигнала, поскольку импульс не только
        обрабатывается, но и параллельно вычисляется.
        Еще одно уточнение - здесь также есть внутреннее состояние сети. Т.е. не только сохраняется влияние осциллятора
        самого на себя, но и осциллятора на осциллятор. Это решение весьма спорное.

        Да, можно было интегрированием через дифуры, но, c'mon, и так сложная система, а с этим мы вообще потеряем в
        производительности

        :param x_input: Входной сигнал
        :return x_out: Выходной сигнал
        """
        self.x_inter_weight += x_input
        x_out = self.x_inter_weight @ self.weight  # Вычисляем выходной сигнал

        self._plasticity(self.x_inter_weight)  # Обновляем веса
        self._apoptosis()
        self._neurogenesis()
        self.x_inter_weight = x_out
        return x_out

    def _plasticity(self, x):
        """
        Изменяет веса, имитируя LTP/LTD

        :param x: начальные входные данные
        :return:
        """
        # Получаем поток информации через дендриты и находим скользящую среднюю активности
        self.information_flow = self.weight * x.transpose(dim0=1, dim1=0)
        # print((self.information_flow != 0).sum())
        # print(self.information_flow.sum())
        # self.ema_activity_dendrites = (self.decay_activity_dendrites * self.information_flow +
        #                                self.ema_activity_dendrites * (1 - self.decay_activity_dendrites))

        # Определяем изменение весов
        self.inc_weight = (self.amplitude_ltp - self.amplitude_ltp * (
            torch.sigmoid((self.epsilon_weight * self.information_flow) ** 3)) ** 0.1 -
                           self.amplitude_ltd * (torch.sigmoid(
                    (self.epsilon_weight_range * self.epsilon_weight * self.information_flow) ** 3)) ** 0.1)
        self.weight += self.inc_weight * self.weight / (
                    self.weight.abs() + 1e-3) * self.dt  # Не забываем про знак синапса!

    def _apoptosis(self):
        mask = (self.weight.abs() > self.apoptosis_threshold).to(dtype=self.weight.dtype, device=self.device)
        self.weight = self.weight * mask

    def _neurogenesis(self):
        # Определяем какие нейроны будут созданы
        new_connection = (torch.rand(size=self.weight.size(), dtype=self.weight.dtype,
                                     device=self.device) < self.neurogenesis_chance).to(dtype=self.weight.dtype)

        # Определяем которые из них будут тормозными
        new_connection *= ((torch.rand(size=self.weight.size(), dtype=self.weight.dtype,
                                       device=self.device) >= self.inhibitory_ratio).to(
            dtype=self.weight.dtype) * 2 - 1)

        # Масштабируем силу связи
        new_connection *= self.new_connection_strength

        # Находим свободные связи в текущей матрице
        new_connection = new_connection * (self.weight == 0)

        # Добавляем к существующей матрице
        self.weight += new_connection

    def __standardize_distribution_1d(self, distributions):
        standard_distributions = []
        for distribution in distributions:
            standard_distributions.append(
                {
                    "start_X": min(distribution["start_X"], self.volume),
                    "end_X": min(distribution["end_X"], self.volume),
                    "start_value": distribution["start_value"],
                    "end_value": distribution["end_value"]
                }
            )
        return standard_distributions

    def __apply_distribution_1d(self, distributions, value):
        for distribution in distributions:
            value[:, distribution["start_X"]:distribution["end_X"]] += torch.rand(
                size=(1, distribution["end_X"] - distribution["start_X"]), dtype=value.dtype, device=self.device
            ) * (distribution["end_value"] - distribution["start_value"]) + distribution["start_value"]

        return value

    def __standardize_distribution_2d(self, distributions):
        standard_distributions = []
        for distribution in distributions:
            standard_distributions.append(
                {
                    "start_X": int(min(distribution["start_X"], self.volume)),
                    "end_X": int(min(distribution["end_X"], self.volume)),
                    "start_Y": int(min(distribution["start_X"], self.volume)),
                    "end_Y": int(min(distribution["end_X"], self.volume)),
                    "start_value": distribution["start_value"],
                    "end_value": distribution["end_value"]
                }
            )
        return standard_distributions

    def __apply_distribution_2d(self, distributions, value):
        for distribution in distributions:
            value[
            distribution["start_Y"]:distribution["end_Y"], distribution["start_X"]:distribution["end_X"]
            ] += torch.rand(
                size=(distribution["end_Y"] - distribution["start_Y"], distribution["end_X"] - distribution["start_X"]),
                dtype=value.dtype, device=self.device
            ) * (distribution["end_value"] - distribution["start_value"]) + distribution["start_value"]

        return value


@torch.no_grad()
def main():
    iteration = 60 * 10

    oscillator = Neuromorphic_Oscillator_Network(
        volume=4000,

        transient=iteration * 4,
        device="cuda",
        dtype=torch.float16
    )
    iteration = int(iteration / oscillator.dt)

    oscillator.burn_transient(transient=iteration * 4)

    signal_from_all_network = []
    signal_from_10_neurons = []
    weights = []
    information_flow = []
    for i in tqdm(range(iteration)):
        if i % int(20 / oscillator.dt):
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
        information_flow.append(
            (oscillator.ema_activity_dendrites).mean().clone().detach().cpu()
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
    information_flow = torch.tensor(
        data=information_flow,
        dtype=torch.float32,
        device="cpu"
    )
    print("\n" + "=" * 20 + "Для x" + "=" * 20)
    print("Для всей сети")
    print(signal_from_all_network)
    print(nolds.hurst_rs(signal_from_all_network))
    print(nolds.dfa(signal_from_all_network))
    print("Для нескольких 10 нейронов")
    print(signal_from_10_neurons)
    print(nolds.hurst_rs(signal_from_10_neurons))
    print(nolds.dfa(signal_from_10_neurons))

    print("\n" + "=" * 20 + "Для ВЕСОВ" + "=" * 20)
    print(weights)
    print(nolds.hurst_rs(weights))
    print(nolds.dfa(weights))

    print("\n" + "=" * 20 + "Для INFORMATION_FLOW" + "=" * 20)
    print(information_flow)
    print(nolds.hurst_rs(information_flow))
    print(nolds.dfa(information_flow))

    plt.plot(
        torch.arange(1, iteration * oscillator.dt + 1, oscillator.dt),
        signal_from_all_network
    )
    plt.plot(
        torch.arange(1, iteration * oscillator.dt + 1, oscillator.dt),
        signal_from_10_neurons
    )
    plt.grid()
    plt.show()

    plt.plot(
        torch.arange(1, iteration * oscillator.dt + 1, oscillator.dt),
        weights
    )
    plt.grid()
    plt.show()

    plt.plot(
        torch.arange(1, iteration * oscillator.dt + 1, oscillator.dt),
        information_flow
    )
    plt.grid()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(17)
    main()