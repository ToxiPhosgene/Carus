import torch
from tqdm import tqdm


class Oscillator:
    """
    This is the benchmark
    """
    def __init__(self, volume, time_step=0.001, generator=None, seed=111, device="cpu"):
        self.time_step = time_step
        self._volume = volume
        self.device = device

        if generator == None:
            self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def __call__(self, x):
        return x

    def size(self):
        return self._volume


class Original_Chua_Oscillator(Oscillator):
    """
    THE ORIGINAL Chua oscillator with piecewise linear nonlinearity.
    """

    def __init__(self, volume, alpha=8.98462, beta=14.286, m0=-1.143, m1=-0.714,
                 x0=0.1, y0=0.0, z0=0.0, dt=0.01, device="cpu", dtype=torch.float32):
        super().__init__(volume=volume)

        self.volume = volume
        self.dt = dt
        self.device = device
        self.dtype = dtype

        # System parameters (original Chua's parameters)
        self.alpha = torch.full((1, volume), alpha, dtype=self.dtype, device=device)
        self.beta = torch.full((1, volume), beta, dtype=self.dtype, device=device)
        self.m0 = torch.full((1, volume), m0, dtype=self.dtype, device=device)
        self.m1 = torch.full((1, volume), m1, dtype=self.dtype, device=device)

        # Initial conditions
        self.x = torch.full((1, volume), x0, dtype=self.dtype, device=device)
        self.y = torch.full((1, volume), y0, dtype=self.dtype, device=device)
        self.z = torch.full((1, volume), z0, dtype=self.dtype, device=device)

    def chua_nonlinear(self, x):
        """Piecewise linear characteristic of the Chua's diode"""
        return self.m1 * x + 0.5 * (self.m0 - self.m1) * (
                torch.abs(x + 1) - torch.abs(x - 1)
        )

    def equations(self, x, y, z, delta):
        """The Chua's system of differential equations"""
        f = self.chua_nonlinear(x)
        dx = self.alpha * (y - x - f) + delta
        dy = x - y + z
        dz = -self.beta * y
        return dx, dy, dz

    def __update_rk4(self, delta):
        """
        The 4th order Runge-Kutta method
        """
        # Step 1: Calculate the coefficients k1
        k1x, k1y, k1z = self.equations(self.x, self.y, self.z, delta)

        # Step 2: Calculate the coefficients k2 (using a half step from k1)
        x_temp = self.x + k1x * self.dt / 2.0
        y_temp = self.y + k1y * self.dt / 2.0
        z_temp = self.z + k1z * self.dt / 2.0
        k2x, k2y, k2z = self.equations(x_temp, y_temp, z_temp, delta)

        # Step 3: Calculate the coefficients k3 (using a half step from k2)
        x_temp = self.x + k2x * self.dt / 2.0
        y_temp = self.y + k2y * self.dt / 2.0
        z_temp = self.z + k2z * self.dt / 2.0
        k3x, k3y, k3z = self.equations(x_temp, y_temp, z_temp, delta)

        # Step 4: Calculate the coefficients k4 (using the full step from k3)
        x_temp = self.x + k3x * self.dt
        y_temp = self.y + k3y * self.dt
        z_temp = self.z + k3z * self.dt
        k4x, k4y, k4z = self.equations(x_temp, y_temp, z_temp, delta)

        # Step 5: Update the states with the correct weights
        self.x += (k1x + 2 * k2x + 2 * k3x + k4x) * self.dt / 6.0
        self.y += (k1y + 2 * k2y + 2 * k3y + k4y) * self.dt / 6.0
        self.z += (k1z + 2 * k2z + 2 * k3z + k4z) * self.dt / 6.0

        return self.x.clone().detach()

    def forward(self, x):
        return x

    def burn_transient(self, transient, disable_tqdm=False):
        for _ in tqdm(range(transient), desc="Skip transient", disable=disable_tqdm):
            self.__update_rk4(
                torch.zeros(
                    self.alpha.size(),
                    dtype=self.dtype,
                    device=self.alpha.device
                )
            )

    def get_oscillation(self, x):
        """
        Purely for access to the Runge-Kurt
        :param x:
        :return:
        """
        x = self.__update_rk4(x)
        return x

    def __call__(self, x, *args, **kwargs):
        x = self.__update_rk4(x)
        x = self.forward(x)
        return x


class Neuromorphic_Oscillator_Chua(Original_Chua_Oscillator):
    """
    Here, in short, it is implemented:
            - axonal activity to block overactive neurons;
            - homeostatic plasticity, in order to smoothly reduce the activity of neurons when they "hang" in one phase;
            - exponential average activity;
            - normalization.
    """

    def __init__(self, epsilon_g=5, epsilon_h=1, decay_memory=300, epsilon_homeostasis=10.0,
                 epsilon_axonal_activity=0.05, **kwargs):
        super().__init__(**kwargs)
        """            
        epsilon_g: coefficient regulating normalization;
        epsilon_h: coefficient regulating normalization;
        decay_memory: a coefficient that regulates the attenuation of activity memory;
        epsilon_homeostasis: coefficient regulating homeostasis;
        epsilon_axonal_activity: coefficient regulating axonal activity (actively affects DFA);
        dtype: the parameter that corresponds to the type of data we are working on.
        """
        # Calculate how to mathematically express g_k mathematically using epsilon or float16
        self.epsilon_g = epsilon_g
        self.epsilon_h = epsilon_h

        self.active_memory = torch.zeros(size=(1, self.alpha.size(dim=-1)), dtype=self.dtype, device=self.device)
        self.decay_memory = 2 / (1 + decay_memory / self.dt)  # It can be expanded to a random value for each element of the system.
        self.epsilon_homeostasis = epsilon_homeostasis
        self.homeostasis_ratio = torch.ones(size=self.alpha.size(), dtype=self.dtype, device=self.device)

        self.epsilon_axonal_activity = epsilon_axonal_activity
        self.axonal_activity_ratio = torch.ones(size=self.alpha.size(), dtype=self.dtype, device=self.device)

    def burn_transient(self, transient, disable_tqdm=False):
        """
        A function for skipping transits.

        :param transient:
        :param disable_tqdm:
        :return:
        """
        x = torch.zeros(size=self.alpha.size(), dtype=self.dtype, device=self.device)
        for _ in tqdm(range(transient), desc="Skip transient", disable=disable_tqdm):
            x = self.__call__(x)

    def __call__(self, x_input: torch.Tensor, *args, **kwargs):
        """
        Sequential function calls. Information passage pipeline.

        :param x: входное значение
        :param args:
        :param kwargs:
        :return: выходное значение x
        """

        x = self.forward(x_input)  # We skip by certain mechanisms, in particular matrix multiplication, which gives over-large values.
        x = self._normalization(x)  # We bring the values to some more or less normal value.
        x = self.get_oscillation(x)  # We produce data oscillation (aka processing)
        x = self._homeostasis(x)  # We evaluate the "fatigue" of a neuron from staying in the same phase
        x_out = self._axonal_activity(x)  # We evaluate the readiness of the axon for the action potential

        return x_out

    def _normalization(self, x):
        """
        This block stabilizes the chaotic attractor to a cyclical one.
        :param x:
        :return:

        Perhaps it should be replaced with an ordinary sigmoid? Otherwise, half of the input signals in the inference will be +100 units.
            No. Not worth it. Experiments have confirmed that raising the exponent to a power of >10 leads to network separation.
            It is better to set up the current system for a smooth descent. There are no degrees in it, which is safe to enter.
        """
        f = self.epsilon_g * torch.sin(3.14 / (2 * self.epsilon_g) * x)
        g = -x.abs() + self.epsilon_g
        h = (g / (g.abs() + self.epsilon_h) + 1) * 0.5

        x = h * f
        return x

    def _homeostasis(self, x):
        """
        Think about how to implement this homeostasis to control the oscillator itself, and not just the output value.

        :param x:
        :return:
        """
        self.active_memory = self.decay_memory * x + self.active_memory * (1 - self.decay_memory)

        x = x * self.homeostasis_ratio * self.dt
        self.homeostasis_ratio = ((1 / (2.71 ** (self.active_memory ** 2 - 2 * 2.71) + 1)) ** 4 +
                                  1 / (2.71 ** ((self.epsilon_homeostasis * self.active_memory) ** 2 - 2 * 2.71) + 1))

        self.homeostasis_ratio *= 1

        return x

    def _axonal_activity(self, x):
        """
        This function directly affects the Chua oscillator, namely, the correction of the internal state of x, sharply directing
        it is used if it is outside the acceptable range.

        :param x:
        :return:
        """
        self.x = self.x * self.axonal_activity_ratio  # First, we change and adjust the operation of the oscillator

        # Then we calculate the new correctors for the next step.
        r = (1 / (self.x ** 2 + 1) - 0.5 * self.epsilon_axonal_activity) * self.epsilon_axonal_activity
        self.axonal_activity_ratio = (r / (r.abs() + 1e-3) + 1) * 0.5
        return x


class Neuromorphic_Oscillator_Network(Neuromorphic_Oscillator_Chua):
    def __init__(
            self,
            # Пластичность
            distribution_epsilon_weight: list=[],
            distribution_epsilon_weight_range: list=[],
            distribution_decay_activity_dendrites: list=[],
            distribution_amplitude_ltd: list=[],
            distribution_amplitude_ltp: list=[],

            # Апоптоз и Нейрогенез
            distribution_apoptosis_threshold: list=[],
            distribution_inhibitory_ratio: list=[],
            distribution_neurogenesis_chance: list=[],
            distribution_new_connection_strength: list=[],

            # Параметры осциллятора
            distribution_alpha: list=[],
            distribution_beta: list=[],
            distribution_m0: list=[],
            distribution_m1: list=[],

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

        # ================== ИНИЦИАЛИЗАЦИЯ ================= #
        self.distribution_epsilon_weight = distribution_epsilon_weight
        self.distribution_epsilon_weight_range = distribution_epsilon_weight_range
        self.distribution_decay_activity_dendrites = distribution_decay_activity_dendrites
        self.distribution_amplitude_ltd = distribution_amplitude_ltd
        self.distribution_amplitude_ltp = distribution_amplitude_ltp
        self.distribution_apoptosis_threshold = distribution_apoptosis_threshold
        self.distribution_inhibitory_ratio = distribution_inhibitory_ratio
        self.distribution_neurogenesis_chance = distribution_neurogenesis_chance
        self.distribution_new_connection_strength = distribution_new_connection_strength
        self.distribution_alpha = distribution_alpha
        self.distribution_beta = distribution_beta
        self.distribution_m0 = distribution_m0
        self.distribution_m1 = distribution_m1
        # ================================================== #

        # ====== СОЗДАНИЕ МАТРИЦЫ СИНАПТИЧЕСКИХ ВЕСОВ ====== #
        self.weight = torch.randn(size=(self.volume, self.volume), dtype=self.dtype, device=self.device)
        # ================================================== #

        # ================== ПЛАСТИЧНОСТЬ ================== #
        self.epsilon_weight = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.epsilon_weight_range = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.decay_activity_dendrites = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.amplitude_ltd = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.amplitude_ltp = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.ema_activity_dendrites = torch.zeros(size=(1, self.volume), dtype=self.dtype,
                                                  device=self.device)

        distribution_epsilon_weight = self.__standardize_distribution_2d(self.distribution_epsilon_weight)
        distribution_epsilon_weight_range = self.__standardize_distribution_2d(self.distribution_epsilon_weight_range)
        distribution_decay_activity_dendrites = self.__standardize_distribution_2d(
            self.distribution_decay_activity_dendrites)
        distribution_amplitude_ltd = self.__standardize_distribution_2d(self.distribution_amplitude_ltd)
        distribution_amplitude_ltp = self.__standardize_distribution_2d(self.distribution_amplitude_ltp)

        self.__apply_distribution_2d(distribution_epsilon_weight, self.epsilon_weight)
        self.__apply_distribution_2d(distribution_epsilon_weight_range, self.epsilon_weight_range)
        self.__apply_distribution_2d(distribution_decay_activity_dendrites, self.decay_activity_dendrites)
        self.__apply_distribution_2d(distribution_amplitude_ltd, self.amplitude_ltd)
        self.__apply_distribution_2d(distribution_amplitude_ltp, self.amplitude_ltp)

        self.decay_activity_dendrites = 2 / (
                1 + self.decay_activity_dendrites / self.time_step)  # Можно расширить до случайного значения на каждый элемент системы
        # ================================================== #

        # =============== АПОПТОЗ И НЕЙРОГЕНЕЗ ============= #
        self.apoptosis_threshold = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.inhibitory_ratio = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.neurogenesis_chance = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)
        self.new_connection_strength = torch.zeros_like(input=self.weight, dtype=self.dtype, device=self.device)

        distribution_apoptosis_threshold = self.__standardize_distribution_2d(self.distribution_apoptosis_threshold)
        distribution_inhibitory_ratio = self.__standardize_distribution_2d(self.distribution_inhibitory_ratio)
        distribution_neurogenesis_chance = self.__standardize_distribution_2d(self.distribution_neurogenesis_chance)
        distribution_new_connection_strength = self.__standardize_distribution_2d(
            self.distribution_new_connection_strength)

        self.__apply_distribution_2d(distribution_apoptosis_threshold, self.apoptosis_threshold)
        self.__apply_distribution_2d(distribution_inhibitory_ratio, self.inhibitory_ratio)
        self.__apply_distribution_2d(distribution_neurogenesis_chance, self.neurogenesis_chance)
        self.__apply_distribution_2d(distribution_new_connection_strength, self.new_connection_strength)
        # ================================================== #

        # ============== ВНУТРЕННЕЕ СОСТОЯНИЕ ============== #
        self.x_inter_weight = torch.zeros(size=(1, self.volume), dtype=self.dtype, device=self.device)
        # ================================================== #

        # =========== OSCILLATOR HYPERPARAMETERS =========== #
        self.alpha = torch.zeros(size=(1, self.volume), dtype=self.dtype)
        self.beta = torch.zeros(size=(1, self.volume), dtype=self.dtype)
        self.m0 = torch.zeros(size=(1, self.volume), dtype=self.dtype)
        self.m1 = torch.zeros(size=(1, self.volume), dtype=self.dtype)

        distribution_alpha = self.__standardize_distribution_1d(self.distribution_alpha)
        distribution_beta = self.__standardize_distribution_1d(self.distribution_beta)
        distribution_m0 = self.__standardize_distribution_1d(self.distribution_m0)
        distribution_m1 = self.__standardize_distribution_1d(self.distribution_m1)

        self.__apply_distribution_1d(distribution_alpha, self.alpha)
        self.__apply_distribution_1d(distribution_beta, self.beta)
        self.__apply_distribution_1d(distribution_m0, self.m0)
        self.__apply_distribution_1d(distribution_m1, self.m1)
        # ================================================== #

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
        def h(x):
            return -(x ** 2 - 1.5 * torch.e)

        # Получаем поток информации через дендриты и находим скользящую среднюю активности
        self.information_flow = self.weight * x.transpose(dim0=1, dim1=0)
        # print((self.information_flow != 0).sum())
        # print(self.information_flow.sum())
        # self.ema_activity_dendrites = (self.decay_activity_dendrites * self.information_flow +
        #                                self.ema_activity_dendrites * (1 - self.decay_activity_dendrites))

        # Определяем изменение весов
        self.inc_weight = (self.amplitude_ltp - self.amplitude_ltp * (
            torch.sigmoid((h(self.epsilon_weight * self.information_flow)) ** 3)) ** 0.1 -
                           self.amplitude_ltd * (torch.sigmoid(
                    (h(self.epsilon_weight_range * self.epsilon_weight * self.information_flow)) ** 3)) ** 0.1)
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
