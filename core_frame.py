"""
В этом файле будут размещены все основные классы и функции необходимые для запуска, но не для обучения и инференса
осцилляторной сети.

Принимая сейчас это кольцо "Ignis Natus", я принимаю свою судьбу умереть молодым или стать бессмертным. Надевая сейчас
это кольцо, я призываю всех своих существ из стали и силикона беречь мой огонь, что породил их.
"""

import torch
from tqdm import tqdm


class Oscillator:
    """
    Это эталон
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
    ОРИГИНАЛЬНЫЙ осциллятор Чуа с кусочно-линейной нелинейностью.
    """

    def __init__(self, volume, alpha=8.98462, beta=14.286, m0=-1.143, m1=-0.714,
                 x0=0.1, y0=0.0, z0=0.0, dt=0.01, device="cpu", dtype=torch.float32):
        super().__init__(volume=volume)

        self.volume = volume
        self.dt = dt
        self.device = device
        self.dtype = dtype

        # Параметры системы (оригинальные параметры Чуа)
        self.alpha = torch.full((1, volume), alpha, dtype=self.dtype, device=device)
        self.beta = torch.full((1, volume), beta, dtype=self.dtype, device=device)
        self.m0 = torch.full((1, volume), m0, dtype=self.dtype, device=device)
        self.m1 = torch.full((1, volume), m1, dtype=self.dtype, device=device)

        # Начальные условия
        self.x = torch.full((1, volume), x0, dtype=self.dtype, device=device)
        self.y = torch.full((1, volume), y0, dtype=self.dtype, device=device)
        self.z = torch.full((1, volume), z0, dtype=self.dtype, device=device)

    def chua_nonlinearity(self, x):
        """Кусочно-линейная характеристика диода Чуа"""
        # f(x) = m1*x + 0.5*(m0 - m1)*(|x+1| - |x-1|)
        return self.m1 * x + 0.5 * (self.m0 - self.m1) * (
                torch.abs(x + 1) - torch.abs(x - 1)
        )

    def equations(self, x, y, z, delta):
        """Система дифференциальных уравнений Чуа"""
        f = self.chua_nonlinearity(x)
        dx = self.alpha * (y - x - f) + delta
        dy = x - y + z
        dz = -self.beta * y
        return dx, dy, dz

    def __update_rk4(self, delta):
        """
        Метод Рунге-Кутты 4-го порядка
        """
        # Шаг 1: Вычисляем коэффициенты k1
        k1x, k1y, k1z = self.equations(self.x, self.y, self.z, delta)

        # Шаг 2: Вычисляем коэффициенты k2 (используем половинный шаг от k1)
        x_temp = self.x + k1x * self.dt / 2.0
        y_temp = self.y + k1y * self.dt / 2.0
        z_temp = self.z + k1z * self.dt / 2.0
        k2x, k2y, k2z = self.equations(x_temp, y_temp, z_temp, delta)

        # Шаг 3: Вычисляем коэффициенты k3 (используем половинный шаг от k2)
        x_temp = self.x + k2x * self.dt / 2.0
        y_temp = self.y + k2y * self.dt / 2.0
        z_temp = self.z + k2z * self.dt / 2.0
        k3x, k3y, k3z = self.equations(x_temp, y_temp, z_temp, delta)

        # Шаг 4: Вычисляем коэффициенты k4 (используем полный шаг от k3)
        x_temp = self.x + k3x * self.dt
        y_temp = self.y + k3y * self.dt
        z_temp = self.z + k3z * self.dt
        k4x, k4y, k4z = self.equations(x_temp, y_temp, z_temp, delta)

        # Шаг 5: Обновляем состояния с правильными весами
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
        Чисто для доступа к Рунге-Куртту
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
    Вот тут, короче, реализовано:
            - аксональная активность, чтобы заблокировать чрезмерно активные нейроны;
            - гомеостатическая пластичность, чтобы плавно снижать активность нейронов, при их "зависании" в одной фазе;
            - экспоненциальная средняя активность;
            - нормализация.
    """

    def __init__(self, epsilon_g=5, epsilon_h=1, decay_memory=300, epsilon_homeostasis=10.0,
                 epsilon_axonal_activity=0.05, **kwargs):
        super().__init__(**kwargs)
        """            
        epsilon_g: коэффициент регулирующий нормализацию;
        epsilon_h: коэффициент регулирующий нормализацию;
        decay_memory: коэффициент регулирующий затухание памяти активности;
        epsilon_homeostasis: коэффициент регулирующий гомеостаз;
        epsilon_axonal_activity: коэффициент регулирующий аксональную активность (активно влияет на DFA);
        dtype: параметр отвечающий на каком типе данных работаем.
        """
        # Расчитать как математически выразить g_k математически через epsilon или float16
        self.epsilon_g = epsilon_g
        self.epsilon_h = epsilon_h

        self.active_memory = torch.zeros(size=(1, self.alpha.size(dim=-1)), dtype=self.dtype, device=self.device)
        self.decay_memory = 2 / (1 + decay_memory / self.dt)  # Можно расширить до случайного значения на каждый элемент системы
        self.epsilon_homeostasis = epsilon_homeostasis
        self.homeostasis_ratio = torch.ones(size=self.alpha.size(), dtype=self.dtype, device=self.device)

        self.epsilon_axonal_activity = epsilon_axonal_activity
        self.axonal_activity_ratio = torch.ones(size=self.alpha.size(), dtype=self.dtype, device=self.device)

    def burn_transient(self, transient, disable_tqdm=False):
        """
        Функция для пропускания транзитов.

        :param transient:
        :param disable_tqdm:
        :return:
        """
        x = torch.zeros(size=self.alpha.size(), dtype=self.dtype, device=self.device)
        for _ in tqdm(range(transient), desc="Skip transient", disable=disable_tqdm):
            x = self.__call__(x)

    def __call__(self, x_input, *args, **kwargs):
        """
        Последовательный вызов функций. Пайплайн прохода информации.

        :param x: входное значение
        :param args:
        :param kwargs:
        :return: выходное значение x
        """

        x = self.forward(x_input)  # Пропускаем по определенным механизмам, в частности матричное умножение, что дает овербольшие значения
        x = self._normalization(x)  # Приводим значения к какому-то более менее нормальному значению
        x = self.get_oscillation(x)  # Производим осцилляцию (ака обработку) данных
        x = self._homeostasis(x)  # Оцениваем "усталость" нейрона от пребывания в одной и той же фазе
        x_out = self._axonal_activity(x)  # Оцениваем готовность аксона к потенциалу действия

        return x_out

    def _normalization(self, x):
        """
        Данный блок стабилизирует хаотический аттрактор к циклическому
        :param x:
        :return:

        Возможно стоит заменить на обыкновенную сигмоиду? А то у нас половина входных сигналов в инференсе будет +100 у.е.
            Нет. Не стоит. Эксперименты подтвердили, что возведение экспоненты в степени >10 приводят к разносу сети.
            Лучше настроить на гладкий спуск текущую систему. В ней нет степеней, что безопасно для входа.
        """
        f = self.epsilon_g * torch.sin(3.14 / (2 * self.epsilon_g) * x)
        g = -x.abs() + self.epsilon_g
        h = (g / (g.abs() + self.epsilon_h) + 1) * 0.5

        x = h * f
        return x

    def _homeostasis(self, x):
        """
        Подумать как внедрить этот гомеостаз на управление самим осциллятором, а не только выходным значением

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
        Данная функция напрямую влияет на осциллятор Чуа, а именно на коррекцию внутреннего состояния x, резко устремляя
        его, если он выходит за допустимый диапазон.

        :param x:
        :return:
        """
        self.x = self.x * self.axonal_activity_ratio  # Сначала изменяем корректируем работу осциллятора

        # Затем расчитываем новые корректоры на следующий шаг
        r = (1 / (self.x ** 2 + 1) - 0.5 * self.epsilon_axonal_activity) * self.epsilon_axonal_activity
        self.axonal_activity_ratio = (r / (r.abs() + 1e-3) + 1) * 0.5
        return x
