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

    def chua_nonlinearity(self, x):
        """Piecewise linear characteristic of the Chua's diode"""
        # f(x) = m1*x + 0.5*(m0 - m1)*(|x+1| - |x-1|)
        return self.m1 * x + 0.5 * (self.m0 - self.m1) * (
                torch.abs(x + 1) - torch.abs(x - 1)
        )

    def equations(self, x, y, z, delta):
        """The Chua's system of differential equations"""
        f = self.chua_nonlinearity(x)
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

    def __call__(self, x_input, *args, **kwargs):
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
