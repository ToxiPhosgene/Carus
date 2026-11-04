from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import math
from tqdm import tqdm
import os

torch.manual_seed(111)
permanent_path_to_img = "images_for_test/"
path_to_files_for_article = "files_for_article/"

# Transformation to tensor and back
transform = transforms.Compose([
    transforms.ToTensor(),
])
reverse_transform = transforms.ToPILImage()

def find_center(img_size: torch.Size) -> torch.Tensor:
    height = img_size[-1] // 2
    width = img_size[-2] // 2
    center = torch.tensor(data=[width, height], dtype=torch.int)
    return center


class Retina(nn.Module):
    def __init__(self, img_size: torch.Size, maximum_in_periphery: int, last_layer: bool, collision: float, *, dtype=torch.float32) -> None:
        super().__init__()
        self.dtype = dtype
        self.maximum_in_periphery = maximum_in_periphery
        self.last_layer = last_layer

        self.center = find_center(img_size)
        """
        The structure of receptor field dictionary:
        
            _receptor_field = {
                size:
                    "kernel":
                    "mask":
            }
        """
        self.off_receptor_field = self.__get_receptor_field(img_size, collision, on_center=False)
        self.on_receptor_field = self.__get_receptor_field(img_size, collision, on_center=True)

        self.sizes_off = list(self.off_receptor_field.keys())
        self.sizes_on = list(self.on_receptor_field.keys())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height = x.size(dim=-2)
        width = x.size(dim=-1)

        groups = 1 if self.last_layer else 3

        off_img_rf = torch.zeros_like(x)
        on_img_rf = torch.zeros_like(x)

        for off_center_key, on_center_key in tqdm(zip(self.sizes_off, self.sizes_on)):
            off_kernel = self.off_receptor_field[off_center_key]["kernel"]
            on_kernel = self.on_receptor_field[on_center_key]["kernel"]

            off_mask = self.off_receptor_field[off_center_key]["mask"]
            on_mask = self.on_receptor_field[on_center_key]["mask"]

            off_img_rf += conv_keep_size(x, off_kernel, groups=groups) * off_mask
            on_img_rf += conv_keep_size(x, on_kernel, groups=groups) * on_mask

            # Проверка: размеры должны совпадать с входом
            assert off_img_rf.shape[-2:] == (height, width), (off_img_rf.shape, height, width)
            assert on_img_rf.shape[-2:] == (height, width), (on_img_rf.shape, height, width)

        # Firstly, we computed the receptor fields and when we are calculating activation. Just because receptor fields have influence from neighborhood field
        off_img_rf = F.relu(remove_zero_rows_cols(off_img_rf))  # Оставляем relu, т.к. leaky лишь масштабирует сумму на (1−α). Выбор relu/leaky = вопрос усиления, а не формы нелинейности
        on_img_rf = F.relu(remove_zero_rows_cols(on_img_rf))

        return on_img_rf + off_img_rf

    def __get_conv_by_axis(self, length, img_size, x_or_y:str, collision) -> torch.Tensor:
        s = []
        if x_or_y.lower() == "x":
            x = 0
            while x < length:
                s.append(
                    int(math.sqrt(self.s_function(x, 0, w=img_size[2], h=img_size[1]))) + 1
                )
                x += int(s[-1] * (1 - collision)) if int(s[-1] * (1 - collision)) != 0 else int(s[-1])
        elif x_or_y.lower() == "y":
            y = 0
            while y < length:
                s.append(
                    int(math.sqrt(self.s_function(0, y, w=img_size[2], h=img_size[1]))) + 1
                )
                y += int(s[-1] * (1 - collision) if int(s[-1] * (1 - collision)) != 0 else int(s[-1]))

        convs = []
        for step in s:
            conv = torch.zeros(1, step)
            conv[:, step // 2 - (1 if step % 2 == 0 else 0)] = step
            convs.append(
                conv
            )

        conv = torch.concatenate(convs, dim=-1)
        return conv

    def __get_receptor_field(self, img_size: torch.Size, collision: float, on_center: bool) -> dict:
        center = self.center
        self.x_conv = self.__get_conv_by_axis(length=center[-1], img_size=img_size, x_or_y="x", collision=collision)
        self.y_conv = self.__get_conv_by_axis(length=center[-2], img_size=img_size, x_or_y="y", collision=collision)

        """
        receptor_field_sizes - this is how large receptor fields
        receptor_placement - this is where cluster receptor is placement
        """

        receptor_field_sizes = self.y_conv.T @ self.x_conv  # Матрица центров свертки
        receptor_field_sizes = receptor_field_sizes[:center[0], :center[1]]  # Crop the raw tensor to image size
        receptor_placement = torch.zeros_like(receptor_field_sizes)

        indices = torch.nonzero(receptor_field_sizes, as_tuple=True)
        receptor_placement[indices] = (minmax(1/(receptor_field_sizes[indices])) + 1) * receptor_field_sizes[indices].max()
        receptor_placement = receptor_placement.sqrt().round()

        # Here we're finding unique IDs
        receptor_ids = receptor_placement.unique()
        receptor_ids = receptor_ids[torch.nonzero(receptor_ids, as_tuple=True)].tolist()

        receptor_fields = {}

        for size in receptor_ids:
            receptor_fields[size] = {}
            # When we're using then to create once the kernels with this IDs
            sigma = math.sqrt(size)
            receptor_fields[size]["kernel"] = dog_kernel(size=size, sigma_c=sigma, sigma_s=sigma * 3 / 2, last_layer=self.last_layer, on_center=on_center)

            # Here we're creating masks for convolution. We will being convolution few times instead of thousand
            mask = (receptor_placement == size).float()
            mask = torch.concatenate([torch.flip(mask, dims=[0]), mask], dim=0)
            mask = torch.concatenate([torch.flip(mask, dims=[1]), mask], dim=1)
            receptor_fields[size]["mask"] = mask

        # This section is going to need in method self.recovery_img()
        self.x_conv = (self.x_conv != 0).sum()
        self.y_conv = (self.y_conv != 0).sum()
        return receptor_fields

    def s_function(self, x, y, w, h):
        x = 2 * math.pi * (1 / w * x) ** 2
        y = 2 * math.pi * (1 / h * y) ** 2
        denominator = x + y + 1

        s_value = math.cos(- math.pi / denominator) + 1
        s_value = s_value * self.maximum_in_periphery * 0.5
        return s_value


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def z_scope(x):
    return (x - x.mean()) / x.std()


def normalization(x, epsilon_g:float=5.0, epsilon_h:float=1.0):
    """
    Данный блок стабилизирует хаотический аттрактор к циклическому
    :param epsilon_h:
    :param epsilon_g:
    :param x:
    :return:

    Возможно стоит заменить на обыкновенную сигмоиду? А то у нас половина входных сигналов в инференсе будет +100 у.е.
        Нет. Не стоит. Эксперименты подтвердили, что возведение экспоненты в степени >10 приводят к разносу сети.
        Лучше настроить на гладкий спуск текущую систему. В ней нет степеней, что безопасно для входа.
    """
    f = epsilon_g * torch.sin(3.14 / (2 * epsilon_g) * x)
    g = -x.abs() + epsilon_g
    h = (g / (g.abs() + epsilon_h) + 1) * 0.5

    x = h * f
    return x


def gaussian_kernel(size, sigma):
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return g / g.sum()


def dog_kernel(size=9, sigma_c=1.0, sigma_s=3.0, last_layer: bool=False, on_center=True):
    dim_ = 0 if last_layer else 1
    kc = gaussian_kernel(size, sigma_c)
    ks = gaussian_kernel(size, sigma_s)
    k = kc - ks                           # center without periphery
    k = k - k.mean()
    k = torch.concatenate([k.unsqueeze(dim=0) for _ in range(3)]).unsqueeze(dim=dim_)
    # k = torch.ones(size=(3, 1, int(size), int(size))) / size ** 2
    return -k if on_center else k


def symmetric_pad_for_conv(k):
    """Возвращает (pad_left, pad_right) такие, что при stride=1, dilation=1
    выход conv2d по данной оси равен входу. Работает для чётных и нечётных k."""
    left = (k - 1) // 2
    right = k - 1 - left          # для нечётного k: right == left; для чётного: right == left + 1
    return left, right


def conv_keep_size(x, weight, groups):
    """conv2d с сохранением H и W для произвольного размера ядра."""
    kh, kw = weight.shape[-2], weight.shape[-1]
    pl_h, pr_h = symmetric_pad_for_conv(kh)
    pl_w, pr_w = symmetric_pad_for_conv(kw)
    # F.pad порядок для последних 2 измерений: (left, right, top, bottom)
    x_padded = F.pad(x, (pl_w, pr_w, pl_h, pr_h))
    return F.conv2d(x_padded, weight, stride=1, padding=0, groups=groups)


def remove_zero_rows_cols(t: torch.Tensor):
    # t: [C, H, W]
    # строка H нулевая, если во всех C и W только нули
    row_mask = (t != 0).any(dim=(0, 2))  # [H]
    col_mask = (t != 0).any(dim=(0, 1))  # [W]

    return t[:, row_mask][:, :, col_mask]


def main():
    # Create a folders
    if not os.path.isdir(permanent_path_to_img):
        os.mkdir(permanent_path_to_img)
    if not os.path.isdir(path_to_files_for_article):
        os.mkdir(path_to_files_for_article)

    normal = normalization  # Select normalization tool

    # Open and transform images
    img = Image.open(permanent_path_to_img + "Lenna" + ".png")
    img_ten = transform(img)

    if img_ten.size(-1) % 2 != 0:
        img_ten = img_ten[:, :, :-1]
    if img_ten.size(-2) % 2 != 0:
        img_ten = img_ten[:, :-1, :]

    plt.imshow(reverse_transform(img_ten))
    print("Img size: ", img_ten.size())
    plt.show()

    k = -0.5
    delta = torch.tanh((img_ten - 0.5) / k)
    delta = normal(delta)

    reverse_transform(minmax(delta)).save(path_to_files_for_article + "delta_pr_Lenna.png")

    plt.imshow(
        reverse_transform(delta)
    )
    plt.show()

    plt.imshow(
        reverse_transform(minmax(delta))
    )
    plt.show()

    print("Delta size: ", delta.size())
    retina = Retina(img_size=delta.size(), maximum_in_periphery=100, last_layer=False, collision=1/4)
    img_rf = retina(delta)
    img_rf = normal(img_rf)
    print("Size after first layer: ", img_rf.size())

    plt.imshow(reverse_transform(img_rf))
    plt.savefig(path_to_files_for_article + "clear_convolution.eps", bbox_inches='tight')
    plt.show()

    plt.imshow(reverse_transform(minmax(img_rf)))
    plt.savefig(path_to_files_for_article + "recovery_convolution.eps", bbox_inches='tight')
    plt.show()

    plt.imshow(reverse_transform(minmax(img_rf)**2))
    plt.savefig(path_to_files_for_article + "recovery_convolution_contrast.eps", bbox_inches='tight')
    plt.show()

    retina_ganglion = Retina(img_size=img_rf.size(), maximum_in_periphery=110, last_layer=True, collision=1/4)

    output = retina_ganglion(img_rf)
    output = normal(output)
    print("Size after first layer: ", output.size())

    plt.imshow(reverse_transform(output))
    plt.savefig(path_to_files_for_article + "final_clear_convolution.eps", bbox_inches='tight')
    plt.show()

    plt.imshow(reverse_transform(minmax(output)))
    plt.savefig(path_to_files_for_article + "final_recovery_convolution.eps", bbox_inches='tight')
    plt.show()

    plt.imshow(reverse_transform(minmax(output**2)))
    plt.savefig(path_to_files_for_article + "final_recovery_convolution_contrast.eps", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
