# -*- coding: utf-8 -*-


from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import math
from tqdm import tqdm


class Retina(nn.Module):
    def __init__(self, img_size: torch.Size, maximum_in_periphery: int, last_layer: float, collision: float, *, dtype=torch.float32) -> None:
        super().__init__()
        self.dtype = dtype
        self.maximum_in_periphery = maximum_in_periphery
        self.fovea_area = 0
        self.last_layer = last_layer

        self.center = self.__find_center(img_size)
        self.off_receptor_field = self.__get_receptor_field(img_size, self.fovea_area, collision, on_center=False)
        self.on_receptor_field = self.__get_receptor_field(img_size, self.fovea_area, collision, on_center=True)

        self.off_receptor_field = self.off_receptor_field[0::2]
        self.on_receptor_field = self.on_receptor_field[1::2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lst = []
        for on_kernel, off_kernel in tqdm(zip(self.on_receptor_field, self.off_receptor_field), desc="A convolution is taking place"):
            for x_sign in [-1, 1]:
                for y_sign in [-1, 1]:
                    lst.append(
                        self.conv_at_point(
                            x,
                            on_kernel['kernal'],
                            self.center[0] + y_sign * on_kernel['y'],
                            self.center[1] + x_sign * on_kernel['x'],
                        )
                    )

                    lst.append(
                        self.conv_at_point(
                            x,
                            off_kernel['kernal'],
                            self.center[0] + y_sign * off_kernel['y'],
                            self.center[1] + x_sign * off_kernel['x'],
                        )
                    )

        img_rf = self.recovery_img(lst)
        return img_rf

    def recovery_img(self, x: list) -> torch.Tensor:
        """
        This function is recovery images from list to quarter pieces and then concatenate them all
        :param x:
        :return:
        """
        channels = 1 if self.last_layer else 3

        def to_quarter(tiles):
            q = torch.cat(tiles, dim=-1)
            target_shape = (channels, self.y_conv.item(), self.x_conv.item())
            print(target_shape)

            if q.numel() > channels * self.y_conv * self.x_conv:
                q = q[:, :, :, :self.y_conv * self.x_conv]
            elif q.numel() < channels * self.y_conv * self.x_conv:
                k = self.x_conv / self.y_conv
                most_probable_dimension_x, most_probable_dimension_y =  torch.sqrt(q.size(dim=-1) * k).int(), torch.sqrt(q.size(dim=-1) / k).int()

                self.y_conv, self.x_conv = most_probable_dimension_y, most_probable_dimension_x
                q = q[:, :, :, :self.y_conv * self.x_conv]

                # raise ValueError(
                #     f"Too small input tensor: получено {tuple(q.shape)} "
                #     f"(numel={q.numel()}), ожидалось {target_shape} "
                #     f"(numel={channels * self.y_conv * self.x_conv})"
                # )
            target_shape = (channels, self.y_conv.item(), self.x_conv.item())
            print(target_shape, q.size())
            return q.reshape(target_shape)  # reshape безопаснее view

        top_left = to_quarter([x for pair in zip(x[0::8], x[1::8]) for x in pair])
        down_left = to_quarter([x for pair in zip(x[2::8], x[3::8]) for x in pair])
        top_right = to_quarter([x for pair in zip(x[4::8], x[5::8]) for x in pair])
        down_right = to_quarter([x for pair in zip(x[6::8], x[7::8]) for x in pair])

        top = torch.concatenate([top_left.flip([-1]), top_right], dim=-1)
        button = torch.concatenate([down_left.flip([-1]), down_right], dim=-1)
        x = torch.concatenate([top.flip([-2]), button], dim=-2)
        return x

    def __get_conv_by_axis(self, length, img_size, x_or_y:str, collision) -> torch.Tensor:
        s = []
        if x_or_y.lower() == "x":
            x = 0
            while x < length:
                s.append(
                  int(math.sqrt(self.__S_function(x, 0, w=img_size[2], h=img_size[1]))) + 1
                )
                x += int(s[-1] * (1 - collision)) if int(s[-1] * (1 - collision)) != 0 else int(s[-1])
        elif x_or_y.lower() == "y":
            y = 0
            while y < length:
                s.append(
                  int(math.sqrt(self.__S_function(0, y, w=img_size[2], h=img_size[1]))) + 1
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

    def __get_receptor_field(self, img_size: torch.Size, fovea_area: int, collision: float, on_center: bool) -> list:
        receptor_field = []
        center = self.center
        self.x_conv = self.__get_conv_by_axis(length=center[-1], img_size=img_size, x_or_y="x", collision=collision)
        self.y_conv = self.__get_conv_by_axis(length=center[-2], img_size=img_size, x_or_y="y", collision=collision)
        conv = self.y_conv.T @ self.x_conv  # Матрица центров свертки

        indices = torch.nonzero(conv, as_tuple=True)
        conv[indices] = (minmax(1/(conv[indices])) + 1) * conv[indices].max()
        conv = conv.sqrt().round()

        # Новое. Реализует поиск уникальных размеров
        sizes = conv.unique().tolist()
        kernals = {}
        for size in sizes:
            sigma = math.sqrt(size)
            kernals[size] = dog_kernel(size=size, sigma_c=sigma, sigma_s=sigma * 3/2, last_layer=self.last_layer, on_center=on_center)

        i = 0
        for index_0, index_1 in tqdm(zip(indices[0], indices[1]), desc="Creating kernels"):
            kernal = kernals[conv[index_0, index_1].tolist()]
            receptor_field.append(
                {
                    'x': index_1,
                    'y': index_0,
                    'kernal': kernal if kernal.sum() != 0 else kernal + (-1.0 if on_center else 1.0),
                }
            )
            i += 1
        self.x_conv = (self.x_conv != 0).sum()
        self.y_conv = (self.y_conv != 0).sum()
        return receptor_field

    def __find_center(self, img_size: torch.Size) -> torch.Tensor:
        height = img_size[-1] // 2
        width = img_size[-2] // 2
        center = torch.tensor(data=[width, height], dtype=torch.int)
        return center

    def conv_at_point(self, x, kernel, i, j, padding=0):
        """
        x: (C_in, H, W) или (1, C_in, H, W)
        kernel: (C_out, C_in, kH, kW)
        i, j: координаты центра окна (в пикселях)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        kH, kW = kernel.shape[-2:]
        rH, rW = kH // 2, kW // 2

        H, W = x.size(-2), x.size(-1)          # H = rows, W = cols

        # top-left corner of the window, clamped into the image
        top  = min(max(i - rH, 0), H - kH)
        left = min(max(j - rW, 0), W - kW)

        patch = x[:, :, top:top + kH, left:left + kW]

        assert patch.shape[-2:] == (kH, kW), (
            f"patch {tuple(patch.shape[-2:])} != kernel {(kH, kW)} "
            f"at (i={i}, j={j}), x shape={tuple(x.shape)}, rH={rH}, rW={rW}"
        )
        return F.conv2d(patch, kernel, groups=1 if self.last_layer else 3, padding=padding)

    def __S_function(self, x, y, w, h):
        x = 2 * math.pi * (1 / w * x) ** 2
        y = 2 * math.pi * (1 / h * y) ** 2
        dinominator = x + y + 1

        S_value = math.cos(- math.pi / dinominator) + 1
        S_value = S_value * self.maximum_in_periphery * 0.5
        return S_value


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


def dog_kernel(size=9, sigma_c=1.0, sigma_s=3.0, last_layer=False, on_center=True):
    dim_ = 0 if last_layer else 1
    kc = gaussian_kernel(size, sigma_c)
    ks = gaussian_kernel(size, sigma_s)
    k = kc - ks                           # center without periphery
    k = k - k.mean()
    k = torch.concatenate([k.unsqueeze(dim=0) for _ in range(3)]).unsqueeze(dim=dim_)
    k = torch.ones(size=(3, 1, int(size), int(size))) / size ** 2
    return -k if on_center else k


def main():
    # Transformation to tensor and back
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    reverse_transform = transforms.ToPILImage()

    normal = normalization  # Select normalization tool

    # Open and transform images
    permanent_path_to_img = "images_for_test/"
    path_to_files_for_article = "files_for_article/"
    img = Image.open(permanent_path_to_img + "2" + ".jpg")
    img_ten = transform(img)

    plt.imshow(img)
    print("Img size: ", img_ten.size())
    plt.show()

    # k = torch.randn_like(input=img_ten).abs() / (-2)
    k = -0.5

    delta = torch.tanh((img_ten - 0.5) / (k))
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

    print(img_rf.size())
    retina_ganglion = Retina(img_size=img_rf.size(), maximum_in_periphery=110, last_layer=True, collision=1/4)

    output = retina_ganglion(img_rf)
    output = normal(output)
    print(output.size())

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
    torch.manual_seed(111)
    main()
