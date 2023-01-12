import typing

import torch
import torch.nn as nn
from erutils.lightning import M

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Conv(M):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int = None, g: int = 1,
                 activation: [str, torch.nn] = None, using_bn: bool = False, batch_first: bool = True,
                 form: int = -1):
        super(Conv, self).__init__()
        self.batch_first = batch_first
        self.form = form
        self.to(DEVICE)
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, bias=False,
                              padding=p if p is not None else 0, groups=g).to(DEVICE)
        nn.init.xavier_normal_(self.conv.weight.data)

        self.activation = (
            eval(activation) if isinstance(activation, str) else activation
        ) if activation is not None else nn.SiLU()

        self.batch_norm = nn.BatchNorm2d(c2) if using_bn is True else nn.Identity()

    def forward(self, x) -> torch.Tensor:
        x = self.batch_norm(self.activation(self.conv(x))) if not self.batch_first else self.activation(
            self.batch_norm(self.conv(x)))
        return x


class TConv(M):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, p: int = 0,
                 activation: [str, torch.nn] = None, using_bn: bool = False, batch_first: bool = True,
                 form: int = -1):
        super(TConv, self).__init__()

        self.batch_first = batch_first
        self.form = form
        self.to(DEVICE)
        self.conv = nn.ConvTranspose2d(c1, c2, kernel_size=k, stride=s, groups=g, bias=False, padding=p).to(DEVICE)
        nn.init.xavier_normal_(self.conv.weight.data)

        self.activation = (
            eval(activation) if isinstance(activation, str) else activation
        ) if activation is not None else nn.SiLU()

        self.batch_norm = nn.BatchNorm2d(c2) if using_bn is True else nn.Identity()

    def forward(self, x) -> torch.Tensor:
        x = self.batch_norm(self.activation(self.conv(x))) if not self.batch_first else self.activation(
            self.batch_norm(self.conv(x)))
        return x


class Concat(M):
    def __init__(self, dim, form):
        super(Concat, self).__init__()
        self.form = form
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)


class Neck(M):
    def __init__(self, c1, c2, e=0.5, shortcut=False, form: int = -1):
        super(Neck, self).__init__()
        c_ = int(c2 * e)
        self.form = form
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c_, c2, k=3, s=1, p=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        ck = self.cv2(self.cv1(x[self.form]))

        k = x + ck if self.add else ck

        return k


class C3(M):
    def __init__(self, c1, c2, e=0.5, n=1, shortcut=True, form: int = -1):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.form = form
        self.cv1 = Conv(c1, c_, k=3, s=1, p=1)
        self.cv2 = Conv(c1, c_, k=3, s=1, p=1)
        self.cv3 = Conv(c_ * 2, c2, k=3, p=1)
        self.m = nn.Sequential(*(Neck(c_, c_, shortcut=shortcut, e=0.5) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv2(x)), self.cv1(x)), dim=1))


class C4P(C3):
    def __init__(self, c, e=0.5, n=1, ct=2, form: int = -1):
        super(C4P, self).__init__(c1=c, c2=c, e=e, n=n)
        self.form = form
        self.ct = ct

    def forward(self, x):
        for _ in range(self.ct):
            x = self.cv3(torch.cat((self.m(self.cv2(x)), self.cv1(x)), dim=1)) + x
        return x


class RepConv(M):
    def __init__(self, c, e=0.5, n=3, form: int = -1):
        super(RepConv, self).__init__()
        c_ = int(c * e)
        self.form = form
        self.layer = nn.ModuleList()
        # self.layer.append(
        #     *(Conv(c1=c if i == 0 else c_, c2=c_ if i == 0 else c, kernel_size=3, padding=1, stride=1, batch=False)
        #       for i in range(n)))
        for i in range(n):
            self.layer.append(
                Conv(c1=c if i == 0 else c_, c2=c_ if i == 0 else c, k=3, p=1, s=1))

    def forward(self, x):
        x_ = x
        for layer in self.layer:
            x = layer.forward(x)
        return x_ + x


class ConvSc(RepConv):
    def __init__(self, c, n=4, form: int = -1):
        super(ConvSc, self).__init__(c=c, e=1, n=n)
        self.form = form

    def forward(self, x):
        x_ = x.detach().clone()
        for layer in self.layer:
            x = layer(x) + x
        return x + x_


class ResidualBlock(M):
    def __init__(self, c1, n: int = 4, use_residual: bool = True, form: int = -1):
        super(ResidualBlock, self).__init__()
        self.use_residual = use_residual
        self.n = n
        self.to(DEVICE)
        self.layer = nn.ModuleList()
        self.form = form

        for _ in range(n):
            self.layer.append(
                nn.Sequential(
                    Conv(c1, c1 * 2, s=1, p=0, k=1),
                    Conv(c1 * 2, c1, s=1, p=1, k=3)
                )
            )

    def forward(self, x) -> torch.Tensor:
        c = x
        for layer in self.layer:
            x = layer(x)
        return x + c if self.use_residual else x


class CV1(M):
    def __init__(self, c1, c2, e=0.5, n=1, shortcut=False, dim=-3, form: int = -1):
        super(CV1, self).__init__()
        c_ = int(c2 * e)
        if shortcut:
            c2 = c1
        self.c = Conv(c1, c_, k=3, p=1, s=1)
        self.form = form
        self.v = Conv(c1, c_, k=3, p=1, s=1)
        self.m = nn.Sequential(
            *(Conv(c_ * 2 if i == 0 else c2, c2, k=3, s=1, p=1) for i in range(n)))
        self.sh = c1 == c2
        self.dim = dim

    def forward(self, x):
        c = torch.cat((self.c(x), self.v(x)), dim=self.dim)
        return self.m(c) if not self.sh else self.m(
            torch.cat((self.c(x), self.v(x)), dim=self.dim)) + x


class UC1(M):
    def __init__(self, c1, c2, e=0.5, dim=-3, form: int = -1):
        super(UC1, self).__init__()
        self.form = form
        c_ = int(c2 * e)
        self.c = Conv(c1=c1, c2=c_, k=1, s=1)
        self.v = Conv(c1=c1, c2=c_, k=1, s=1)
        self.m = Conv(c1=c_, c2=c2, k=1, s=1)
        self.dim = dim

    def forward(self, x):
        return self.m(torch.cat((self.c(x), self.v(x)), dim=self.dim))


class MP(M):
    def __init__(self, k=2, form: int = -1):
        super(MP, self).__init__()
        self.form = form
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        x = self.m(x)
        return x


class SP(M):
    def __init__(self, k=3, s=1, form: int = -1):
        super(SP, self).__init__()
        self.form = form
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        x = self.m(x)
        return x


class LP(M):
    def __init__(self, dim: int = None):
        super(LP, self).__init__()
        self.dim = dim

    def forward(self, l1, l2, dim_f: int = 1):
        return torch.cat((l1, l2), dim=dim_f if self.dim is None else self.dim)


class UpSample(M):
    def __init__(self, s: int = 2, m: str = 'nearest', form: int = -1):
        super(UpSample, self).__init__()
        self.form = form
        self.u = nn.Upsample(scale_factor=s, mode=m)

    def forward(self, x):
        x = self.u(x)
        return x


# class SPP(M):
#     def __init__(self):
#         super(SPP, self).__init__()
#         self.n = SPPCSPC(4, 4, False )
#


class SPPCSPC(M):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        x = self.cv7(torch.cat((y1, y2), dim=1))
        return x


class GeneratorBlock(M):
    def __init__(self, in_size, out_size, k: int = 3, s: int = 1, p: int = 0, fl: bool = False,
                 act: typing.Union[str] = None):
        super(GeneratorBlock, self).__init__()
        self.act = act
        self.s = s
        self.fl = fl
        self.in_size = in_size
        self.out_size = out_size
        self.k = k

        self.m = nn.Sequential(
            TConv(in_size, out_size, k=k, s=s, activation=act, using_bn=True, p=p),
        ) if not fl else nn.Sequential(
            TConv(in_size, out_size, p=p, k=k, s=s, activation=act, using_bn=False),
        )

    def forward(self, x):
        return self.m(x)


class DiscriminatorBlock(M):
    def __init__(self, in_size, out_size, k: int = 3, s: int = 1, p: int = 0, fl: bool = False,
                 act: typing.Union[str] = None):
        super(DiscriminatorBlock, self).__init__()
        self.act = act
        self.s = s
        self.fl = fl
        self.in_size = in_size
        self.out_size = out_size
        self.k = k

        self.m = nn.Sequential(
            Conv(in_size, out_size, k=k, s=s, activation=act, using_bn=True, batch_first=True, p=p),
        ) if not fl else nn.Sequential(
            Conv(in_size, out_size, k=k, s=s, activation=act, using_bn=False, batch_first=True, p=p),
        )

    def forward(self, x):
        return self.m(x)
