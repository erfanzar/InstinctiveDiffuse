import typing

from .modules import Conv, TConv, PixelNorm
from torch import nn
from erutils import fprint, HyperParameters
import torch
from erutils.lightning import pars_model_v2


class GeneratorVPC(nn.Module):
    def __init__(self, ic: int = 256, oc: int = 3, cc: int = 512, j: int = 64, debug: bool = False):
        super(GeneratorVPC, self).__init__()
        self.m = nn.ModuleList()
        self.ic = ic
        self.debug = debug
        assert (j * 2) % ic != 0, 'j times 2 should not be equal or less than ic'
        assert cc % j == 0, 'Assert Failed'
        # for isa in range(1, cc + 1):
        #     if cc % isa == 0:
        #         fprint(isa)
        c = [f for f in range(ic, cc + j, j)][::-1] + [ic - j, ic - (j * 2), j]
        self.channels = c
        self.channel_len = len(c)
        if self.debug:
            fprint('Model Created With ', *c, ' Channels Len :', len(c))
        for i, cl in enumerate(c):
            c1, c2, k, s, p = ic if i == 0 else c[i - 1], cl, 3 if i == 0 else 4, 1 if i == 0 else 2, 0 if i == 0 else 1

            self.m.append(nn.Sequential(
                TConv(c1=c1, c2=c2, k=k, s=s, p=p, use_bn=True),
                Conv(c2, c2, 3, 1, 1, act='nn.Identity()'),
            ) if i != len(c) - 1 else nn.Sequential(
                TConv(c1=c1, c2=oc, k=k, s=s, p=p, use_bn=False, act='nn.Identity()'),
                nn.Tanh()
            ))

        fprint(f'Created With {sum(v.numel() for v in self.parameters()) / 1e6} M Parameters ')

    @classmethod
    def init_weight(cls, m):
        if isinstance(m, Conv):
            torch.nn.init.xavier_uniform_(m.c.weight)

    def configure_optimizer(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (Conv, TConv)
        blacklist_weight_modules = (nn.Linear,)
        for index_a, (nm1, p1) in enumerate(self.named_modules()):
            if self.debug:
                st = '|| {:>5} {:>35} {:>25} ||'.format(index_a, '%s' % nm1, sum(p.numel() for p in p1.parameters()))
                fprint(st)
                print('-' * len(st))
            for nm2, p2 in p1.named_parameters():
                fpn = '%s.%s' % (nm1, nm2) if nm1 else nm2

                if nm2.endswith('bias'):
                    no_decay.add(fpn)
                elif nm2.endswith('weight') and isinstance(p1, whitelist_weight_modules):
                    decay.add(fpn)
                elif nm2.endswith('weight') and isinstance(p1, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 2e-1},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=4e-4)
        return optimizer

    def forward(self, inputs, ):

        for m in self.m:
            inputs = m(inputs)
        return inputs


class DiscriminatorVPC(nn.Module):
    def __init__(self, ic: int = 3, cc: int = 512, j: int = 64, debug: bool = False):
        super(DiscriminatorVPC, self).__init__()
        self.m = nn.ModuleList()
        oc: int = 10
        self.ic = ic
        self.debug = debug

        assert cc % j == 0, 'Assert Failed'
        # for isa in range(1, cc + 1):
        #     if cc % isa == 0:
        #         fprint(isa)
        c = [f for f in range(0, cc + j, j)][1:]
        self.channel_len = len(c)
        if self.debug:
            fprint('Model Created With ', *c, ' Channels Len :', len(c))
        for i, cl in enumerate(c):
            c1, c2, k, s, p = ic if i == 0 else c[i - 1], cl, 3 if i == 0 else 4, 1 if i == 0 else 2, 0 if i == 0 else 1
            s = nn.Sequential(
                Conv(c1, c2, 3, 2, 1, use_bn=True),
                Conv(c2, c2, 3, 1, 1)
            ) if i != len(c) - 1 else nn.Sequential(
                Conv(c1, j, 1, 1, 0, use_bn=False, act='nn.Identity()'),
                nn.Flatten(),
                nn.Linear(j * 9, oc),
                nn.Sigmoid()
            )
            self.m.append(s)

        fprint(f'Created With {sum(v.numel() for v in self.parameters()) / 1e6} M Parameters ')

    @classmethod
    def init_weight(cls, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    def configure_optimizer(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (Conv, TConv)
        blacklist_weight_modules = (nn.Linear,)
        for index_a, (nm1, p1) in enumerate(self.named_modules()):
            if self.debug:
                st = '|| {:>5} {:>35} {:>25} ||'.format(index_a, '%s' % nm1, sum(p.numel() for p in p1.parameters()))
                fprint(st)
                print('-' * len(st))
            for nm2, p2 in p1.named_parameters():
                fpn = '%s.%s' % (nm1, nm2) if nm1 else nm2

                if nm2.endswith('bias'):
                    no_decay.add(fpn)
                elif nm2.endswith('weight') and isinstance(p1, whitelist_weight_modules):
                    decay.add(fpn)
                elif nm2.endswith('weight') and isinstance(p1, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 2e-1},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=4e-4)
        return optimizer

    def forward(self, inputs):
        b, _, _, _ = inputs.shape

        for m in self.m:
            inputs = m(inputs)

        inputs = inputs.view(b, -1)
        return inputs


class ModelConfig(nn.Module):
    def __init__(self, config):
        """
            config must contain
        config.model, config.c_req, config.detail, config.print_status, config.sc,config.imports
        :param config:
        :param config:
        """

        super(ModelConfig, self).__init__()
        self.m, self.s = pars_model_v2(cfg=config.model, c_req=config.c_req, detail=config.detail,
                                       print_status=config.print_status, sc=config.sc,
                                       imports=config.imports, )
        self.s: tuple[str] = tuple(str(self.s))

    def forward(self, x: torch.Tensor):

        save = []
        for i, m in enumerate(self.m):
            if m.f != -1:
                x = save[m.f % i]
            x = m(x)
            i = str(i)
            save.append(x if i in self.s else None)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
