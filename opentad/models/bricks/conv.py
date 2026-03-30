import copy
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from ..builder import MODELS

from spikingjelly.clock_driven import functional, surrogate, layer
from spikingjelly.activation_based import neuron
from torch.nn.parameter import Parameter
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode


def get_all_spike_stats(model):
    spike_total = 0
    neuron_total = 0
    for m in model.modules():
        if isinstance(m, IF):
            spike_total += m.spike_count_batch
            neuron_total += m.neuron_count_batch
    return spike_total, neuron_total


class SpikingNorm(nn.Module):
    def __init__(self, momentum=0.1, scale=True, sigmoid=True, eps=1e-6):
        super(SpikingNorm, self).__init__()
        self.sigmoid = sigmoid
        self.eps = eps
        self.lock_max = False
        if scale:
            self.scale = Parameter(torch.Tensor([1.0]))
        else:
            self.register_buffer("scale", torch.ones(1))
        if self.sigmoid:
            self.scale.data *= 10.0
        self.momentum = momentum
        self.register_buffer("running_max", torch.ones(1))

    def calc_scale(self):
        if self.sigmoid:
            scale = torch.sigmoid(self.scale)
        else:
            scale = torch.abs(self.scale)
        return scale

    def calc_v_th(self):
        return self.running_max * self.calc_scale() + self.eps

    def forward(self, x):
        if self.training and (not self.lock_max):
            self.running_max = (
                (1 - self.momentum) * self.running_max + self.momentum * torch.max(F.relu(x)).item()
            )
        x = torch.clamp(F.relu(x) / self.calc_v_th(), min=0.0, max=1.0)
        return x

    def extra_repr(self):
        if self.sigmoid:
            return "v_th={}, scale={}, running_max={}".format(
                self.calc_v_th(), torch.sigmoid(self.scale.data), self.running_max.data
            )
        else:
            return "v_th={}, scale={}, running_max={}".format(
                self.calc_v_th(), torch.abs(self.scale.data), self.running_max.data
            )

    def extract_running_max(self):
        x = self.running_max.data
        self.running_max.data = self.running_max.data / x
        return x

    def extract_scale(self):
        x = self.scale.data
        self.running_max.data = self.running_max.data * x
        self.scale.data = torch.Tensor([1.0]).to(self.scale.device)
        return x


decay = 0.25


class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        self.act = act
        self.qtrick = MultiSpike16()

    def forward(self, x):
        x = x.unsqueeze(0)
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i]
            else:
                mem = x[i]
            spike = self.qtrick(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output.squeeze(0)


class MultiSpike4(nn.Module):
    class quant4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)


class MultiSpike8(nn.Module):
    class quant8(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=8))

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 8] = 0
            return grad_input

    def forward(self, x):
        return self.quant8.apply(x)


class MultiSpike16(nn.Module):
    class quant16(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=16))

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 16] = 0
            return grad_input

    def forward(self, x):
        return self.quant16.apply(x)


class MultiSpike2(nn.Module):
    class quant2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=2))

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 2] = 0
            return grad_input

    def forward(self, x):
        return self.quant2.apply(x)


class MultiSpike1(nn.Module):
    class quant1(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=1))

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 1] = 0
            return grad_input

    def forward(self, x):
        return self.quant1.apply(x)


class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()


class RemoveTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.mean(0).squeeze(0)


class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0] / self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)


class RepeatTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        x_seq = x_seq.unsqueeze(0).repeat(self.T, 1, 1, 1)
        return x_seq


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


myfloor = GradFloor.apply


class IF(nn.Module):
    """Integrate-and-Fire spiking neuron.

    When T > 0, operates in SNN mode with T timesteps.
    When T == 0, operates in ANN quantization mode.

    T is set externally via set_model_timesteps() in the test script,
    so there is no need to manually modify this default value.
    """

    def __init__(self, T=0, L=2, thresh=2.0, tau=1.0, gama=1.0):
        super(IF, self).__init__()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.expand = RepeatTemporalDim(T)
        self.merge = RemoveTemporalDim(T)
        self.L = L
        self.T = T
        self.loss = 0
        self.spike_count_batch = 0
        self.neuron_count_batch = 0

    def forward(self, x):
        neuron_count = 0
        spike_count = 0
        if self.T > 0:
            thre = self.thresh.data
            x = self.expand(x)
            mem = 0.5 * thre
            spike_pot = []
            for t in range(self.T):
                mem = mem + x[t, ...]
                spike = torch.heaviside(mem - thre, torch.tensor(0.0))
                self.spike_count_batch += spike.sum().item()
                self.neuron_count_batch += spike.numel()
                spike = spike * thre
                mem = mem - spike
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
        else:
            x = x / self.thresh
            x = torch.clamp(x, 0, 1)
            x = myfloor(x * self.L + 0.5) / self.L
            x = x * self.thresh
        return x


def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1)
    return x


@MODELS.register_module()
class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.with_norm = norm_cfg is not None

        conv_cfg_base = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if self.with_norm:
            conv_cfg_base["bias"] = False

        assert conv_cfg is None or isinstance(conv_cfg, dict)
        if conv_cfg is not None:
            conv_cfg_base.update(conv_cfg)

        self.conv = nn.Conv1d(**conv_cfg_base)

        if self.with_norm:
            norm_cfg = copy.copy(norm_cfg)
            norm_type = norm_cfg["type"]
            norm_cfg.pop("type")
            self.norm_type = norm_type

            if norm_type == "BN":
                self.norm = nn.BatchNorm1d(num_features=out_channels, **norm_cfg)
            elif norm_type == "GN":
                self.norm = nn.GroupNorm(num_channels=out_channels, **norm_cfg)
            elif norm_type == "LN":
                self.norm = nn.LayerNorm(out_channels)

        assert act_cfg is None or isinstance(act_cfg, dict)
        self.with_act = act_cfg is not None

        if self.with_act:
            act_cfg = copy.copy(act_cfg)
            act_type = act_cfg["type"]
            act_cfg.pop("type")

            if act_type == "relu":
                self.act = IF()
            else:
                self.act = eval(act_type)(**act_cfg)

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask=None):
        x = self.conv(x)
        if mask is not None:
            if mask.shape[-1] != x.shape[-1]:
                mask = (
                    F.interpolate(mask.unsqueeze(1).to(x.dtype), size=x.size(-1), mode="nearest")
                    .squeeze(1)
                    .to(mask.dtype)
                )
            x = x * mask.unsqueeze(1).float().detach()

        if self.with_norm:
            if self.norm_type == "LN":
                x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.norm(x)

        if self.with_act:
            x = self.act(x)

        if mask is not None:
            x = x * mask.unsqueeze(1).float().detach()
            return x, mask
        else:
            return x
