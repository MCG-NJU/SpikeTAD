from torch import Tensor
import torch
import torch.nn as nn


def heaviside0(x: torch.Tensor):
    return ((x >= 0)).to(x)


def matmul_(x, y):
    """Chunked matrix multiplication to reduce peak memory usage."""
    window = 100
    chunks = []
    for i in range(0, x.shape[-2], window):
        x_chunk = x[..., i : i + window, :]
        out_chunk = x_chunk @ y
        chunks.append(out_chunk)
    out = torch.cat(chunks, dim=-2)
    return out


class TwoSideNeuron(nn.Module):
    """Multi-threshold two-sided spiking neuron for SNN inference."""

    def __init__(self, scale_p=1.0, scale_n=1.0, place=None, times=3, gap=2):
        super(TwoSideNeuron, self).__init__()
        self.scale_p = scale_p
        self.scale_n = scale_n
        self.place = place
        self.times = times
        self.gap = gap
        self.t = 0
        self.neuron = None
        self.spike_count_batch = 0
        self.neuron_count_batch = 0

    def forward(self, x):
        if self.t == 0:
            self.neuron = torch.zeros_like(x)
        self.neuron += x
        v_threshold1 = self.scale_p
        v_threshold2 = self.scale_n
        for i in range(self.times - 1):
            v_threshold1 *= self.gap
            v_threshold2 *= self.gap
        fire_list = []
        acc_fire = torch.zeros_like(x)
        not_fire = torch.ones_like(x)
        for i in range(self.times):
            tmp1 = heaviside0(self.neuron - v_threshold1 + self.scale_p / 2)
            tmp2 = heaviside0(-self.neuron - v_threshold2 + self.scale_n / 2)
            tmp1 *= v_threshold1
            tmp2 *= -v_threshold2
            tmp1 *= not_fire
            tmp2 *= not_fire
            not_fire[tmp1 != 0] = 0
            not_fire[tmp2 != 0] = 0
            self.neuron -= tmp1
            self.neuron -= tmp2
            acc_fire += tmp1 + tmp2
            v_threshold1 /= self.gap
            v_threshold2 /= self.gap
        self.t += 1
        fire_list.append(acc_fire)
        return fire_list

    def reset(self):
        self.t = 0
        self.neuron = None


def get_data_from_place(place, p, n, args):
    """Get multi-threshold neuron parameters based on placement in the network."""
    sp = p
    sn = n
    gap = 2
    if place == "fc2":
        sp = 0.25
        sn = 0.08
    times = 8
    if place in ("q", "k", "v"):
        times = 8
    if place == "s":
        times = 8
        sp = 0.0125
        sn = 0.0125
    return sp, sn, times, gap


def replace_testneuron_by_twosideneuron(model, args):
    """Replace TestNeuron modules with TwoSideNeuron for SNN inference."""
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_testneuron_by_twosideneuron(module, args)
        if "testneuron" == module.__class__.__name__.lower():
            place = model._modules[name].place
            scale_p, scale_n, times, gap = get_data_from_place(
                place,
                float(model._modules[name].scale_p[0]),
                float(model._modules[name].scale_n[0]),
                args,
            )
            print(name, scale_p, scale_n, place)
            model._modules[name] = TwoSideNeuron(
                scale_p=scale_p, scale_n=scale_n, place=place, times=times, gap=gap
            )
    return model


class MyTestPlace(nn.Module):
    """Placeholder module that passes input through, replaced during SNN conversion."""

    def __init__(self, place=None):
        super(MyTestPlace, self).__init__()
        self.place = place

    def forward(self, x, *args):
        return [x]


class TestNeuron(nn.Module):
    """Neuron that accumulates threshold statistics during calibration."""

    def __init__(self, place=None, percent=None):
        super(TestNeuron, self).__init__()
        self.place = place
        self.percent = percent
        self.num = 0
        self.scale_p = torch.nn.Parameter(torch.FloatTensor([0.0]))
        self.scale_n = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, x, times=2, gap=3, show=0, tmptime=0, scaletimes=1):
        x2 = x.reshape(-1)
        threshold = torch.kthvalue(x2, int(self.percent * x2.numel()), dim=0).values.item()
        self.scale_p = torch.nn.Parameter((self.scale_p * self.num + threshold) / (self.num + 1))
        threshold = -torch.kthvalue(x2, int((1 - self.percent) * x2.numel()), dim=0).values.item()
        self.scale_n = torch.nn.Parameter((self.scale_n * self.num + threshold) / (self.num + 1))
        self.num += 1
        return [x]

    def reset(self):
        pass


def replace_test_by_testneuron(model, percent=None):
    """Replace MyTestPlace modules with TestNeuron for threshold calibration."""
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_test_by_testneuron(module, percent)
        if module.__class__.__name__.lower() == "mytestplace":
            model._modules[name] = TestNeuron(place=model._modules[name].place, percent=percent)
    return model


class exp_comp_neuron(nn.Module):
    """Exponential computation neuron for incremental nonlinear function evaluation."""

    def __init__(self, func, *args, **keywords):
        super(exp_comp_neuron, self).__init__(*args, **keywords)
        self.tot = None
        self.func = func
        self.t = 0

    def forward(self, x):
        if self.tot is None:
            last = torch.zeros_like(x)
            self.tot = x.clone()
        else:
            last = self.func(self.tot / self.t) * self.t
            self.tot += x
        self.t += 1
        now = self.func(self.tot / self.t) * self.t
        now = now.half()
        return now - last

    def reset(self):
        self.tot = None
        self.t = 0


def replace_nonlinear_by_neuron(model):
    """Replace Softmax/GELU with exp_comp_neuron for incremental SNN computation.

    Also replaces LayerNorm in specific positions (norm1, norm2, fc_norm).
    """
    name_list = ["norm1", "norm2", "fc_norm"]
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_nonlinear_by_neuron(module)
        if "softmax" in module.__class__.__name__.lower() or "gelu" in module.__class__.__name__.lower():
            model._modules[name] = exp_comp_neuron(func=model._modules[name])
        if "layernorm" in module.__class__.__name__.lower() and (name in name_list):
            model._modules[name] = exp_comp_neuron(func=model._modules[name])
    return model


class MyAt(nn.Module):
    """Matrix multiplication wrapper, replaced by AtNeuron during SNN conversion."""

    def __init__(self):
        super(MyAt, self).__init__()

    def forward(self, x, y):
        return x @ y


class AtNeuron(nn.Module):
    """Incremental matrix multiplication neuron for SNN inference."""

    def __init__(self):
        super(AtNeuron, self).__init__()
        self.tot_a = None
        self.tot_b = None
        self.tot_t = None
        self.t = 0

    def forward(self, x, y):
        if self.t == 0:
            self.tot_a = x
            self.tot_b = y
            self.tot_t = matmul_(x, y)
            self.t = 1
            return self.tot_t
        else:
            self.tot_t += matmul_(x, y) + matmul_(x, self.tot_b) + matmul_(self.tot_a, y)
            self.tot_a += x
            self.tot_b += y
            self.t += 1
            return (
                matmul_(x, self.tot_b) + matmul_(self.tot_a, y) - matmul_(x, y)
            ) / (self.t - 1) - self.tot_t / (self.t * (self.t - 1))

    def reset(self):
        self.tot_a = None
        self.tot_b = None
        self.tot_t = None
        self.t = 0


def replace_at_by_neuron(model):
    """Replace MyAt modules with AtNeuron for incremental SNN inference."""
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_at_by_neuron(module)
        if module.__class__.__name__.lower() == "myat":
            model._modules[name] = AtNeuron()
    return model


def reset_net(model):
    """Reset all neuron states in the model (for SNN inference between samples)."""
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            reset_net(module)
        if "neuron" in module.__class__.__name__.lower():
            module.reset()
    return model


class BaseMonitor:
    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.name_records_index = {}
        self._enable = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()


class SOPMonitor(BaseMonitor):
    """Monitor for synaptic operations (SOPs) and neuron fire rates."""

    def __init__(self, net: nn.Module, type=1):
        super().__init__()
        for name, m in net.named_modules():
            if type == 1 and m.__class__.__name__.lower() == "linear":
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook1(name)))
            if type == 1 and m.__class__.__name__.lower() == "atneuron":
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook1_2(name)))
            if type == 2 and m.__class__.__name__.lower() == "twosideneuron":
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook2(name)))

    def cal_sop1(self, x: Tensor, m: nn.Linear):
        y = torch.zeros_like(x)
        y[x != 0] = 1
        with torch.no_grad():
            out0 = (torch.nn.functional.linear(y, torch.ones_like(m.weight), torch.zeros_like(m.bias))).sum()
            sum0 = (
                torch.nn.functional.linear(torch.ones_like(y), torch.ones_like(m.weight), torch.zeros_like(m.bias))
            ).sum()
            return out0, sum0

    def create_hook1(self, name):
        def hook1(m: nn.Linear, x: Tensor, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop1(x[0], m))

        return hook1

    def cal_sop1_2(self, A: Tensor, B: Tensor, m: AtNeuron):
        tmp_A = torch.ones_like(A)
        tmp_B = torch.ones_like(B)
        sum0 = (tmp_A @ tmp_B).sum()
        tmp_A = torch.zeros_like(A)
        tmp_B = torch.zeros_like(B)
        tmp_A[A != 0] = 1
        tmp_B[B != 0] = 1
        tmp_As = torch.zeros_like(m.tot_a)
        tmp_As[m.tot_a != 0] = 1
        tmp_Bs = torch.zeros_like(m.tot_b)
        tmp_Bs[m.tot_b != 0] = 1
        out01 = (tmp_A @ tmp_B).sum()
        out02 = (tmp_A @ tmp_Bs).sum()
        out03 = (tmp_As @ tmp_B).sum()
        out0 = out01 + out02 + out03
        return out0, sum0

    def create_hook1_2(self, name):
        def hook1_2(m: AtNeuron, x, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop1_2(x[0], x[1], m))

        return hook1_2

    def cal_sop2(self, x):
        num_elements = x[0].numel()
        tmp = []
        for index, i in enumerate(x):
            tmp.append((i != 0).sum())
        return sum(tmp), num_elements, tmp

    def create_hook2(self, name):
        def hook2(m: TwoSideNeuron, x: Tensor, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop2(y))

        return hook2
