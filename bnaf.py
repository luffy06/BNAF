import torch
import math
import numpy as np
from para_search import *

def print_tensor(x, name=None):
    assert len(x.shape) == 2
    if name != None:
        print(name, '(%d, %d)' % (x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        # print('%d\t' % i, end='\t')
        for j in range(x.shape[1]):
            print('%.32f' % x[i][j], end='\t' if j != x.shape[1] - 1 else '\n')

print_detail=False

def act(x, limited_exp: bool = False):
    if limited_exp:
        mask = (x > 0).int()
        return mask * (2 - torch.exp(-x)) + (1 - mask) * torch.exp(x)
    return torch.exp(x)


class Encoder(torch.nn.Module):
    def __init__(self, dim, shifts):
        super(Encoder, self).__init__()
        self.type = type
        self.dim = dim
        self.shifts = shifts

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        if self.dim == 2:
            h1 = torch.floor(outputs)
            h2 = outputs - torch.floor(outputs)
            outputs = torch.cat((h1, h2), 1)
        elif self.dim == 4:
            h1 = outputs
            h2 = torch.floor(outputs)
            outputs = (outputs - h2) * self.shifts
            h3 = torch.floor(outputs)
            h4 = outputs - h3
            outputs = torch.cat((h1, h2, h3, h4), 1)
        if self.training:
            return outputs, 0
        else:
            if print_detail:
                print_tensor(outputs, 'Encoder')
            return outputs

    def save_weights(self, f):
        pass

    def derivative(self, w):
        return w

    def __repr__(self):
        return 'Encoder(dim={}, shifts={})'.format(self.dim, self.shifts)


class Decoder(torch.nn.Module):
    def __init__(self, dim, type, trim):
        super(Decoder, self).__init__()
        self.type = type
        if self.type == 'weighted_sum':
            self.w = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(dim, 1)))

    def forward(self, inputs: torch.Tensor):
        if self.type == 'trim':
            outputs = inputs
        elif self.type == 'sum':
            outputs = inputs.sum(-1).unsqueeze(1)
        elif self.type == 'weighted_sum':
            w = act(self.w)
            outputs = inputs.matmul(w)
        else:
            print('Wrong Decoder Type {}'.format(self.type))
            exit()
        if self.training:
            return outputs, 0
        else:
            if print_detail:
                if self.type == 'weighted_sum':
                    print_tensor(w, 'Weight in Decoder')
                print_tensor(outputs, 'Decoder')
            return outputs

    def save_weights(self, f):
        if self.type == 'weighted_sum':
            w = act(self.w)
            f.write('%d\t%d\n' % (w.shape[0], w.shape[1]))
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    f.write('%.16f\t' % (w[i, j].item()))
                f.write('\n')

    def derivative(self, w):
        if self.type == 'weighted_sum':
            return w.matmul(act(self.w)) if w != None else act(self.w)
        return w

    def __repr__(self):
        return 'Decoder(type={})'.format(self.type)


class Sequential(torch.nn.Sequential):
    """
    Class that extends ``torch.nn.Sequential`` for computing the output of
    the function alongside with the log-det-Jacobian of such transformation.
    """

    def forward(self, inputs: torch.Tensor):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The output tensor and the log-det-Jacobian of this transformation.
        """

        log_det_jacobian = 0.0
        if self.training:
            for i, module in enumerate(self._modules.values()):
                inputs, log_det_jacobian_ = module(inputs)
                log_det_jacobian = log_det_jacobian + log_det_jacobian_
            return inputs, log_det_jacobian
        else:
            for i, module in enumerate(self._modules.values()):
                inputs = module(inputs)
            return inputs

    def save_weights(self, f):
        w = None
        for i, module in enumerate(self._modules.values()):
            module.save_weights(f)
            w = module.derivative(w)
        # print('Derivative:', w)

class BNAF(torch.nn.Sequential):
    """
    Class that extends ``torch.nn.Sequential`` for constructing a Block Neural
    Normalizing Flow.
    """

    def __init__(self, *args, res: str = None):
        """
        Parameters
        ----------
        *args : ``Iterable[torch.nn.Module]``, required.
            The modules to use.
        res : ``str``, optional (default = None).
            Which kind of residual connection to use. ``res = None`` is no residual
            connection, ``res = 'normal'`` is ``x + f(x)`` and ``res = 'gated'`` is
            ``a * x + (1 - a) * f(x)`` where ``a`` is a learnable parameter.
        """

        super(BNAF, self).__init__(*args)

        self.res = res

        if res == "gated":
            self.gate = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(1)))

    def forward(self, inputs: torch.Tensor):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The output tensor and the log-det-Jacobian of this transformation.
        """

        outputs = inputs
        grad = None

        if self.training:
            for module in self._modules.values():
                outputs, grad = module(outputs, grad)

                grad = grad if len(grad.shape) == 4 else grad.view(grad.shape + [1, 1])

            assert inputs.shape[-1] == outputs.shape[-1]

            if self.res == "normal":
                return inputs + outputs, torch.nn.functional.softplus(grad.squeeze()).sum(-1)
            elif self.res == "gated":
                return self.gate.sigmoid() * outputs + (1 - self.gate.sigmoid()) * inputs, (
                    torch.nn.functional.softplus(grad.squeeze() + self.gate)
                    - torch.nn.functional.softplus(self.gate)
                ).sum(-1)
            else:
                return outputs, grad.squeeze().sum(-1)
        else:
            for module in self._modules.values():
                outputs = module(outputs)

            assert inputs.shape[-1] == outputs.shape[-1]

            if self.res == "normal":
                return inputs + outputs
            elif self.res == "gated":
                return self.gate.sigmoid() * outputs + (1 - self.gate.sigmoid()) * inputs
            else:
                return outputs

    def derivative(self, w):
        for i, module in enumerate(self._modules.values()):
            w = module.derivative(w)
        return w

    def save_weights(self, f):
        for module in self._modules.values():
            module.save_weights(f)

    def _get_name(self):
        return "BNAF(res={})".format(self.res)


class Permutation(torch.nn.Module):
    """
    Module that outputs a permutation of its input.
    """

    def __init__(self, in_features: int, p: list = None):
        """
        Parameters
        ----------
        in_features : ``int``, required.
            The number of input features.
        p : ``list`` or ``str``, optional (default = None)
            The list of indeces that indicate the permutation. When ``p`` is not a
            list, if ``p = 'flip'``the tensor is reversed, if ``p = None`` a random
            permutation is applied.
        """

        super(Permutation, self).__init__()

        self.in_features = in_features

        if p is None:
            self.p = np.random.permutation(in_features)
        elif p == "flip":
            self.p = list(reversed(range(in_features)))
        else:
            self.p = p

    def forward(self, inputs: torch.Tensor):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The permuted tensor and the log-det-Jacobian of this permutation.
        """
        if self.training:
            return inputs[:, self.p], 0
        else:
            return inputs[:, self.p]

    def save_weights(self, f):
        pass
    
    def derivative(self, w):
        return w

    def __repr__(self):
        return "Permutation(in_features={}, p={})".format(self.in_features, self.p)


class MaskedWeight(torch.nn.Module):
    """
    Module that implements a linear layer with block matrices with positive diagonal blocks.
    Moreover, it uses Weight Normalization (https://arxiv.org/abs/1602.07868) for stability.
    """

    def __init__(
        self, in_features: int, out_features: int, dim: int, bias: bool = True, 
        all_pos: bool = False, limited_exp: bool = False
    ):
        """
        Parameters
        ----------
        in_features : ``int``, required.
            The number of input features per each dimension ``dim``.
        out_features : ``int``, required.
            The number of output features per each dimension ``dim``.
        dim : ``int``, required.
            The number of dimensions of the input of the flow.
        bias : ``bool``, optional (default = True).
            Whether to add a parametrizable bias.
        """

        super(MaskedWeight, self).__init__()
        self.in_features, self.out_features, self.dim = in_features, out_features, dim
        self.all_pos = all_pos
        self.limited_exp = limited_exp

        weight = torch.zeros(out_features, in_features)
        for i in range(dim):
            weight[
                i * out_features // dim : (i + 1) * out_features // dim,
                0 : (i + 1) * in_features // dim,
            ] = torch.nn.init.xavier_uniform_(
                torch.Tensor(out_features // dim, (i + 1) * in_features // dim)
            )

        self._weight = torch.nn.Parameter(weight)
        self._diag_weight = torch.nn.Parameter(
            torch.nn.init.uniform_(torch.Tensor(out_features, 1)).log()
        )

        self.bias = (
            torch.nn.Parameter(
                torch.nn.init.uniform_(
                    torch.Tensor(out_features),
                    -1 / math.sqrt(out_features),
                    1 / math.sqrt(out_features),
                )
            )
            if bias
            else 0
        )

        mask_d = torch.zeros_like(weight)
        for i in range(dim):
            mask_d[
                i * (out_features // dim) : (i + 1) * (out_features // dim),
                i * (in_features // dim) : (i + 1) * (in_features // dim),
            ] = 1

        self.register_buffer("mask_d", mask_d)

        mask_o = torch.ones_like(weight)
        for i in range(dim):
            mask_o[
                i * (out_features // dim) : (i + 1) * (out_features // dim),
                i * (in_features // dim) :,
            ] = 0

        self.register_buffer("mask_o", mask_o)

    def get_weights(self):
        """
        Computes the weight matrix using masks and weight normalization.
        It also compute the log diagonal blocks of it.
        """
        if self.training:
            w = act(self._weight, self.limited_exp) * self.mask_d \
                + (act(self._weight, self.limited_exp) if self.all_pos else self._weight) * self.mask_o

            w_squared_norm = (w ** 2).sum(-1, keepdim=True)

            w = self._diag_weight.exp() * w / w_squared_norm.sqrt()

            wpl = self._diag_weight + self._weight - 0.5 * torch.log(w_squared_norm)

            return w.t(), wpl.t()[self.mask_d.bool().t()].view(
                self.dim, self.in_features // self.dim, self.out_features // self.dim)
        else:
            w = act(self._weight, self.limited_exp) * self.mask_d \
                + (act(self._weight, self.limited_exp) if self.all_pos else self._weight) * self.mask_o

            w_squared_norm = (w ** 2).sum(-1, keepdim=True)

            w = self._diag_weight.exp() * w / w_squared_norm.sqrt()

            return w.t()

    def forward(self, inputs, grad: torch.Tensor = None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal block of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous
        transformations combined with this transformation.
        """
        if self.training:
            w, wpl = self.get_weights()

            g = wpl.transpose(-2, -1).unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)

            return inputs.matmul(w) + self.bias, torch.logsumexp(
                    g.unsqueeze(-2) + grad.transpose(-2, -1).unsqueeze(-3), -1) if grad is not None else g
        else:
            w = self.get_weights()
            if print_detail:
                print_tensor(w, 'Weight in MaskedWeight Layer')
                print_tensor(inputs.matmul(w) + self.bias, 'MaskedWeight Layer')
            return inputs.matmul(w) + self.bias

    def derivative(self, w):
        ww = self.get_weights()
        return w.matmul(ww) if w != None else ww

    def save_weights(self, f):
        w = self.get_weights()
        f.write('%d\t%d\n' % (w.shape[0], w.shape[1]))
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                f.write('%.16f\t' % (w[i, j].item()))
            f.write('\n')

    def __repr__(self):
        return "MaskedWeight(in_features={}, out_features={}, dim={}, bias={}, positive_weights={}, limited_exp={})".format(
            self.in_features,
            self.out_features,
            self.dim,
            not isinstance(self.bias, int),
            self.all_pos,
            self.limited_exp,
        )


class Tanh(torch.nn.Tanh):
    """
    Class that extends ``torch.nn.Tanh`` additionally computing the log diagonal
    blocks of the Jacobian.
    """

    def forward(self, inputs, grad: torch.Tensor = None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal blocks of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous
        transformations combined with this transformation.
        """

        if self.training:
            g = -2 * (inputs - math.log(2) + torch.nn.functional.softplus(-2 * inputs))
            return torch.tanh(inputs), (g.view(grad.shape) + grad) if grad is not None else g
        else:
            if print_detail:
                print_tensor(torch.tanh(inputs), 'Tanh')
            return torch.tanh(inputs)

    def derivative(self, w):
        return w

    def save_weights(self, f):
        pass


class ReLU(torch.nn.ReLU):
    """
    Class that extends ``torch.nn.Tanh`` additionally computing the log diagonal
    blocks of the Jacobian.
    """

    def forward(self, inputs, grad: torch.Tensor = None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal blocks of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous
        transformations combined with this transformation.
        """

        if self.training:
            g = (inputs > 0).double()
            return torch.nn.functional.relu(inputs), (g.view(grad.shape) + grad) if grad is not None else g
        else:
            if print_detail:
                print_tensor(torch.nn.functional.relu(inputs), 'ReLU')
            return torch.nn.functional.relu(inputs)

    def derivative(self, w):
        return w

    def save_weights(self, f):
        pass
