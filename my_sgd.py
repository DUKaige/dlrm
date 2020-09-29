import torch
from torch.optim.optimizer import Optimizer, required

acc_max = 2 ** 8

class MySGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
    """
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        self.multiplier = 0.000001
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(MySGD, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sparse'] = False
                state['low_prec_para'] = torch.clone(p)
                state['sum'] = torch.full_like(p, 0, memory_format=torch.preserve_format)
                state['low_prec'] = torch.clamp(torch.floor(state['sum'] / self.multiplier), 0, acc_max - 1)

    def __setstate__(self, state):
        super(MySGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    param_state = self.state[p]
                    param_state['sparse'] = True
                    d_p = p.grad
                    d_p_low_prec = p.grad
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)
                    if momentum != 0:
                        if 'sum' not in param_state:
                            cl = torch.clone(d_p).detach()
                            param_state['sum'] += cl
                            param_state['low_prec'] += cl / self.multiplier
                        else:
                            param_state['sum'].mul_(momentum).add_(d_p, alpha=1 - dampening)
                            param_state['low_prec'].mul_(momentum).add_(d_p/self.multiplier, alpha=1 - dampening)
                            param_state['low_prec'] = torch.floor(param_state['low_prec'])
                            torch.clamp(param_state['low_prec'], 0, acc_max - 1)

                        if nesterov:
                            d_p = d_p.add(param_state['sum'], alpha=momentum)
                            d_p_low_prec = d_p_low_prec.add(['low_prec'], alpha=momentum)
                        else:
                            d_p = param_state['sum']
                            d_p_low_prec = param_state['low_prec']
                    p.add_(d_p, alpha=-group['lr'])
                    param_state['low_prec_para'].add_(d_p_low_prec, alpha=-group['lr'])

                else:
                    d_p = p.grad
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'sum' not in param_state:
                            buf = param_state['sum'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['sum']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf

                    p.add_(d_p, alpha=-group['lr'])

        return loss