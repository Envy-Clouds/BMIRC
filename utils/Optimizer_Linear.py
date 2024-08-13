import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from scipy.special import binom

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def Binom(gamma,accumulated_step):
    accumulated_step_list = list(range(1, accumulated_step + 1))
    c_value = []
    for i in accumulated_step_list:
        c = binom(gamma, i)
        c_value.append(c)
    return c_value

def Getgamma(max_gamma, min_gamma, current_step, last_step):
    gamma = max_gamma-((current_step-1)/(last_step-1)) * (max_gamma-min_gamma)
    return gamma

class Adagrad(Optimizer):
    """Implements Adagrad algorithm.

    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(self, params, lr=required, lr_decay=0, weight_decay=0, initial_accumulator_value=0,
                 gammas=None, accumulated_step=None, last_step=None, device=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))

        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(Adagrad, self).__init__(params, defaults)

        self.accumulated_step = accumulated_step

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if gamma is not None and accumulated_step is not None:
            self.c_value = Binom(gamma, accumulated_step)
            self.c_value = torch.tensor(self.c_value, device=self.device, dtype=torch.float32)  # Convert list to tensor
        else:
            self.c_value = torch.zeros(accumulated_step, device=self.device)  # Use a default tensor if c_value is None

        self.symbol = torch.ones([self.accumulated_step])
        self.symbol = self.symbol.to(self.device)# # (5,1,1)
        for i in range(self.accumulated_step):
            if i % 2 != 0:
                self.symbol[i] = -1

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p.data, initial_accumulator_value)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                # symbol 和 c_value 形状 (5) ->  (5, 1, 1...) num_dim个1
                num_dim = len(grad.size())  # (5, d_p.size())
                state['symbol'] = self.symbol.to(p.device)
                state['c_value'] = self.c_value.to(p.device)
                #state['v'] = state['v'].to(p.device)
                for n in range(num_dim):
                    state['symbol'] = state['symbol'].unsqueeze(-1)
                    state['c_value'] = state['c_value'].unsqueeze(-1)

                # 初始5个位置为0
                zeros = torch.zeros_like(grad).unsqueeze(0)  # (1, d_p.size())
                state['v'] = torch.cat([zeros] * 5, 0)  # (5, d_p.size())


                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                    grad = grad.add(p.data, alpha=group['weight_decay'])


                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])  # group['lr_decay']=0, clr= group['lr']

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)
                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum'].sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    if state['step'] == 1:
                        state['v'][0] = grad * grad

                    buf = state['symbol'] * state['v'] * state['c_value']
                    buf = torch.sum(buf, 0)

                    tmp = state['v'].clone()
                    state['v'][1:] = tmp.data[:-1]
                    state['v'][0] = buf

                    buf.addcmul_(grad, grad, value=1)
                    std = buf.sqrt().add_(1e-10)
                    p.data.addcdiv_(grad, std, value=-clr)

                    # state['sum'].addcmul_(1, grad, grad)
                    # std = state['sum'].sqrt().add_(1e-10)
                    # p.data.addcdiv_(-clr, grad, std)

        return loss

class SGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, gammas=None, accumulated_step=None, last_step=None, device=None):
        if lr is not required and lr < 0.0:
            raise ValueError("无效的学习率: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("无效的动量值: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("无效的权重衰减值: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov动量需要动量大于0且阻尼为0")
        super(SGD, self).__init__(params, defaults)

        self.accumulated_step = accumulated_step
        self.gammas = gammas
        self.current_step = 0
        self.last_step = last_step
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.symbol = torch.ones([self.accumulated_step])
        self.symbol = self.symbol.to(self.device)  # (5,1,1)
        for i in range(self.accumulated_step):
            if i % 2 != 0:
                self.symbol[i] = -1
    
    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)#遍历参数组（param_groups）中的每个group。对于每个group，调用setdefault方法，

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()#如果closure不为None，则调用closure函数来计算损失值，并将结果赋值给loss变量。
            
        self.current_step += 1  # step 加1

        for group in self.param_groups:  # 遍历参数组
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:  # 遍历参数
                if p.grad is None:     #检查其梯度（grad）
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)#将参数的梯度与参数本身的值相加，并乘以weight_decay，即进行权重衰减操作。
                if momentum != 0:
                    # 初始化 c_value函数， 动态gammas is a list containing max_gamma and min_gamma
                    if isinstance(self.gammas, list) and self.accumulated_step is not None and self.last_step is not None:
                        gamma = Getgamma(self.gammas[0], self.gammas[1], self.current_step, self.last_step)
                        self.c_value = Binom(gamma, self.accumulated_step)
                        self.c_value = torch.tensor(self.c_value, device=self.device, dtype=torch.float32)  # Convert list to tensor
                    else:
                        self.c_value = torch.zeros(self.accumulated_step, device=self.device)
                        
                    param_state = self.state[p]#获取参数在优化器的状态（param_state）。这段代码的作用是对参数的梯度进行处理，包括权重衰减和动量的操作。
                    if 'momentum_buffer' not in param_state:  # 初始化
                        zeros = torch.zeros_like(d_p).unsqueeze(0)  # 创建一个与参数梯度d_p形状相同的全零张量（zeros），
                                                                    # 并在第0维上添加一个维度，使其形状变为(1, d_p.size())。
                        param_state['momentum_buffer'] = torch.cat([zeros] * self.accumulated_step, 0)  # 形成一个形状为(accumulated_step, d_p.size())的动量缓冲区。
                        param_state['momentum_buffer'][0] = torch.clone(d_p).detach()#将参数梯度d_p的克隆副本赋值给动量缓冲区的第0个元素，
                                                                                     # 并使用detach方法将其从计算图中分离。
                        buf = param_state['momentum_buffer'][0]

                        num_dim = len(d_p.size())  # 获取参数梯度d_p的维度数量
                        param_state['symbol'] = self.symbol
                        param_state['c_value'] = self.c_value
                        for n in range(num_dim):
                            param_state['symbol'] = param_state['symbol'].unsqueeze(-1)
                            param_state['c_value'] = param_state['c_value'].unsqueeze(-1)
                            #使用循环将self.symbol和self.c_value的维度扩展到与参数梯度d_p相同的维度
                    else:
                        buf = (param_state['symbol'] * param_state['momentum_buffer'] * param_state['c_value'])
                        #计算动量更新的中间结果buf，三者相乘并求和
                        buf = torch.sum(buf, 0) + d_p #然后，将buf与参数梯度d_p相加，得到最终的动量更新结果。
                        tmp = param_state['momentum_buffer'].clone()
                        param_state['momentum_buffer'][1:] = tmp.data[:-1]
                        #使用tmp变量保存param_state['momentum_buffer']的副本，并将其向后平移一个位置
                        param_state['momentum_buffer'][0] = buf #将buf赋值给param_state['momentum_buffer']的第0个元素。
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf  #如果nesterov为True，则将动量更新结果d_p与buf的加权和相加；否则，直接将buf赋值给d_p。
                p.data.add_(d_p, alpha=-group['lr'])#使用p.data.add_方法对参数的值进行更新，减去学习率（group['lr']）乘以动量更新结果d_p。
        return loss

class MyAdam(Optimizer):
    """Implements Adagrad algorithm.

    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(self, params, lr=required, lr_decay=0, weight_decay=0, initial_accumulator_value=0,
                 gamma=None, accumulated_step=None,device=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))

        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(MyAdam, self).__init__(params, defaults)

        self.accumulated_step = accumulated_step

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if gamma is not None and accumulated_step is not None:
            self.c_value = Binom(gamma, accumulated_step)
            self.c_value = torch.tensor(self.c_value, device=self.device, dtype=torch.float32)  # Convert list to tensor
        else:
            self.c_value = torch.zeros(accumulated_step, device=self.device)

        self.symbol = torch.ones([self.accumulated_step])
        self.symbol = self.symbol.to(self.device)  # # (5,1,1)
        for i in range(self.accumulated_step):
            if i % 2 != 0:
                self.symbol[i] = -1

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p.data, initial_accumulator_value)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                # symbol 和 c_value 形状 (5) ->  (5, 1, 1...) num_dim个1
                num_dim = len(grad.size())  # (5, d_p.size())
                state['symbol'] = self.symbol.to(p.device)
                state['c_value'] = self.c_value.to(p.device)
                for n in range(num_dim):
                    state['symbol'] = state['symbol'].unsqueeze(-1)
                    state['c_value'] = state['c_value'].unsqueeze(-1)

                # 初始5个位置为0
                zeros = torch.zeros_like(grad).unsqueeze(0)  # (1, d_p.size())
                state['s'] = torch.cat([zeros] * 5, 0)  # (5, d_p.size())
                state['v'] = torch.cat([zeros] * 5, 0)  # (5, d_p.size())

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])  # group['lr_decay']=0, clr= group['lr']

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)
                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum'].sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    if state['step'] == 1:
                        state['s'][0] = grad * grad
                        state['v'][0] = torch.clone(grad).detach()

                    if state['symbol'].is_cuda!=1 or state['s'].is_cuda!=1 or state['c_value'].is_cuda!=1:
                        if state['symbol'].is_cuda!=1:print('state[\'symbol\'] isnt cuda\ntype:',type(state['symbol']))
                        if state['s'].is_cuda != 1: print('state[\'s\' isnt cuda]\ntype:',type(state['symbol']))
                        if state['c_value'].is_cuda != 1: print('state[\'c_value\' isnt cuda]\ntype:',type(state['symbol']))
                    buf_s = state['symbol'] * state['s'] * state['c_value']
                    buf_s = torch.sum(buf_s, 0)
                    buf_v = state['symbol'] * state['v'] * state['c_value']
                    buf_v = torch.sum(buf_v, 0) + grad

                    tmp_s = state['s'].clone()
                    state['s'][1:] = tmp_s.data[:-1]
                    state['s'][0] = buf_s
                    tmp_v = state['v'].clone()
                    state['v'][1:] = tmp_v.data[:-1]
                    state['v'][0] = buf_v

                    buf_s.addcmul_(grad, grad, value=1)
                    std = buf_s.sqrt().add_(1e-10)
                    p.data.addcdiv_(buf_v, std, value=-clr)


                    # state['sum'].addcmul_(1, grad, grad)
                    # std = state['sum'].sqrt().add_(1e-10)
                    # p.data.addcdiv_(-clr, grad, std)

        return loss


class MyAdamW(Optimizer):
    def __init__(self, params, lr=required, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False,
                 c_value=None, accumulated_step=5, device=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

        super(MyAdamW, self).__init__(params, defaults)

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if c_value is None:
            self.c_value = torch.zeros(accumulated_step, device=self.device)  # Use a default tensor if c_value is None
        else:
            self.c_value = torch.tensor(c_value, device=self.device)  # Convert list to tensor
        self.accumulated_step = accumulated_step

        self.symbol = torch.ones([self.accumulated_step])
        self.symbol = self.symbol.to(self.device)  # (5,1,1)
        for i in range(self.accumulated_step):
            if i % 2 != 0:
                self.symbol[i] = -1

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Add op to the c_value and symbol tensors
                # ... your code here to handle c_value and symbol ...

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group['amsgrad']:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

        return loss
