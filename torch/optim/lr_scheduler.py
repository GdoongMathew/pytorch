import abc
import types
import math
import warnings
import weakref
import collections
from collections import Counter
from typing import Sequence, Optional, Dict, Any, Union, Literal, Callable, NoReturn
from typing_extensions import deprecated
from functools import wraps, partial

from torch import inf
from bisect import bisect_right

from .optimizer import Optimizer

__all__ = [
    'LambdaLR',
    'MultiplicativeLR',
    'StepLR',
    'MultiStepLR',
    'ConstantLR',
    'LinearLR',
    'ExponentialLR',
    'SequentialLR',
    'CosineAnnealingLR',
    'ChainedScheduler',
    'ReduceLROnPlateau',
    'CyclicLR',
    'CosineAnnealingWarmRestarts',
    'OneCycleLR',
    'PolynomialLR',
    'Scheduler',
]

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class _SchedulerBase(abc.ABC):
    """https://github.com/pytorch/pytorch/issues/67760"""
    optimizer: Optimizer

    @abc.abstractmethod
    def state_dict(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_state_dict(self, state_dict):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, epoch: Optional[int] = ..., **kwargs):
        raise NotImplementedError


class _enable_get_lr_call:

    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False


def _format_param(name: str, param_groups: Sequence[Dict[str, Any]], param: Union[float, Sequence[float]]):
    """Return correctly formatted targets for each param group."""
    if isinstance(param, (float, int)):
        param = [param] * len(param_groups)

    elif isinstance(param, (list, tuple)) and len(param) != len(param_groups):
        raise ValueError(f"expected {len(param_groups)} values for {name}, got {len(param)}")

    return param


class Scheduler(_SchedulerBase):

    def __init__(
            self,
            optimizer: Optimizer,
            param_groups: Sequence[Dict[str, Any]] = None,
            last_step=-1,
            total_iters: Optional[int] = None,
    ):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an `Optimizer`.")
        self.optimizer = optimizer
        self.last_step = last_step

        if total_iters is not None:
            if not isinstance(total_iters, int):
                raise TypeError(
                    f"expected `total_iters` to be an int, "
                    f"but got {type(total_iters).__name__} in {self.__class__.__name__}."
                )
            if total_iters < 1:
                raise ValueError(
                    f"expected `total_iters` to be greater than 0, "
                    f"but got {total_iters} in {self.__class__.__name__}."
                )
        self.total_iters = total_iters

        param_groups = param_groups or optimizer.param_groups
        if not isinstance(param_groups, collections.abc.Sequence):
            raise TypeError(f"param_groups should be a sequence of mappings, but got {type(param_groups)}.")

        # Initialize epoch and base learning rates
        self.base_targets = [{} for _ in range(len(param_groups))]
        for gi, group in enumerate(param_groups):
            for target in self.targets:
                if f"initial_{target}" not in group:
                    if self.last_step != -1:
                        raise KeyError(
                            f"param `initial_{target}` is not specified in param_groups[{gi}] "
                            "when resuming an optimizer."
                        )
                    group.setdefault(f"initial_{target}", group[target])
                    self.base_targets[gi][f"initial_{target}"] = group[target]

        self.param_groups = param_groups

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self._initial_step()

    @property
    def targets(self) -> Sequence[str]:
        """The targets that the scheduler can update."""
        raise NotImplementedError("targets() must be implemented in derived classes")

    @property
    def last_targets(self):
        """The targets at the last step for each parameter group."""
        return [{key: param_group[key] for key in self.targets} for param_group in self.param_groups]

    @property
    def last_step(self):
        """The value last time passed to `lr_scheduler.step().` as `step`."""
        return self._last_step

    @last_step.setter
    def last_step(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"expected `last_step` to be an int, but got {type(value).__name__}.")
        self._last_step = value

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_targets(self, **kwargs) -> Optional[Sequence[Dict[str, Any]]]:
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None, **kwargs):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        if epoch is not None:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
            self.last_step = epoch
        else:
            self.last_step += 1

        if self.total_iters is not None and self.last_step >= self.total_iters:
            # If total_iter is set, the scheduler will stop updating learning rate after total_iter steps.
            return

        with _enable_get_lr_call(self):
            targets = self.get_targets(step=self.last_step, **kwargs)

            if targets is None:
                return

            if any(map(lambda x: not isinstance(x, dict), targets)):
                raise TypeError("get_targets should return a sequence of dicts")

        assert len(targets) == len(self.param_groups), (
            f"expected {len(self.param_groups)} targets, but got {len(targets)}"
        )

        for param_group, target in zip(self.param_groups, targets):
            param_group.update(target)


# Including _LRScheduler for backwards compatibility
# Subclass instead of assign because we want __name__ of _LRScheduler to be _LRScheduler (assigning would make it LRScheduler).
class _LRScheduler(Scheduler):
    pass


class LambdaLR(Scheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
            self,
            optimizer: Optimizer,
            lr_lambda, param_groups: Sequence[Dict[str, Any]] = None,
            **kwargs,
    ):
        param_groups = param_groups or optimizer.param_groups
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}")
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, param_groups=param_groups, **kwargs)

    @property
    def targets(self) -> Sequence[str]:
        return ['lr']

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_targets(self, *, step, **kwargs) -> Optional[Sequence[Dict[str, Any]]]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [
            {self.targets[0]: base_target[f"initial_{self.targets[0]}"] * lmbda(step)}
            for lmbda, base_target in zip(self.lr_lambdas, self.base_targets)
        ]


class MultiplicativeLR(LambdaLR):
    """Multiply the learning rate of each parameter group by the factor given
    in the specified function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> lmbda = lambda epoch: 0.95
        >>> scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    @property
    def targets(self) -> Sequence[str]:
        return ['lr']

    def get_targets(self, *, step, **kwargs):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
        if not step:
            return None
        return [
            {self.targets[0]: param["lr"] * lmbda(step)}
            for lmbda, param in zip(self.lr_lambdas, self.param_groups)
        ]


class StepLR(Scheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
            self,
            optimizer: Optimizer,
            step_size: int = 1,
            gamma: float = 0.1,
            **kwargs,
    ):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, **kwargs)

    @property
    def targets(self) -> Sequence[str]:
        return ['lr']

    def get_targets(self, *, step, **kwargs) -> Sequence[Dict[str, Any]]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        target = self.targets[0]
        return [
            {target: base_target[f"initial_{target}"] * self.gamma ** (step // self.step_size)}
            for base_target in self.base_targets
        ]


class MultiStepLR(Scheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
            self,
            optimizer: Optimizer,
            milestones: Sequence[int],
            gamma: float = 0.1,
            **kwargs,
    ):
        self.milestones: Counter = Counter(sorted(milestones))
        self.gamma = gamma
        super().__init__(optimizer, **kwargs)

    @property
    def targets(self) -> Sequence[str]:
        return ['lr']

    def get_targets(self, *, step, **kwargs) -> Optional[Sequence[Dict[str, Any]]]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        target = self.targets[0]
        if step not in self.milestones:
            return None

        return [
            {target: param_group[target] * self.gamma ** self.milestones[step]}
            for param_group in self.param_groups
        ]


class ConstantLR(Scheduler):
    """Multiply the learning rate of each parameter group by a small constant factor until the
    number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such multiplication of the small constant factor can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler multiplies the learning rate by the factor.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if epoch == 0
        >>> # lr = 0.025   if epoch == 1
        >>> # lr = 0.025   if epoch == 2
        >>> # lr = 0.025   if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = ConstantLR(optimizer, factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
            self,
            optimizer: Optimizer,
            factor: float = 1.0 / 3,
            total_iters: int = 5,
            **kwargs,
    ):
        if factor > 1.0 or factor < 0:
            raise ValueError('Constant multiplicative factor expected to be between 0 and 1.')

        self.factor = factor
        super().__init__(optimizer, total_iters=total_iters, **kwargs)

    @property
    def targets(self) -> Sequence[str]:
        return ["lr"]

    def get_targets(self, *, step, **kwargs) -> Optional[Sequence[Dict[str, Any]]]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        target = self.targets[0]
        if not step:
            return [{target: param_group[target] * self.factor} for param_group in self.param_groups]

        if step != self.total_iters:
            return None

        return [
            {target: param_group[target] * (1.0 / self.factor)}
            for param_group in self.param_groups
        ]


class LinearLR(Scheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
            self,
            optimizer,
            start_factor=1.0 / 3,
            end_factor=1.0,
            total_iters=5,
            **kwargs,
    ):
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError('Starting multiplicative factor expected to be greater than 0 and less or equal to 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        super().__init__(optimizer, total_iters=total_iters, **kwargs)

    @property
    def targets(self) -> Sequence[str]:
        return ["lr"]

    def get_targets(self, *, step, **kwargs) -> Sequence[Dict[str, Any]]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        target = self.targets[0]
        if not step:
            return [{target: param_group[target] * self.start_factor} for param_group in self.param_groups]

        if step > self.total_iters:
            return [{target: param_group[target]} for param_group in self.param_groups]

        return [
            {target: param_group[target] * (1. + (self.end_factor - self.start_factor) /
                                         (self.total_iters * self.start_factor + (step - 1) * (
                                                 self.end_factor - self.start_factor)))}
            for param_group in self.param_groups
        ]


@deprecated("Use `StepLR(step_size=1)` instead.")
class ExponentialLR(Scheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.
    """

    def __init__(
            self,
            optimizer: Optimizer,
            gamma: float,
            **kwargs,
    ):
        self.gamma = gamma
        super().__init__(optimizer, **kwargs)

    @property
    def targets(self) -> Sequence[str]:
        return ["lr"]

    def get_targets(self, *, step, **kwargs) -> Sequence[Dict[str, Any]]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        target = self.targets[0]
        return [{target: param_group[target] * self.gamma} for param_group in self.param_groups]


class PolynomialLR(Scheduler):
    """Decays the learning rate of each parameter group using a polynomial function
    in the given total_iters. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iter (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (float): The power of the polynomial. Default: 1.0.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Assuming optimizer uses lr = 0.001 for all groups
        >>> # lr = 0.001     if epoch == 0
        >>> # lr = 0.00075   if epoch == 1
        >>> # lr = 0.00050   if epoch == 2
        >>> # lr = 0.00025   if epoch == 3
        >>> # lr = 0.0       if epoch >= 4
        >>> scheduler = PolynomialLR(optimizer, total_iters=4, power=1.0)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
            self,
            optimizer: Optimizer,
            power: float = 1.0,
            total_iters: int = 5,
            **kwargs,
    ):
        self.power = power
        super().__init__(optimizer, total_iters=total_iters, **kwargs)

    @property
    def targets(self) -> Sequence[str]:
        return ["lr"]

    def get_targets(self, *, step, **kwargs):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if not step:
            return None

        decay_factor = ((1.0 - step / self.total_iters) / (
                1.0 - (step - 1) / self.total_iters)) ** self.power
        return [{self.targets[0]: group[self.targets[0]] * decay_factor} for group in self.param_groups]


class CosineAnnealingLR(Scheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
            self,
            optimizer: Optimizer,
            total_iters: int,
            eta_min=0,
            **kwargs,
    ):
        self.eta_min = eta_min
        super().__init__(optimizer, total_iters=total_iters, **kwargs)

    @property
    def targets(self) -> Sequence[str]:
        return ["lr"]

    def get_targets(self, *, step, **kwargs) -> Optional[Sequence[Dict[str, Any]]]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        target = self.targets[0]

        if not step:
            return
        elif self._step_count == 1 and step > 0:
            return [
                {target: self.eta_min + (base_target[target] - self.eta_min) *
                         (1 + math.cos(step * math.pi / self.total_iters)) / 2}
                for base_target, group in
                zip(self.base_targets, self.param_groups)
            ]
        elif (step - 1 - self.total_iters) % (2 * self.total_iters) == 0:
            return [
                {target: group['lr'] + (base_target[target] - self.eta_min) *
                         (1 - math.cos(math.pi / self.total_iters)) / 2}
                for base_target, group in
                zip(self.base_targets, self.param_groups)
            ]
        return [
            {target: (1 + math.cos(math.pi * step / self.total_iters)) /
                     (1 + math.cos(math.pi * (step - 1) / self.total_iters)) *
                     (group['lr'] - self.eta_min) + self.eta_min}
            for group in self.param_groups
        ]


class ReduceLROnPlateau(Scheduler):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): The number of allowed epochs with no improvement after
            which the learning rate will be reduced.
            For example, consider the case of having no patience (`patience = 0`).
            In the first epoch, a baseline is established and is always considered good as there's no previous baseline.
            In the second epoch, if the performance is worse than the baseline,
            we have what is considered an intolerable epoch.
            Since the count of intolerable epochs (1) is greater than the patience level (0),
            the learning rate is reduced at the end of this epoch.
            From the third epoch onwards, the learning rate continues to be reduced at the end of each epoch
            if the performance is worse than the baseline. If the performance improves or remains the same,
            the learning rate is not adjusted.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(
            self,
            optimizer: Optimizer,
            param_groups: Optional[Sequence[Dict[str, Any]]] = None,
            mode: Literal["min", "max"] = 'min',
            factor: float = 0.1,
            patience: int = 10,
            threshold: float = 1e-4,
            threshold_mode: Literal["rel", "abs"] = "rel",
            cooldown: int = 0,
            min_lr: Union[float, Sequence[float]] = 0.0,
            eps: float = 1e-8,
            **kwargs,
    ):

        if factor >= 1.0 or factor < 0.0:
            raise ValueError('Factor should be 0.0 <= x < 1.0.')
        self.factor = factor

        param_groups = param_groups or optimizer.param_groups

        # Attach optimizer
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(param_groups):
                raise ValueError(f"expected {len(param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(param_groups)

        self.patience = patience

        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.num_bad_iters = 0

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self.best = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps

        self._init_is_better()
        self._reset()

        super().__init__(optimizer=optimizer, param_groups=param_groups, **kwargs)

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step(metrics=None)

    @property
    def targets(self) -> Sequence[str]:
        return ["lr"]

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_iters = 0

    def get_targets(
            self,
            *,
            step: int,
            metrics: float,
            **kwargs
    ) -> Optional[Sequence[Dict[str, Any]]]:

        if not step:
            return

        current = float(metrics)
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_iters = 0
        else:
            self.num_bad_iters += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_iters = 0  # ignore any bad epochs in cooldown

        if self.num_bad_iters > self.patience:
            lrs = self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_iters = 0
        else:
            lrs = None

        return lrs

    def _reduce_lr(self) -> Sequence[Dict[str, Any]]:
        lrs = [{} for _ in range(len(self.param_groups))]
        for i, (param_group, min_lr) in enumerate(zip(self.param_groups, self.min_lrs)):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, min_lr)
            lrs[i] = {"lr": new_lr if old_lr - new_lr > self.eps else old_lr}
        return lrs

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        if self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        if self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        # mode == 'max' and epsilon_mode == 'abs':
        return a > best + self.threshold

    def _init_is_better(self):
        if self.mode not in {"min", "max"}:
            raise ValueError(f"mode `{self.mode}` is unknown!")
        if self.threshold_mode not in {'rel', 'abs'}:
            raise ValueError(f"threshold mode {self.threshold_mode} is unknown!")

        self.mode_worse = inf if self.mode == 'min' else -inf

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._init_is_better()


class CyclicLR(Scheduler):
    r"""Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This class has three built-in policies, as put forth in the paper:

    * "triangular": A basic triangular cycle without amplitude scaling.
    * "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
    * "exp_range": A cycle that scales initial amplitude by :math:`\text{gamma}^{\text{cycle iterations}}`
      at each cycle iteration.

    This implementation was adapted from the github repo: `bckenstler/CLR`_

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then 'mode' is ignored.
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()


    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(
            self,
            optimizer: Optimizer,
            max_lr: Union[float, Sequence[float]],
            base_lr: Optional[Union[float, Sequence[float]]] = None,
            param_groups: Optional[Sequence[Dict[str, Any]]] = None,
            step_size_up: int = 2000,
            step_size_down: Optional[int] = None,
            mode: Literal["triangular", "triangular2", "exp_range"] = "triangular",
            gamma: float = 1.,
            scale_fn: Optional[Callable[[float], float]] = None,
            scale_mode: Literal["cycle", "iterations"] = "cycle",
            cycle_momentum: bool = True,
            base_momentum: Optional[Union[float, Sequence[float]]] = 0.8,
            max_momentum: Optional[Union[float, Sequence[float]]] = 0.9,
            last_step: int = -1,
    ):
        param_groups = param_groups or optimizer.param_groups

        max_lrs = _format_param('max_lr', param_groups, max_lr)
        base_lrs = [group["lr"] for group in param_groups] \
            if not base_lr else _format_param("base_lr", param_groups, base_lr)

        if last_step == -1:
            for base_lr, max_lr, group in zip(base_lrs, max_lrs, param_groups):
                group.update({
                    "lr": base_lr,
                    "base_lr": base_lr,
                    "max_lr": max_lr,
                })

        step_size_down = step_size_down or step_size_up

        total_iters = step_size_up + step_size_down
        self.step_ratio = step_size_up / total_iters

        self.mode = mode
        self.gamma = gamma
        self.cycle_momentum = cycle_momentum

        self._scale_fn_ref = None
        self._scale_fn_custom = scale_fn
        self.scale_mode = scale_mode
        self._init_scale_fn()

        if self.cycle_momentum:
            if 'momentum' not in optimizer.defaults and 'betas' not in optimizer.defaults:
                raise ValueError('optimizer must support momentum or beta1 with `cycle_momentum` option enabled')

            self.use_beta1 = 'betas' in optimizer.defaults

            base_momentums = _format_param('base_momentum', param_groups, base_momentum)
            max_momentums = _format_param('max_momentum', param_groups, max_momentum)
            if last_step == -1:
                for m_momentum, b_momentum, group in zip(
                        max_momentums,
                        base_momentums,
                        param_groups,
                ):
                    if self.use_beta1:
                        group['betas'] = (m_momentum, *group['betas'][1:])
                    else:
                        group['momentum'] = m_momentum
                    group['max_momentum'] = m_momentum
                    group['base_momentum'] = b_momentum

        super().__init__(
            optimizer=optimizer,
            param_groups=param_groups,
            total_iters=total_iters,
            last_step=last_step,
        )

    @property
    def targets(self) -> Sequence[str]:
        targets = ["lr"]
        if self.cycle_momentum:
            targets += ["momentum"] if self.use_beta1 else ["betas"]
        return targets

    def _init_scale_fn(self):
        if self._scale_fn_custom is not None:
            return
        if self.mode == 'triangular':
            self._scale_fn_ref = self._triangular_scale_fn
            self.scale_mode = 'cycle'
        elif self.mode == 'triangular2':
            self._scale_fn_ref = self._triangular2_scale_fn
            self.scale_mode = 'cycle'
        elif self.mode == 'exp_range':
            self._scale_fn_ref = partial(self._exp_range_scale_fn, self.gamma)
            self.scale_mode = 'iterations'
        else:
            raise ValueError(f'mode {self.mode} is invalid and scale_fn is None')

    def scale_fn(self, x):
        if self._scale_fn_custom is not None:
            return self._scale_fn_custom(x)
        return self._scale_fn_ref(x)  # static method

    @staticmethod
    def _triangular_scale_fn(x):
        return 1.

    @staticmethod
    def _triangular2_scale_fn(x):
        return 1 / (2. ** (x - 1))

    @staticmethod
    def _exp_range_scale_fn(gamma, x):
        return gamma ** x

    # def get_lr(self):
    def get_targets(self, *, step: int, **kwargs) -> Optional[Sequence[Dict[str, Any]]]:
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        cycle = math.floor(1 + step / self.total_iters)
        x = 1. + step / self.total_iters - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        rets = [{} for _ in self.param_groups]
        for pg_i, param_group in enumerate(self.param_groups):
            base_lr, max_lr = param_group["base_lr"], param_group["max_lr"]
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == "cycle":
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(step)
            ret = {"lr": lr}

            if self.cycle_momentum:
                base_momentum, max_momentum = param_group["base_momentum"], param_group["max_momentum"]
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == 'cycle':
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(step)
                if self.use_beta1:
                    ret["betas"] = (momentum, *param_group['betas'][1:])
                else:
                    ret["momentum"] = momentum

            rets[pg_i] = ret
        return rets

    def state_dict(self):
        state = super().state_dict()
        # We are dropping the `_scale_fn_ref` attribute because it is a
        # `weakref.WeakMethod` and can't be pickled.
        state.pop('_scale_fn_ref')
        fn = state.pop('_scale_fn_custom')
        state['_scale_fn_custom'] = None
        if fn is not None and not isinstance(fn, types.FunctionType):
            # The _scale_fn_custom will only be saved if it is a callable object
            # and not if it is a function or lambda.
            state['_scale_fn_custom'] = fn.__dict__.copy()

        return state

    def load_state_dict(self, state_dict):
        fn = state_dict.pop('_scale_fn_custom')
        super().load_state_dict(state_dict)
        if fn is not None:
            self._scale_fn_custom.__dict__.update(fn)
        self._init_scale_fn()


class CosineAnnealingWarmRestarts(Scheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
            self,
            optimizer: Optimizer,
            T_0: int,
            T_mult: int = 1,
            eta_min: Union[int, float] = 0,
            last_step: int = -1,
            **kwargs,
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if not isinstance(eta_min, (float, int)):
            raise ValueError(f"Expected float or int eta_min, but got {eta_min} of type {type(eta_min)}")
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_step
        super().__init__(optimizer, last_step=last_step, **kwargs)

    @property
    def targets(self) -> Sequence[str]:
        return ["lr"]

    def get_lr(self):
        return [
            {self.targets[0]: self.eta_min + (base_target[f"initial_{self.targets[0]}"] - self.eta_min) * (
                    1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2}
            for base_target in self.base_targets
        ]

    def get_targets(self, *, step: int, **kwargs) -> Optional[Sequence[Dict[str, Any]]]:
        """Step could be called after every batch update

        Example:
            >>> # xdoctest: +SKIP("Undefined vars")
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>         scheduler.step(epoch + i / iters)

        This function can be called in an interleaved way.

        Example:
            >>> # xdoctest: +SKIP("Undefined vars")
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if step >= self.T_0:
            if self.T_mult == 1:
                self.T_cur = step % self.T_0
            else:
                n = int(math.log((step / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                self.T_cur = step - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                self.T_i = self.T_0 * self.T_mult ** n
        else:
            self.T_i = self.T_0
            self.T_cur = step

        return self.get_lr()


class OneCycleLR(Scheduler):
    r"""Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    The 1cycle learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch

    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    The default behaviour of this scheduler follows the fastai implementation of 1cycle, which
    claims that "unpublished work has shown even better results by using only two phases". To
    mimic the behaviour of the original paper instead, set ``three_phase=True``.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
        pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
            linear annealing.
            Default: 'cos'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.85
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.95
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        three_phase (bool): If ``True``, use a third phase of the schedule to annihilate the
            learning rate according to 'final_div_factor' instead of modifying the second
            phase (the first two phases will be symmetrical about the step indicated by
            'pct_start').
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         optimizer.step()
        >>>         scheduler.step()


    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """

    def __init__(
            self,
            optimizer: Optimizer,
            max_lr: Union[float, Sequence[float]],
            total_iters: Optional[int] = None,
            epochs: Optional[int] = None,
            steps_per_epoch: Optional[int] = None,
            param_groups: Optional[Sequence[Dict[str, Any]]] = None,
            pct_start: float = 0.3,
            anneal_strategy: Literal["cos", "linear"] = "cos",
            cycle_momentum: bool = True,
            base_momentum: float = 0.85,
            max_momentum: float = 0.95,
            div_factor: float = 25.,
            final_div_factor: float = 1e4,
            three_phase: bool = False,
            last_step: int = -1,
    ):
        param_groups = param_groups or optimizer.param_groups

        # Validate total_steps
        if total_iters is None and epochs is None and steps_per_epoch is None:
            raise ValueError("You must define either `total_iters` OR (epochs AND steps_per_epoch)")
        if total_iters is None:
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError(f"Expected positive integer epochs, but got {epochs}")
            if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                raise ValueError(f"Expected positive integer steps_per_epoch, but got {steps_per_epoch}")
            total_iters = epochs * steps_per_epoch

        if three_phase:
            self._schedule_phases = [
                {
                    'end_step': float(pct_start * total_iters) - 1,
                    'start_lr': 'initial_lr',
                    'end_lr': 'max_lr',
                    'start_momentum': 'max_momentum',
                    'end_momentum': 'base_momentum',
                },
                {
                    'end_step': float(2 * pct_start * total_iters) - 2,
                    'start_lr': 'max_lr',
                    'end_lr': 'initial_lr',
                    'start_momentum': 'base_momentum',
                    'end_momentum': 'max_momentum',
                },
                {
                    'end_step': total_iters - 1,
                    'start_lr': 'initial_lr',
                    'end_lr': 'min_lr',
                    'start_momentum': 'max_momentum',
                    'end_momentum': 'max_momentum',
                },
            ]
        else:
            self._schedule_phases = [
                {
                    'end_step': float(pct_start * total_iters) - 1,
                    'start_lr': 'initial_lr',
                    'end_lr': 'max_lr',
                    'start_momentum': 'max_momentum',
                    'end_momentum': 'base_momentum',
                },
                {
                    'end_step': total_iters - 1,
                    'start_lr': 'max_lr',
                    'end_lr': 'min_lr',
                    'start_momentum': 'base_momentum',
                    'end_momentum': 'max_momentum',
                },
            ]

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError(f"Expected float between 0 and 1 pct_start, but got {pct_start}")

        # Validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError(f"anneal_strategy must by one of 'cos' or 'linear', instead got {anneal_strategy}")

        self.anneal_func = self._annealing_cos if anneal_strategy == "cos" else self._annealing_linear

        # Initialize learning rate variables
        max_lrs = _format_param('max_lr', param_groups, max_lr)
        if last_step == -1:
            for idx, group in enumerate(param_groups):
                group['initial_lr'] = max_lrs[idx] / div_factor
                group['max_lr'] = max_lrs[idx]
                group['min_lr'] = group['initial_lr'] / final_div_factor

        # Initialize momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            if 'momentum' not in self.optimizer.defaults and 'betas' not in self.optimizer.defaults:
                raise ValueError('optimizer must support momentum or beta1 with `cycle_momentum` option enabled')
            self.use_beta1 = 'betas' in self.optimizer.defaults
            max_momentums = _format_param('max_momentum', param_groups, max_momentum)
            base_momentums = _format_param('base_momentum', param_groups, base_momentum)
            if last_step == -1:
                for m_momentum, b_momentum, group in zip(max_momentums, base_momentums, param_groups):
                    if self.use_beta1:
                        group['betas'] = (m_momentum, *group['betas'][1:])
                    else:
                        group['momentum'] = m_momentum
                    group['max_momentum'] = m_momentum
                    group['base_momentum'] = b_momentum

        super().__init__(optimizer, total_iters=total_iters, last_step=last_step)

    @property
    def targets(self) -> Sequence[str]:
        targets = ["lr"]
        if self.cycle_momentum:
            targets += ["momentum"] if self.use_beta1 else ["betas"]
        return targets

    @staticmethod
    def _annealing_cos(start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    @staticmethod
    def _annealing_linear(start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    def get_targets(self, *, step: int, **kwargs) -> Optional[Sequence[Dict[str, Any]]]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if step > self.total_iters:
            raise ValueError(
                f"Tried to step {step} times. The specified number of total steps is {self.total_iters}."
            )

        targets = [{} for _ in range(len(self.param_groups))]

        for pg_i, group in enumerate(self.param_groups):
            start_step = 0
            for i, phase in enumerate(self._schedule_phases):
                end_step = phase['end_step']
                if step <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step - start_step) / (end_step - start_step)
                    computed_lr = self.anneal_func(group[phase['start_lr']], group[phase['end_lr']], pct)
                    if self.cycle_momentum:
                        computed_momentum = self.anneal_func(group[phase['start_momentum']],
                                                             group[phase['end_momentum']], pct)
                    break
                start_step = phase['end_step']

            target = {"lr": computed_lr}
            if self.cycle_momentum:
                if self.use_beta1:
                    target["betas"] = (computed_momentum, *group['betas'][1:])
                else:
                    target["momentum"] = computed_momentum
            targets[pg_i] = target

        return targets


class ComposeScheduler(_SchedulerBase, abc.ABC):
    def __init__(
            self,
            schedulers: Sequence[Scheduler],
            total_iters: Optional[int] = None,
            last_step: int = -1
    ):
        if len(schedulers) < 1:
            raise ValueError(f"{self.__class__.__name__} expects at least one scheduler, but got no scheduler.")

        base_optimizer = schedulers[0].optimizer

        for scheduler_idx, scheduler in enumerate(schedulers):
            if not isinstance(scheduler, Scheduler):
                raise TypeError(
                    f"{self.__class__.__name__} expects all schedulers to be of type `Scheduler`, but got "
                    f"an object of type {type(scheduler)} at index {scheduler_idx}"
                )
            if scheduler.optimizer != base_optimizer:
                raise ValueError(
                    f"{self.__class__.__name__} expects all schedulers to belong to the same optimizer, but "
                    f"got schedulers at index {scheduler_idx} to be different than the optimizer passed in."
                )
        self.schedulers = schedulers
        self.last_step = last_step

        if total_iters is not None:
            if not isinstance(total_iters, int):
                raise TypeError(
                    f"Expected integer type for total_iters, but got {type(total_iters)} in {self.__class__.__name__}."
                )
            if total_iters < 1:
                raise ValueError(
                    f"Expected positive integer for total_iters, but got {total_iters} in {self.__class__.__name__}."
                )

        self.total_iters = total_iters

    @property
    def last_step(self):
        return self._last_step

    @last_step.setter
    def last_step(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"Expected integer type for last_step, but got {type(value)}.")
        self._last_step = value

    def step(self, epoch: Optional[int] = None, **kwargs):
        if epoch is not None:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
            self.last_step = epoch
        else:
            self.last_step += 1

        if self.total_iters is not None and self.last_step >= self.total_iters:
            return

        self.step_schedulers(step=self.last_step, **kwargs)

    @abc.abstractmethod
    def step_schedulers(self, *, step: int, **kwargs) -> NoReturn:
        raise NotImplementedError(f"get_targets() is not implemented in {self.__class__.__name__}")

    @property
    def optimizer(self) -> Optimizer:
        return self.schedulers[0].optimizer

    def state_dict(self):
        state_dict = {key: value if key != "schedulers" else [scheduler.state_dict() for scheduler in self.schedulers]
                      for key, value in self.__dict__.items()}
        return state_dict

    def load_state_dict(self, state_dict):
        schedulers_state_dict = state_dict.pop('schedulers')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        self.__dict__["schedulers"] = [schedulers_state_dict]

        for idx, scheduler_state_dict in enumerate(schedulers_state_dict):
            self.schedulers[idx].load_state_dict(scheduler_state_dict)

    @property
    def last_targets(self) -> Dict[str, Sequence[Dict[str, Any]]]:
        return {
            scheduler.__class__.__name__: scheduler.last_targets
            for scheduler in self.schedulers
        }


class SequentialLR(ComposeScheduler):
    """Receives the list of schedulers that is expected to be called sequentially during
    optimization process and milestone points that provides exact intervals to reflect
    which scheduler is supposed to be called at a given epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): Does nothing.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.1     if epoch == 0
        >>> # lr = 0.1     if epoch == 1
        >>> # lr = 0.9     if epoch == 2
        >>> # lr = 0.81    if epoch == 3
        >>> # lr = 0.729   if epoch == 4
        >>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        >>> scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
            self,
            schedulers: Sequence[Scheduler],
            milestones: Sequence[int],
            **kwargs
    ):
        super().__init__(schedulers=schedulers, **kwargs)
        milestones = sorted(milestones)
        if len(milestones) != len(self.schedulers) - 1:
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                f"than the number of milestone points, but got number of schedulers {len(self.schedulers)} and the "
                f"number of milestones to be equal to {len(milestones)}"
            )

        if self.total_iters is not None:
            if milestones[-1] >= self.total_iters:
                raise ValueError(
                    f"Last milestone point {milestones[-1]} should be less than total_iters {self.total_iters}"
                )

        self.milestones = milestones

        # Reset learning rates back to initial values
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"]

        # Perform the initial step for only the first scheduler
        self.schedulers[0]._initial_step()

    def step_schedulers(self, *, step: int, **kwargs) -> NoReturn:
        idx = bisect_right(self.milestones, step)
        scheduler = self.schedulers[idx]
        sch_step = step if idx == 0 else step - self.milestones[idx - 1]
        scheduler.step(epoch=sch_step, **kwargs)


class ChainedScheduler(ComposeScheduler):
    """Chains list of learning rate schedulers. It takes a list of chainable learning
    rate schedulers and performs consecutive step() functions belonging to them by just
    one call.

    Args:
        schedulers (list): List of chained schedulers.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.09     if epoch == 0
        >>> # lr = 0.081    if epoch == 1
        >>> # lr = 0.729    if epoch == 2
        >>> # lr = 0.6561   if epoch == 3
        >>> # lr = 0.59049  if epoch >= 4
        >>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        >>> scheduler = ChainedScheduler([scheduler1, scheduler2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def step_schedulers(self, *, step: int, **kwargs) -> NoReturn:
        for scheduler in self.schedulers:
            scheduler.step(epoch=step, **kwargs)
