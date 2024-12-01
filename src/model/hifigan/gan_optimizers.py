import torch
from torch import nn
from abc import ABC


class GANOptimizer:
    def __init__(self, generator_optimizer, discriminator_optimizer) -> None:
        self.generator_optimizer: torch.optim.Optimizer = generator_optimizer
        self.discriminator_optimizer: torch.optim.Optimizer = discriminator_optimizer


class GANLRScheduler:
    def __init__(self, generator_scheduler, discriminator_scheduler) -> None:
        self.generator_scheduler: torch.optim.lr_scheduler.LRScheduler = generator_scheduler
        self.discriminator_scheduler: torch.optim.lr_scheduler.LRScheduler = discriminator_scheduler
