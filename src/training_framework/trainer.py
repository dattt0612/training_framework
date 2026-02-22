from typing import List, Callable, Dict, Any, TYPE_CHECKING
from tqdm import tqdm

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
  from .callback import CallbackManager, Callback
  from .runtime_context import RuntimeContext
  from .event import Phase, Event
import to_device

class Trainer:
  def __init__(
    self,
    model: Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    dev_loader: DataLoader = None,
    callbacks: List[Callback] = None,
    gpu_transform: Callable = None,
    gpu_augment: Callable = None,
    device: str = "cuda",
    grad_clip: float = 1.0
  ):
    self.model = model
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.dev_loader = dev_loader
    self.callbacks = CallbackManager(callbacks)
    self.train_loader = train_loader
    self.gpu_transform = gpu_transform
    self.gpu_augment = gpu_augment
    self.device = device
    self.grad_clip = grad_clip
    self.context = RuntimeContext(self)

  @property
  def dataloader(self):
    if self.context.is_training:
      return self.train_loader
    else:
      return self.dev_loader
  
  def forward(self, batch) -> Dict[str, Any]:
   raise NotImplementedError

  def backward(self):
    loss: Tensor = self.context.forward_results["loss"]
    loss.backward()

  def optim(self):
    self.optimizer.step()
    self.optimizer.zero_grad(set_to_none=True)
    self.context.optim_step += 1

  def do_one_epoch(self):
    self.callbacks.dispatch(Event.EPOCH_START, self.context)

    total_batches = len(self.dataloader)
    for batch_idx, batch in tqdm(enumerate(self.dataloader), total=total_batches):
      batch = to_device(batch, self.device)
  
      # update context
      self.context.batch_idx = batch_idx
      self.context.batch = batch

      self.callbacks.dispatch(Event.BATCH_START, self.context)
      if self.context.skip_batch:
        continue

      # BATCH_PREPARE
      self.callbacks.dispatch(Event.BATCH_PREPARE_START, self.context)
      self.context.batch = self.gpu_transform(self.context.batch)
      if self.context.is_training:
        self.context.batch = self.gpu_augment(self.context.batch)
      self.callbacks.dispatch(Event.BATCH_PREPARE_END, self.context)

      # FORWARD
      self.callbacks.dispatch(Event.FORWARD_START, self.context)
      self.context.forward_results = self.forward(batch=self.context.batch)
      self.callbacks.dispatch(Event.FORWARD_END, self.context)
      
      if self.context.is_training:
        # BACKWARD
        self.callbacks.dispatch(Event.BACKWARD_START, self.context)
        self.backward()
        self.callbacks.dispatch(Event.BACKWARD_END, self.context)

        # OPTIM
        if not self.context.skip_optim:
          self.callbacks.dispatch(Event.OPTIM_START, self.context)
          self.optim()
          self.callbacks.dispatch(Event.OPTIM_END, self.context)

      self.callbacks.dispatch(Event.BATCH_END, self.context)
    self.callbacks.dispatch(Event.EPOCH_END, self.context)

  def train_one_epoch(self):
    self.context.phase = Phase.TRAIN
    self.model.train()
    self.optimizer.zero_grad(set_to_none=True)
    self.do_one_epoch()
    self.context.global_step += 1

  @torch.no_grad()
  def eval_one_epoch(self):
    self.context.phase = Phase.EVAL
    self.model.eval()
    self.do_one_epoch()
  
  def train(self, n_epochs):
    # TRAIN_START
    self.callbacks.dispatch(Event.TRAIN_START, self.context)

    for _ in range(n_epochs):
      self.train_one_epoch()
      if self.dev_loader is not None:
        self.eval_one_epoch()

      self.context.epoch += 1

      if self.context.stop_training:
        break

    # TRAIN_END
    self.callbacks.dispatch(Event.TRAIN_END, self.context)
