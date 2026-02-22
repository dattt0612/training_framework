
from typing import Any, Dict, TYPE_CHECKING
from dataclasses import dataclass, field
if TYPE_CHECKING:
  from .trainer import Trainer
  from .event import Phase


@dataclass
class RuntimeContext:
  # core references
  trainer: Trainer

  # training state
  phase: Phase = Phase.TRAIN
  epoch: int = 0
  global_step: int = 0
  optim_step: int = 0
  
  # batch state
  batch_idx: int = 0
  batch: Any = None

  # forward results
  forward_results: Dict[str, Any] = None

  # control signal
  stop_training: bool = False
  skip_batch: bool = False
  skip_optim: bool = False

  # metrics
  metrics: Dict[str, Any] = field(default_factory=dict)

  @property
  def is_training(self) -> bool:
    return self.phase == Phase.TRAIN