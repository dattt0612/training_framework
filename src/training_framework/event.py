from enum import Enum, auto

class Event(Enum):
  # ===== GLOBAL =====
  TRAIN_START = auto()
  TRAIN_END = auto()

  # ===== EPOCH =====
  EPOCH_START = auto()
  EPOCH_END = auto()

  # ===== TRAIN LOOP =====
  BATCH_START = auto()
  BATCH_END = auto()

  BATCH_PREPARE_START = auto()
  BATCH_PREPARE_END = auto()

  FORWARD_START = auto()
  FORWARD_END = auto()

  BACKWARD_START = auto()
  BACKWARD_END = auto()

  OPTIM_START = auto()
  OPTIM_END = auto()
  
class Phase(Enum):
  TRAIN = auto()
  EVAL = auto()
