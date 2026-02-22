from typing import List
from .event import Event
from .runtime_context import RuntimeContext

class Callback:
  """
  Base class for training callbacks.

  Lifecycle:

      on_train_start
          on_epoch_start
              on_batch_start
              on_batch_end
          on_epoch_end
      on_train_end

  Eval shares the same hooks but with trainer.is_training = False.
  """

  priority: int = 0  # smaller value = executed earlier

  def on_event(self, event: Event, context: RuntimeContext):
    pass

class CallbackManager:
  def __init__(self, callbacks: List[Callback]):
    self.callbacks = sorted(callbacks, key=lambda c: c.priority)

  def dispatch(self, event: Event, context: RuntimeContext):
    for cb in self.callbacks:
        cb.on_event(event, context)