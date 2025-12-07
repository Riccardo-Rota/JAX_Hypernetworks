from flax import nnx
import copy

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_epoch = 0
        #self.best_model = None
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, current_loss: float, current_model: nnx.Module, current_epoch: int):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_epoch = current_epoch
            #self.best_model = copy.deepcopy(current_model) # Copy by value, not reference
            self.best_state = nnx.state(current_model)
            self.counter = 0

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def reset(self):
        self.best_loss = float('inf')
        self.best_epoch = 0
        #self.best_model = None
        self.best_state = None
        self.counter = 0
        self.should_stop = False