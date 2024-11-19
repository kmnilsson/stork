from copy import deepcopy

class EarlyStopping:
    def __init__(self,
                 patience = 10,
                 min_delta = 0.0,
                 mode = 'min',
                 verbose = True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for metrics to minimize (like loss), 'max' for metrics to maximize
            verbose: Whether to print improvement messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.reset()
        
    def reset(self):
        """Reset early stopping state"""
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.best_state = None
        self.stop_training = False
        
    def __call__(self, current_metric, model):
        """
        Check if training should stop
        
        Args:
            current_metric: Current value of monitored metric
            model: The RecurrentSpikingModel instance
            
        Returns:
            bool: Whether training should stop
        """
        improved = (
            (self.mode == 'min' and current_metric < self.best_metric - self.min_delta) or
            (self.mode == 'max' and current_metric > self.best_metric + self.min_delta)
        )
        
        if improved:
            if self.verbose:
                print(f'Validation metric improved from {self.best_metric:.6f} to {current_metric:.6f}')
            self.best_metric = current_metric
            self.best_state = deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            
        self.stop_training = self.counter >= self.patience
        return self.stop_training
    
    def restore_best_state(self, model):
        """Restore the best model state"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
