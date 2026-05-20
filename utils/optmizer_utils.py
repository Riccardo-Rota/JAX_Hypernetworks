import jax
import optax
from optax.contrib._reduce_on_plateau import ReduceLROnPlateauState

def extract_lr_info(opt_state) -> tuple:
    """
    Traverses the optimizer state PyTree to dynamically locate the base learning rate 
    and plateau scale factor. 
    Uses string matching to bypass Optax's private module class-switching.
    """
    base_lr = None
    lr_scale = None

    # Target class names instead of object references
    TARGET_STATES = (
        'ReduceLROnPlateauState', 
        'InjectHyperparamsState', 
        'InjectStatefulHyperparamsState'
    )

    # Instruct JAX to stop traversing if it hits any of the target class names
    def is_target_state(node):
        return type(node).__name__ in TARGET_STATES

    # Extract only the target leaves from the deeply nested opt_state
    leaves = jax.tree_util.tree_leaves(opt_state, is_leaf=is_target_state)

    for leaf in leaves:
        state_name = type(leaf).__name__
        
        if state_name == 'ReduceLROnPlateauState':
            lr_scale = leaf.scale.item() if hasattr(leaf.scale, 'item') else leaf.scale
            
        elif state_name in ('InjectHyperparamsState', 'InjectStatefulHyperparamsState'):
            # Both classes store their values in a hyperparams dict
            lr = leaf.hyperparams.get('learning_rate', None)
            if lr is not None:
                base_lr = lr.item() if hasattr(lr, 'item') else lr

    return base_lr, lr_scale