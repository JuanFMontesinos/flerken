from .python_inheritance import ClassDict

__all__ = ['CHECKPOINT_OPTS', 'TRAINING_OPTIONS', 'EXPERIMENT_OPTIONS']

CHECKPOINT_OPTS = ClassDict({
    'checkpoint_name': 'checkpoint.pth',
    'save_type': 'cycle',  # cycle/iter
    'best_checkpoint_root': 'best',
    'saving_freq': 1
})

TRAINING_OPTIONS = ClassDict({
    'allocate_inputs': True,
    'allocate_outputs': True,
    'allocate_gt': True,
    'enable_train_logger': True,
    'enable_error_logger': True,
    'enable_backup': True
})

EXPERIMENT_OPTIONS = ClassDict({
    'experiment_name': 'datatime',  # 'uuiid/datatime
    'experiment_name_complexity': 0,
    'enable_model_logger': True,
})
