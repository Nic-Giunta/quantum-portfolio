class TorchOptimizationLayer:
    def __init__(self, *args, **kwargs):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError('Install torch separately for differentiable layer experiments.') from exc
        raise NotImplementedError('Optional differentiable layer extension point.')
