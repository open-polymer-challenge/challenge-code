try:
    from .dataset_pyg import PygPolymerDataset
except ImportError:
    pass   

try:
    from .dataset import PolymerDataset
except ImportError:
    pass

try:
    from .evaluate import Evaluator
except ImportError:
    pass