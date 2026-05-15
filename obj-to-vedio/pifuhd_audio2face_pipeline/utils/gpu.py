from utils.logger import get_logger
log = get_logger(__name__)

def get_device():
    from config import GPU
    if not GPU.get('prefer_cuda', True):
        log.info('GPU disabled by config — using CPU')
        return 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            log.info(f'CUDA available: {name} — using GPU')
            return 'cuda'
        else:
            log.info('CUDA not available — falling back to CPU')
            return 'cpu'
    except ImportError:
        log.info('PyTorch not installed — using CPU')
        return 'cpu'

def cuda_available() -> bool:
    return get_device() == 'cuda'
