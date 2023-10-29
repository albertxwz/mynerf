from config.base import Args

class Args(Args):
    dataset: str = "fern" # name of the dataset
    exp_name: str = "baseline"
    base_data: str = "./data/nerf_llff_data/fern" # directory of data
    use_dirs: bool = True # use view direction or not
    L: int = 10 # embeding length of positional encoding of location
    L_dirs: int = 4 # embeding length of positional encoding of direction
    N_samples: int = 64 # number of sampling points
    N_importance: int = 64 # number of finely sampling points
    N_rand: int = 1024 # random seed
    factor: int = 8
    llffhold: int = 8
    raw_noise_std: float = 1e0
    use_ndc: bool = True