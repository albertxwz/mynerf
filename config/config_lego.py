from config.base import Args

class Args(Args):
    dataset: str = "lego" # name of the dataset
    exp_name: str = "correct_config"
    base_data: str = "./data/nerf_synthetic/lego" # directory of data
    use_dirs: bool = True # use view direction or not
    L: int = 10 # embeding length of positional encoding of location
    L_dirs: int = 4 # embeding length of positional encoding of direction
    N_samples: int = 64 # number of sampling points
    N_importance: int = 128 # number of finely sampling points
    white_bkgd: bool = True # white background or not
    N_rand: int = 1024 # random seed
    half_res: bool = True # lower the resolution or not
    lrate_decay: int = 500
    precrop: int = 500