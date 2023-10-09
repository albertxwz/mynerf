from config.base import Args

class Args(Args):
    dataset: str = "lego" # name of the dataset
    exp_name: str = "correct_config"
    base_data: str = "./data/nerf_synthetic/lego" # directory of data
    use_dirs: bool = True # use view direction or not
    L: int = 10 # embeding length of positional encoding of location
    L_dirs = 4 # embeding length of positional encoding of direction
    N_samples = 64 # number of sampling points
    N_importance = 64 # number of finely sampling points
    white_bkgd = True # white background or not
    N_rand = 1024 # random seed
    half_res: bool = True # lower the resolution or not
    lrate_decay: int = 500
    precrop: int = 500