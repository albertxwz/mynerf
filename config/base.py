from tap import Tap

class Args(Tap):
    resume: str = None # the filename of resuming
    nums_iters: int = 200000 # number of iterations
    seed: int = 2023 # random seed
    start_val: int = 5000 # start to eval
    val_epoch: int = 10000 # evaluate every val_epoch epoch
    val_size: int = 1 # randomly choose val_size pics to eval
    ckpt_epoch: int = 5000 # save checkpoint every ckpt_epoch
    lrate: float = 5e-4 # learning rate
    lrate_decay: int = 250
    precrop: int = 0
    test_only: bool = False
    logpath: str = "./log"
    gamma: float = 0.95