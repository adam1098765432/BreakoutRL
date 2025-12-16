import torch
import yaml

def get_device():
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")

# Load yaml config
with open("config.yaml", "r") as f:
  config = yaml.safe_load(f)

# Training parameters
NETWORK_PATH: str = config["network_path"]
REPLAY_BUFFER_PATH: str = config["replay_buffer_path"]
MAX_FLOAT = float('inf')
STATE_SIZE: int = config["state_size"]
ACTION_SIZE: int = config["action_size"]
SUPPORT_SIZE: int = config["support_size"]
HIDDEN_SIZE: int = config["hidden_size"]
DISCOUNT_FACTOR: float = config["discount_factor"]
N_SIMULATIONS: int = config["n_simulations"]
MAX_MOVES: int = config["max_moves"]
LR_INIT: float = config["lr_init"]
LR_DECAY_RATE: float = config["lr_decay_rate"]
LR_DECAY_STEPS: int = config["lr_decay_steps"]
TRAINING_STEPS: int = config["training_steps"]
SAVE_EVERY: int = config["save_every"]
LOG_EVERY: int = config["log_every"]
FETCH_EVERY: int = config["fetch_every"]
SAVE_GAMES_EVERY: int = config["save_games_every"]
UNROLL_STEPS: int = config["unroll_steps"]
TD_STEPS: int = config["td_steps"]
WEIGHT_DECAY: float = config["weight_decay"]
BATCH_SIZE: int = config["batch_size"]
NUM_ACTORS: int = config["num_actors"]
DIRICHLET_ALPHA: float = config["dirichlet_alpha"]
DIRICHLET_FRAC: float = config["dirichlet_frac"]
ACTORS_USE_CUDA: bool = config["actors_use_cuda"]
VALUE_LOSS_WEIGHT: float = config["value_loss_weight"]
CONSISTENCY_LOSS_WEIGHT: float = config["consistency_loss_weight"]
