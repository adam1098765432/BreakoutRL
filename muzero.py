from bridge import Bridge
from config import NUM_ACTORS, TRAINING_STEPS, get_device
from replay_buffer import ReplayBuffer
from self_play import Environment, run_selfplay
from training import train
from utility import launch_job

def muzero(replay_buffer: ReplayBuffer, Env: Environment):
  bridge = Bridge()

  for actor_id in range(NUM_ACTORS):
    launch_job(run_selfplay, actor_id, bridge, TRAINING_STEPS // NUM_ACTORS, Env)

  global device
  device = get_device()
  print(f"Main process using device: {device}")

  train(replay_buffer, bridge, Env)


