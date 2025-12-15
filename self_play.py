import numpy as np
from bridge import Bridge, NetworkBuffer
from config import *
from game import Environment, Game
from mcts import MCTS, Node

def get_root_node(mcts: MCTS, game: Game):
  """
  Get the root node as the current state of the game.
  """
  root = Node(1.0 / ACTION_SIZE)
  current_state = game.get_current_state()
  network_output = mcts.network.initial_forward(current_state)
  mcts.expand_node(root, network_output)
  
  # Add initial state and reward targets
  game.states.append(current_state)
  game.rewards.append(0)

  return root

def run_selfplay(actor_id: int, bridge: Bridge, iterations: int, Env: Environment):
  global device
  device = get_device() if ACTORS_USE_CUDA else torch.device("cpu")
  print(f"Actor {actor_id} using device: {device}")
  
  if device.type == "cuda": torch.set_num_threads(1)

  iterations = int(iterations)
  network_buffer = NetworkBuffer()
  print(f"Playing {iterations} games...")
  for _ in range(iterations):
    fetch_network(actor_id, network_buffer, bridge)
    game = play_game(MCTS(network_buffer.latest_network()), Env, bridge)
    game.compute_priorities(TD_STEPS)
    bridge.send_game(game)
    
def play_game(mcts: MCTS, Env: Environment, bridge: Bridge):
  game = Game(Env=Env)
  
  # Are we supposed to add an initial observation?
  # game.states.append(game.get_current_state())

  action_histogram = [0] * ACTION_SIZE

  with torch.no_grad():
    while not game.terminal() and len(game.actions) < MAX_MOVES:
      root = get_root_node(mcts, game)
      mcts.add_exploration_noise(root)
      mcts.search(root)
      action = mcts.select_action(root)
      action_histogram[action] += 1
      game.apply(action, root)


  logs = {
    "Actions": ' '.join([f"{(action_histogram[i] / len(game.actions)):.2f}" for i in range(ACTION_SIZE)]),
    "Length": f"{len(game.actions):>5d}",
  }

  bridge.send_log(logs)

  return game

def fetch_network(actor_id: int, network_buffer: NetworkBuffer, bridge: Bridge):
  if bridge.has_network(actor_id):
    network = bridge.receive_network(actor_id)
    network_buffer.save_network(network)
    # print(f"Actor {actor_id}: Received latest network")
