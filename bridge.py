from multiprocessing import Queue
import os
import time
from config import *
from networks import Network, UniformNetwork
from game import Game

class Bridge:
  def __init__(self, max_games=1000, num_actors=NUM_ACTORS):
    self.num_actors = num_actors
    self.game_queue = Queue(maxsize=max_games)
    self.weight_queue = [Queue(maxsize=1) for _ in range(num_actors)]
    self.log_queue = Queue(maxsize=1)

  def send_log(self, log: dict[str, str]):
    if self.log_queue.full():
      self.log_queue.get()
    self.log_queue.put(log)

  def receive_log(self) -> dict[str, str]:
    return self.log_queue.get()

  def has_log(self):
    return not self.log_queue.empty()

  def send_game(self, game: Game):
    while self.game_queue.full():
      self.game_queue.get()
    self.game_queue.put(Game.serialize(game))

  def receive_game(self, Env):
    return Game.deserialize(self.game_queue.get(), Env)

  def has_game(self):
    return not self.game_queue.empty()

  def broadcast_network(self, network: Network):
    # Move to cpu before sending to actors
    state_dict = network.state_dict()
    for k in state_dict.keys():
      state_dict[k] = state_dict[k].cpu()

    # Send new weights to all actors
    for actor_id in range(self.num_actors):
      # Drop old weights, keep only newest
      while not self.weight_queue[actor_id].empty():
        self.weight_queue[actor_id].get()
      # Send new weights
      self.weight_queue[actor_id].put(state_dict)

  def receive_network(self, actor_id):
    if not self.weight_queue[actor_id].empty():
      network = Network().to(device)
      network.load_state_dict(self.weight_queue[actor_id].get())
      return network
    return UniformNetwork()
  
  def has_network(self, actor_id):
    return not self.weight_queue[actor_id].empty()

class NetworkBuffer:
  def __init__(self):
    self.network = self.load_network()

  def load_network(self):
    if os.path.exists(NETWORK_PATH):
      return Network.load(NETWORK_PATH)
    else:
      return UniformNetwork()

  def latest_network(self):
    return self.network
    
  def save_network(self, network: Network):
    self.network = network

