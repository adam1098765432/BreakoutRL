from breakout import Breakout
from config import *
from game import Game, Observation
from mcts import MCTS
from networks import Network
from replay_buffer import ReplayBuffer
from self_play import get_root_node
from utility import scalar_to_support, support_to_scalar

network = Network.load(NETWORK_PATH)
game = Game(Env=Breakout)
mcts = MCTS(network)
rb = ReplayBuffer(1, 1)
mcts.n_simulations = 3

with torch.no_grad():
  root = get_root_node(mcts, game)
  mcts.search(root)
  action = mcts.select_action(root)
  game.apply(action, root)
  game.compute_priorities(1)
  rb.add_game(game)
  print(root)
  print(root.children[0])
  print(root.children[1])
  print(root.children[2])

batch = rb.sample_batch(1, 1)

game_state, actions, targets, weight = batch[0]

print("Game state:", game_state)
print("Actions:", actions)
print("Targets:")
for target in targets:
  target_value, target_reward, target_policy = target
  print("Value:", support_to_scalar(target_value, is_prob=True))
  print("Reward:", support_to_scalar(target_reward, is_prob=True))
  print("Policy:", target_policy)
print("Weight:", weight)

# state = game.get_current_state()
# network_output = network.initial_forward(state)
# network_output = network.recurrent_forward(network_output.hidden_state, 0)
# print(network_output)

# with torch.no_grad():
#   root = get_root_node(mcts, game)
#   mcts.search(root)
#   print(root)
#   print(root.children[0])
#   print(root.children[1])
#   print(root.children[2])

# with torch.no_grad():
#   root = get_root_node(mcts, game)
#   mcts.search(root)
#   action = mcts.select_action(root)
#   game.apply(action, root)
#   print(action)

# print(len(game.observations))
# print(Observation.serealize(game.observations[0]))
