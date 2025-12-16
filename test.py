from breakout import Breakout
from config import *
from game import Game
from mcts import MCTS
from networks import Network
from replay_buffer import ReplayBuffer
from self_play import get_root_node
from training import update_weights_parallel
from utility import scalar_to_support, support_to_scalar

network = Network.load(NETWORK_PATH)
game = Game(Env=Breakout)
mcts = MCTS(network)
rb = ReplayBuffer(100, BATCH_SIZE)
mcts.n_simulations = 30

game.states.append(game.get_current_state())
game.rewards.append(0)

with torch.no_grad():
  while not game.terminal() and len(game.actions) < MAX_MOVES:
    root = get_root_node(mcts, game)
    mcts.search(root)
    action = mcts.select_action(root)
    game.apply(action, root)
    game.compute_priorities(5)
    rb.add_game(game)

batch = rb.sample_batch(UNROLL_STEPS, TD_STEPS)[0]

optimizer = torch.optim.Adam(
  network.parameters(),
  lr=LR_INIT,
  weight_decay=WEIGHT_DECAY # L2 regularization (keeps weights small)
)

losses, new_priorities = update_weights_parallel(optimizer, network, batch)

# game_state, actions, targets, weight = batch[0]

# print("Game state:", game_state)
# print(f"Actions ({len(actions)}):", actions)
# print(f"Targets ({len(targets)}):")
# for target in targets:
#   target_value, target_reward, target_policy = target
#   value_val = support_to_scalar(target_value, is_prob=True)
#   reward_val = support_to_scalar(target_reward, is_prob=True)
#   print(f"  Value: {value_val:.4f}, Reward: {reward_val:.4f}, Policy: {target_policy}")
# print("Weight:", weight)

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
