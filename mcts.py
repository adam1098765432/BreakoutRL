from config import *
from utility import *
from networks import *
import random

class MinMaxStats:
  """Keeps track of the highest and lowest mean value in the search tree."""

  def __init__(self, min_val=None, max_val=None):
    self.max = max_val if max_val is not None else -MAX_FLOAT
    self.min = min_val if min_val is not None else MAX_FLOAT

  def update(self, val):
    self.max = max(self.max, val)
    self.min = min(self.min, val)

  def normalize(self, val):
    if self.max > self.min:
      return (val - self.min) / (self.max - self.min)
    return val

class Node:
  """
  Node in the MCTS search tree.
  """
  def __init__(self, prior: float):
    self.visit_count = 0
    self.prior = prior # Probability of picking this action
    self.value_sum = 0 # Mean value
    self.children: dict[int, Node] = {}
    self.hidden_state = None
    self.reward = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count
  
  def __str__(self):
    return f"Node(\n\
      visit_count={self.visit_count},\n\
      prior={self.prior},\n\
      value_sum={self.value_sum},\n\
      expanded={self.expanded()},\n\
      hidden_state={self.hidden_state},\n\
      reward={self.reward}\n\
    )"

class MCTS:
  """
  The MCTS search tree for the MuZero model.
  ### Note:
  There is no rollout since a value estimate is used instead.<br>
  The Dynamics and Prediction models are only used **once** per simulation.
  """
  def __init__(self, network: Network):
    self.network = network
    self.n_simulations = N_SIMULATIONS

  def ucb_score(self, parent: Node, child: Node, min_max_stats: MinMaxStats, c1=1.25, c2=19652):
    """
    :param parent: The parent node
    :param child: The child node
    :param c1: The exploration weight.
    :param c2: The exploration decay.
    """
    discount = DISCOUNT_FACTOR
    prior_weight = np.sqrt(parent.visit_count) / (1 + child.visit_count)
    prior_weight *= (c1 + np.log((parent.visit_count + c2 + 1) / c2))
    prior_score = child.prior * prior_weight
    if child.visit_count > 0:
      value_score = child.reward + discount * min_max_stats.normalize(child.value())
    else:
      value_score = 0
    return prior_score + value_score

  def select_action(node: Node, temperature: float) -> int:
    """
    Taken from: https://github.com/werner-duvaud/muzero-general
    Select action according to the visit count distribution and the temperature.
    The temperature is changed dynamically with the visit_softmax_temperature function
    in the config.
    """
    visit_counts = np.array(
      [child.visit_count for child in node.children.values()], dtype="int32"
    )
    actions = [action for action in node.children.keys()]
    if temperature == 0:
      action = actions[np.argmax(visit_counts)]
    elif temperature == MAX_FLOAT:
      action = np.random.choice(actions)
    else:
      # See paper appendix Data Generation
      visit_count_distribution = visit_counts ** (1 / temperature)
      visit_count_distribution = visit_count_distribution / sum(
        visit_count_distribution
      )
      action = np.random.choice(actions, p=visit_count_distribution)

    return action

  def select_child(self, node: Node, min_max_stats: MinMaxStats):
    scores = [(action, child, self.ucb_score(node, child, min_max_stats)) for action, child in node.children.items()]
    random.shuffle(scores) # Randomly break ties (stops from constantly picking the last action)
    action, child, _ = max(scores, key=lambda x: x[2])
    return action, child

  def search(self, root: Node):
    min_max_stats = MinMaxStats()

    for _ in range(self.n_simulations):
      node = root
      search_path = [node]

      while node.expanded():
        action, node = self.select_child(node, min_max_stats)
        search_path.append(node)

      parent = search_path[-2]
      network_output = self.network.recurrent_forward(parent.hidden_state, action)

      self.expand_node(node, network_output)
      self.backprop(search_path, network_output, min_max_stats)
  
  def expand_node(self, node: Node, network_output: NetworkOutput):
    if node.expanded():
      raise Exception("Tried to expand node twice")

    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    priors = torch.softmax(network_output.policy_logits, dim=1)[0].tolist()

    for i in range(ACTION_SIZE):
      child = Node(priors[i])
      node.children[i] = child

  def backprop(self, search_path: list[Node], network_output: NetworkOutput, min_max_stats: MinMaxStats):
    value = network_output.value

    for node in reversed(search_path):
      node.value_sum += value # Add negative if it's the opponent's turn (this is a single player game)
      node.visit_count += 1
      min_max_stats.update(node.reward + DISCOUNT_FACTOR * node.value())
      value = node.reward + DISCOUNT_FACTOR * value

  def add_exploration_noise(self, node: Node, alpha=DIRICHLET_ALPHA, frac=DIRICHLET_FRAC):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([alpha] * len(actions))
    for a, n in zip(actions, noise):
      node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
