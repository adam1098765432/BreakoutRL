"""
Terms
- Policy: Move to play
- Value: Predicted winner or final score
- Immediate Reward: Reward for taking an action

MuZero
- Uses Monte Carlo Tree Search (MCTS)
- Requires past observations (frames/states) and actions to be stored
- Actions are encoded as constant bias planes
- MCTS predicts the next state using the previous state and action
  - Only masks valid actions at the root of the search tree
- MCTS can proceed past a terminal node, and is expected to return the same terminal state

Loss
 - CELoss()
"""