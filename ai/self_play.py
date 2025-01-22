import numpy as np
import chess

from ai.mcts import MCTSNode, mcts_search, select_action
from ai.utils import encode_board
from ai.move_mapping import MOVE_TO_INDEX, NUM_ACTIONS

class ReplayBuffer:
    def __init__(self, max_size=50000):
        from collections import deque
        self.buffer = deque(maxlen=max_size)

    def add_samples(self, samples):
        self.buffer.extend(samples)

    def sample_all(self):
        return list(self.buffer)

    def __len__(self):
        return len(self.buffer)

def self_play_game(model, simulations_per_move=200, temperature=1.0):
    board = chess.Board()
    states_actions = []

    move_count = 0

    while not board.is_game_over():
        root = MCTSNode(board.copy())
        mcts_search(root, model, simulations=simulations_per_move, c_puct=1.0)

        children_items = list(root.children.items())
        visits = np.array([child.visits for (_, child) in children_items], dtype=np.float32)
        sum_visits = visits.sum()

        policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if sum_visits > 0:
            for (move, child), v in zip(children_items, visits):
                idx = MOVE_TO_INDEX.get(move, None)
                if idx is not None:
                    policy_target[idx] = v / sum_visits

        state_tensor = encode_board(board)
        states_actions.append((state_tensor, policy_target))

        move = select_action(root, temperature)
        if move is None:
            break
        board.push(move)
        move_count += 1

    result = board.result()
    if result == '1-0':
        outcome = 1
    elif result == '0-1':
        outcome = -1
    else:
        outcome = 0

    final_data = []
    player = 1
    for (st, pol) in states_actions:
        val = outcome if player == 1 else -outcome
        final_data.append((st, pol, val))
        player = -player

    return final_data

def play_single_game(model_white, model_black, simulations_per_move=100):
    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            root = MCTSNode(board.copy())
            mcts_search(root, model_white, simulations=simulations_per_move)
            move = select_action(root, temperature=0.0)
        else:
            root = MCTSNode(board.copy())
            mcts_search(root, model_black, simulations=simulations_per_move)
            move = select_action(root, temperature=0.0)

        if move is None:
            break
        board.push(move)

    result = board.result()
    if result == '1-0':
        return 1
    elif result == '0-1':
        return -1
    else:
        return 0

def play_match(modelA, modelB, games=4, simulations_per_move=100):
    scoreA = 0.0
    scoreB = 0.0
    half = games // 2

    for _ in range(half):
        res = play_single_game(modelA, modelB, simulations_per_move)
        if res > 0:
            scoreA += 1.0
        elif res < 0:
            scoreB += 1.0
        else:
            scoreA += 0.5
            scoreB += 0.5

    for _ in range(games - half):
        res = play_single_game(modelB, modelA, simulations_per_move)
        if res > 0:
            scoreB += 1.0
        elif res < 0:
            scoreA += 1.0
        else:
            scoreA += 0.5
            scoreB += 0.5

    return scoreA, scoreB
