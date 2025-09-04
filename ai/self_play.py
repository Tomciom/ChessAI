import numpy as np
import chess
import chess.pgn
import chess.syzygy
import os
import json
import random

from ai.mcts import MCTSNode, mcts_search, select_action, flip_move
from ai.utils import encode_board_perspective
from ai.move_mapping import MOVE_TO_INDEX, NUM_ACTIONS
from ai.knowledge import get_endgame_move, get_opening_move

worker_tablebase = None
worker_opening_book = None

def self_play_game(model, simulations_per_move=200, temperature=1.0, mcts_batch_size=8):
    global worker_tablebase, worker_opening_book

    if worker_tablebase is None:
        try:
            path = os.path.join(os.path.dirname(__file__), '..', 'tablebases')
            if os.path.exists(path) and any(f.endswith(('.rtbw', '.rtbz')) for f in os.listdir(path)):
                worker_tablebase = chess.syzygy.open_tablebase(path)
                print(f"[PID: {os.getpid()}] Self-play worker załadował bazę zakończeń.")
            else:
                worker_tablebase = "error" 
                print(f"[PID: {os.getpid()}] Ostrzeżenie: Folder z bazami zakończeń pusty lub nie istnieje.")
        except Exception as e:
            print(f"[PID: {os.getpid()}] Błąd ładowania bazy zakończeń: {e}")
            worker_tablebase = "error"

    if worker_opening_book is None:
        try:
            path = os.path.join(os.path.dirname(__file__), 'opening_book.json')
            with open(path, 'r') as f:
                worker_opening_book = json.load(f)
            print(f"[PID: {os.getpid()}] Self-play worker załadował księgę otwarć.")
        except Exception as e:
            print(f"[PID: {os.getpid()}] Błąd ładowania księgi otwarć: {e}")
            worker_opening_book = {}

    board = chess.Board()
    states_actions = []

    while not board.is_game_over():
        move = None
        policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)

        opening_move = get_opening_move(board, worker_opening_book)
        if opening_move:
            move = opening_move
            root = MCTSNode(board.copy())
            mcts_search(root, model, simulations=50, c_puct=1.25, mcts_batch_size=mcts_batch_size)
            
            children_items = list(root.children.items())
            if children_items:
                visits = np.array([child.visits for (_, child) in children_items], dtype=np.float32)
                sum_visits = visits.sum()
                if sum_visits > 0:
                    for (m, child), v in zip(children_items, visits):
                        move_for_policy = m if board.turn == chess.WHITE else flip_move(m)
                        if move_for_policy in MOVE_TO_INDEX:
                            idx = MOVE_TO_INDEX.get(move_for_policy)
                            if idx is not None:
                                policy_target[idx] = v / sum_visits

        if move is None:
            if worker_tablebase != "error":
                endgame_move = get_endgame_move(board, worker_tablebase)
                if endgame_move:
                    move = endgame_move
                    move_for_policy = move if board.turn == chess.WHITE else flip_move(move)
                    if move_for_policy in MOVE_TO_INDEX:
                        idx = MOVE_TO_INDEX.get(move_for_policy)
                        if idx is not None:
                            policy_target[idx] = 1.0 
        
        if move is None:
            root = MCTSNode(board.copy())
            mcts_search(root, model, simulations=simulations_per_move, c_puct=1.25, mcts_batch_size=mcts_batch_size)
            
            children_items = list(root.children.items())
            if children_items:
                visits = np.array([child.visits for (_, child) in children_items], dtype=np.float32)
                sum_visits = visits.sum()
                if sum_visits > 0:
                    for (m, child), v in zip(children_items, visits):
                        move_for_policy = m if board.turn == chess.WHITE else flip_move(m)
                        if move_for_policy in MOVE_TO_INDEX:
                            idx = MOVE_TO_INDEX.get(move_for_policy)
                            if idx is not None:
                                policy_target[idx] = v / sum_visits

            move = select_action(root, temperature)
            
        if move is None:
            break

        state_tensor = encode_board_perspective(board)
        states_actions.append((state_tensor, policy_target))
        board.push(move)
    
    result = board.result()
    if result == '1-0': outcome = 1
    elif result == '0-1': outcome = -1
    else: outcome = 0

    final_data = []
    current_player_perspective_outcome = 1
    for (st, pol) in states_actions:
        val = outcome * current_player_perspective_outcome
        final_data.append((st, pol, val))
        current_player_perspective_outcome *= -1

    return final_data


class ReplayBuffer:
    def __init__(self, max_size=50000):
        from collections import deque
        self.buffer = deque(maxlen=max_size)

    def add_samples(self, samples):
        self.buffer.extend(samples)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)

    def sample_all(self):
        return list(self.buffer)

    def __len__(self):
        return len(self.buffer)

def play_single_game(model_white, model_black, simulations_per_move=100):
    tb = None
    try:
        path = os.path.join(os.path.dirname(__file__), '..', 'tablebases')
        if os.path.exists(path) and any(f.endswith(('.rtbw', '.rtbz')) for f in os.listdir(path)):
            tb = chess.syzygy.open_tablebase(path)
    except Exception:
        pass

    board = chess.Board()
    while not board.is_game_over():
        move = None
        endgame_move = get_endgame_move(board, tb)
        if endgame_move:
            move = endgame_move
        else:
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