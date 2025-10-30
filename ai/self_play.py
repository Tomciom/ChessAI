# Plik: ai/self_play.py
# Wersja: Zaktualizowana z dynamiczną temperaturą i poprawioną logiką nagród

import numpy as np
import chess
import chess.pgn
import chess.syzygy
import os
import json
from collections import deque
import random

# Importujemy istniejące komponenty
from ai.mcts import MCTSNode, mcts_search, select_action, flip_move
from ai.utils import encode_board_perspective
from ai.move_mapping import MOVE_TO_INDEX, NUM_ACTIONS
from ai.knowledge import get_endgame_move, get_opening_move

# Globalne zmienne dla workerów, aby nie ładować zasobów w każdej grze
worker_tablebase = None
worker_opening_book = None

def self_play_game(model, simulations_per_move=200, temp_threshold=30, mcts_batch_size=8):
    """
    Rozgrywa jedną partię sam ze sobą, używając dynamicznej temperatury.
    `temp_threshold`: Liczba półruchów, po której temperatura spada do zera.
    """
    global worker_tablebase, worker_opening_book

    # --- Inicjalizacja zasobów (księga, bazy) przy pierwszym uruchomieniu workera ---
    if worker_tablebase is None:
        try:
            path = os.path.join(os.path.dirname(__file__), '..', 'tablebases')
            if os.path.exists(path) and any(f.endswith(('.rtbw', '.rtbz')) for f in os.listdir(path)):
                worker_tablebase = chess.syzygy.open_tablebase(path)
            else:
                worker_tablebase = "error"
        except Exception:
            worker_tablebase = "error"

    if worker_opening_book is None:
        try:
            path = os.path.join(os.path.dirname(__file__), 'opening_book.json')
            with open(path, 'r') as f:
                worker_opening_book = json.load(f)
        except Exception:
            worker_opening_book = {}

    board = chess.Board()
    game_history = []
    
    # Główna pętla gry
    while not board.is_game_over(claim_draw=True):
        move = None
        
        # --- Hierarchia decyzyjna (księga, bazy) ---
        opening_move = get_opening_move(board, worker_opening_book)
        if opening_move:
            move = opening_move
        
        if move is None:
            if worker_tablebase != "error":
                endgame_move = get_endgame_move(board, worker_tablebase)
                if endgame_move:
                    move = endgame_move

        # --- Główna logika MCTS ---
        # Zapisujemy stan PRZED wykonaniem ruchu
        state_tensor = encode_board_perspective(board)
        
        # Obliczamy politykę (rozkład prawdopodobieństw) z MCTS
        root = MCTSNode(board.copy())
        mcts_search(root, model, simulations=simulations_per_move, c_puct=1.25, mcts_batch_size=mcts_batch_size)
        
        policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
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
        
        # Jeśli ruch nie został wybrany z księgi/bazy, wybieramy go teraz z MCTS
        if move is None:
            # Wybór temperatury na podstawie liczby ruchów
            if len(board.move_stack) < temp_threshold:
                current_temperature = 1.0  # Eksploracja
            else:
                current_temperature = 1e-4 # Eksploatacja (bliskie zera)
            move = select_action(root, current_temperature)
            
        if move is None:
            break

        # Zapisujemy parę (stan, polityka) do historii
        game_history.append((state_tensor, policy_target, board.turn))
        board.push(move)
    
    # --- Przypisanie ostatecznego wyniku partii ---
    result = board.result(claim_draw=True)
    if result == '1-0': 
        outcome = 1.0
    elif result == '0-1': 
        outcome = -1.0
    else: 
        outcome = 0.0

    # Przetwarzamy historię, aby przypisać poprawną wartość (nagrodę) do każdego stanu
    final_data = []
    for state, policy, turn in game_history:
        # Jeśli ruch wykonywały białe (turn == chess.WHITE), to wynik jest `outcome`
        # Jeśli ruch wykonywały czarne (turn == chess.BLACK), to wynik jest `-outcome`
        value = outcome if turn == chess.WHITE else -outcome
        final_data.append((state, policy, value))

    return final_data


class ReplayBuffer:
    def __init__(self, max_size=50000):
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


# Poniższe funkcje `play_single_game` i `play_match` nie są używane
# w pętli `train_alphazero_style.py`, ale mogą być przydatne do innych celów,
# więc można je zostawić lub usunąć dla przejrzystości.

def play_single_game(model_white, model_black, simulations_per_move=100):
    tb = None
    try:
        path = os.path.join(os.path.dirname(__file__), '..', 'tablebases')
        if os.path.exists(path) and any(f.endswith(('.rtbw', '.rtbz')) for f in os.listdir(path)):
            tb = chess.syzygy.open_tablebase(path)
    except Exception:
        pass

    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
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

    result = board.result(claim_draw=True)
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