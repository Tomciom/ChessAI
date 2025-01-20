# ai/mcts.py

import numpy as np
import chess

from ai.utils import encode_board
from ai.move_mapping import MOVE_TO_INDEX, NUM_ACTIONS

class MCTSNode:
    def __init__(self, state: chess.Board, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # move -> child Node
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0
        self.expanded = False

    def is_terminal(self):
        return self.state.is_game_over()

    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0

    def expand(self, policy_logits):
        """
        Rozbudowujemy węzeł, tworząc dzieci dla wszystkich legalnych ruchów.
        policy_logits - wektor (NUM_ACTIONS,) wyjściowy z sieci dla tego stanu.
        """
        legal_moves = list(self.state.legal_moves)
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for move in legal_moves:
            if move in MOVE_TO_INDEX:
                idx = MOVE_TO_INDEX[move]
                mask[idx] = 1.0

        masked_policy = policy_logits * mask
        sum_policy = np.sum(masked_policy)
        if sum_policy > 1e-12:
            masked_policy /= sum_policy

        for move in legal_moves:
            if move in MOVE_TO_INDEX:
                idx = MOVE_TO_INDEX[move]
                p = masked_policy[idx]
                next_state = self.state.copy()
                next_state.push(move)
                child_node = MCTSNode(next_state, parent=self)
                child_node.prior = p
                self.children[move] = child_node

        self.expanded = True

    def update(self, value):
        self.visits += 1
        self.value_sum += value

def select_child(node: MCTSNode, c_puct=1.0):
    """
    Wybór dziecka z maksymalnym Q + U.
    Q = child.value()
    U = c_puct * child.prior * sqrt(node.visits) / (1 + child.visits)
    """
    best_score = -999999
    best_move, best_child = None, None

    sqrtN = np.sqrt(node.visits + 1e-8)
    for move, child in node.children.items():
        Q = child.value()
        U = c_puct * child.prior * sqrtN / (1 + child.visits)
        score = Q + U
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
    return best_move, best_child

def backpropagate(path, value):
    """
    Propagujemy wartość w górę. 'value' jest z perspektywy
    gracza, który jest "na ruchu" w liściu. Każde przejście w górę
    zmienia perspektywę (value = -value).
    """
    for node in reversed(path):
        node.update(value)
        value = -value

def mcts_search(root: MCTSNode, model, simulations=1600, c_puct=1.0):
    """
    Uruchamia MCTS z daną liczbą symulacji.
    """
    for _ in range(simulations):
        node = root
        search_path = [node]

        # SELECTION
        while node.expanded and not node.is_terminal():
            move, node = select_child(node, c_puct)
            search_path.append(node)

        # EVALUATION
        if node.is_terminal():
            # Stan końcowy
            result = node.state.result()
            if result == '1-0':
                value = 1
            elif result == '0-1':
                value = -1
            else:
                value = 0
            backpropagate(search_path, value)
            continue

        # Wywołujemy sieć dla stanu "node"
        inp = encode_board(node.state)[np.newaxis, ...]
        policy_logits, value_pred = model.predict(inp, verbose=0)
        policy_logits = policy_logits[0]  # (NUM_ACTIONS,)
        value_pred = value_pred[0][0]     # skalar

        # EXPANSION
        node.expand(policy_logits)

        # BACKPROP
        backpropagate(search_path, value_pred)

def select_action(root: MCTSNode, temperature=1.0):
    """
    Wybieramy ruch z korzenia zgodnie z liczbą wizyt (child.visits).
    """
    if not root.children:
        return None
    visits = np.array([child.visits for child in root.children.values()], dtype=np.float32)
    moves = list(root.children.keys())

    if temperature < 1e-6:
        # deterministycznie
        idx = np.argmax(visits)
        return moves[idx]
    else:
        # stochastycznie ~ visits^(1/T)
        visits = visits ** (1.0 / temperature)
        probs = visits / visits.sum()
        idx = np.random.choice(len(moves), p=probs)
        return moves[idx]
