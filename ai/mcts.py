import numpy as np
import chess

from ai.utils import encode_board_perspective
from ai.move_mapping import MOVE_TO_INDEX, NUM_ACTIONS


def flip_square(sq: int) -> int:
    return 63 - sq

def flip_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        flip_square(move.from_square),
        flip_square(move.to_square),
        promotion=move.promotion
    )

class MCTSNode:
    def __init__(self, state: chess.Board, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0
        self.expanded = False

    def is_terminal(self):
        return self.state.is_game_over()

    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0

    def expand(self, policy_logits):
        legal_moves = list(self.state.legal_moves)
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)

        if self.state.turn == chess.WHITE:
            for move in legal_moves:
                if move in MOVE_TO_INDEX:
                    idx = MOVE_TO_INDEX[move]
                    mask[idx] = 1.0
        else:
            for real_move in legal_moves:
                flipped = flip_move(real_move)
                if flipped in MOVE_TO_INDEX:
                    idx = MOVE_TO_INDEX[flipped]
                    mask[idx] = 1.0

        masked_policy = policy_logits * mask
        sum_policy = np.sum(masked_policy)
        if sum_policy > 1e-12:
            masked_policy /= sum_policy

        if self.state.turn == chess.WHITE:
            for move in legal_moves:
                if move in MOVE_TO_INDEX:
                    idx = MOVE_TO_INDEX[move]
                    p = masked_policy[idx]
                    if p > 0:
                        next_state = self.state.copy()
                        next_state.push(move)
                        child_node = MCTSNode(next_state, parent=self)
                        child_node.prior = p
                        self.children[move] = child_node
        else:
            for real_move in legal_moves:
                flipped = flip_move(real_move)
                if flipped in MOVE_TO_INDEX:
                    idx = MOVE_TO_INDEX[flipped]
                    p = masked_policy[idx]
                    if p > 0:
                        next_state = self.state.copy()
                        next_state.push(real_move)
                        child_node = MCTSNode(next_state, parent=self)
                        child_node.prior = p
                        self.children[real_move] = child_node

        self.expanded = True

    def update(self, value):
        self.visits += 1
        self.value_sum += value


def select_child(node: MCTSNode, c_puct=1.0):
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
    for node in reversed(path):
        node.update(value)
        value = -value


def mcts_search(root: MCTSNode, model, simulations=1600, c_puct=1.0):
    for _ in range(simulations):
        node = root
        path = [node]

        while node.expanded and not node.is_terminal():
            move, next_node = select_child(node, c_puct)
            if next_node is None:
                break
            path.append(next_node)
            node = next_node

        if node.is_terminal():
            result = node.state.result()
            if result == '1-0':
                value = 1
            elif result == '0-1':
                value = -1
            else:
                value = 0
            backpropagate(path, value)
            continue

        inp = encode_board_perspective(node.state)[np.newaxis, ...]
        policy_logits, value_pred = model.predict(inp, verbose=0)
        policy_logits = policy_logits[0]
        value_pred = value_pred[0][0]

        node.expand(policy_logits)

        backpropagate(path, value_pred)

def select_action(root: MCTSNode, temperature=1.0):
    if not root.children:
        return None
    visits = np.array([child.visits for child in root.children.values()], dtype=np.float32)
    moves = list(root.children.keys())

    if temperature < 1e-6:
        idx = np.argmax(visits)
        return moves[idx]
    else:
        visits = visits ** (1.0 / temperature)
        probs = visits / visits.sum()
        idx = np.random.choice(len(moves), p=probs)
        return moves[idx]
