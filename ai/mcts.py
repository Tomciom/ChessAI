import numpy as np
import chess

# Importujemy istniejące komponenty
from ai.utils import encode_board_perspective
from ai.move_mapping import MOVE_TO_INDEX, NUM_ACTIONS


def flip_move(move: chess.Move) -> chess.Move:
    """
    Odwraca ruch, aby był poprawny z perspektywy przeciwnego gracza.
    Używa wbudowanej, niezawodnej funkcji z biblioteki python-chess.
    """
    return chess.Move(
        from_square=chess.square_mirror(move.from_square),
        to_square=chess.square_mirror(move.to_square),
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
        self.virtual_loss = 0

    def is_terminal(self):
        return self.state.is_game_over(claim_draw=True)

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

    sqrtN_parent = np.sqrt(node.visits + 1e-8)

    for move, child in node.children.items():
        Q = child.value() - child.virtual_loss
        U = c_puct * child.prior * sqrtN_parent / (1 + child.visits)
        score = Q + U
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
            
    return best_move, best_child


def backpropagate(path, value):
    for node in reversed(path):
        node.update(value)
        node.virtual_loss -= 1 
        value = -value


def mcts_search(root: MCTSNode, model, simulations=1600, c_puct=1.0, mcts_batch_size=8,
                add_dirichlet_noise=False, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
    
    # --- NOWA SEKCJA: DODAWANIE SZUMU DIRICHLETA ---
    if add_dirichlet_noise:
        # Rozwijamy korzeń, jeśli jeszcze nie był rozwinięty, aby stworzyć dzieci i nadać im `prior`
        if not root.expanded:
             initial_policy, _ = model.predict(np.array([encode_board_perspective(root.state)]), verbose=0)
             root.expand(initial_policy[0])
        
        moves = list(root.children.keys())
        if moves: # Upewniamy się, że są jakieś legalne ruchy
            noise = np.random.dirichlet([dirichlet_alpha] * len(moves))
            
            for i, move in enumerate(moves):
                child_node = root.children[move]
                # Mieszamy oryginalny `prior` z szumem
                child_node.prior = (1 - dirichlet_epsilon) * child_node.prior + dirichlet_epsilon * noise[i]
    # --- KONIEC NOWEJ SEKCJI ---

    # Pętla symulacji MCTS - pozostaje bez zmian
    for _ in range(simulations // mcts_batch_size):
        leaf_nodes_to_expand = []
        paths = []

        for _ in range(mcts_batch_size):
            node = root
            path = [node]

            while node.expanded and not node.is_terminal():
                move, next_node = select_child(node, c_puct)
                if next_node is None:
                    break
                node.virtual_loss += 1
                path.append(next_node)
                node = next_node
            
            if not node.is_terminal():
                leaf_nodes_to_expand.append(node)
                paths.append(path)
            else:
                result = node.state.result(claim_draw=True)
                if result == '1-0': value = 1
                elif result == '0-1': value = -1
                else: value = 0
                
                if node.state.turn == chess.BLACK:
                    value = -value
                    
                backpropagate(path, value)

        if not leaf_nodes_to_expand:
            continue

        states_batch = np.array([encode_board_perspective(n.state) for n in leaf_nodes_to_expand])
        
        if len(states_batch) == 0:
            continue
            
        policy_batch, value_batch = model.predict(states_batch, verbose=0)

        for i, leaf_node in enumerate(leaf_nodes_to_expand):
            policy_logits = policy_batch[i]
            value_pred = value_batch[i][0]
            
            leaf_node.expand(policy_logits)
            backpropagate(paths[i], value_pred)


def select_action(root: MCTSNode, temperature=1.0):
    if not root.children:
        return None
        
    moves = list(root.children.keys())
    visits = np.array([child.visits for child in root.children.values()], dtype=np.float32)

    if temperature < 1e-3:
        idx = np.argmax(visits)
        return moves[idx]
    else:
        sum_visits = visits.sum()
        if sum_visits == 0:
            idx = np.random.randint(len(moves))
            return moves[idx]
            
        probs = visits / sum_visits

        if np.isnan(probs).any():
            print(f"Ostrzeżenie: Wykryto NaN w prawdopodobieństwach (wizyty: {visits}). Wybieram ruch 'chciwie'.")
            idx = np.argmax(visits)
            return moves[idx]

        idx = np.random.choice(len(moves), p=probs)
        return moves[idx]