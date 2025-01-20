class MCTSNode:
    def __init__(self, state: chess.Board, parent=None):
        self.state = state.copy()
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = 0.0

    def is_terminal(self):
        return self.state.is_game_over()

    def is_fully_expanded(self):
        legal_moves = list(self.state.legal_moves)
        return len(self.children) == len(legal_moves)

    def expand(self, move, next_state, prior):
        child_node = MCTSNode(next_state, parent=self)
        child_node.prior = prior
        self.children[move] = child_node
        return child_node

    def update(self, value):
        self.visits += 1
        self.value_sum += value

    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

def select_child(node: MCTSNode, c_puct=1.0):
    """
    PUCT formula: Q + c_puct * P * sqrt(N) / (1 + n)
      Q = child.value()
      P = child.prior
      N = node.visits
      n = child.visits
    """
    best_score = -float('inf')
    best_move = None
    best_child = None

    for move, child in node.children.items():
        Q = child.value()
        P = child.prior
        N = node.visits
        n = child.visits

        uct = Q + c_puct * P * np.sqrt(N) / (1 + n)
        if uct > best_score:
            best_score = uct
            best_move = move
            best_child = child

    return best_move, best_child


def mcts_search(root: MCTSNode, model, simulations=200, c_puct=1.0):
    """
    Wykonuje MCTS z zadaną liczbą symulacji.
    """
    for _ in range(simulations):
        node = root
        path = [node]

        # 1. Selekcja
        while node.is_fully_expanded() and not node.is_terminal():
            move, node = select_child(node, c_puct)
            path.append(node)

        # 2. Ekspansja
        if not node.is_terminal():
            legal_moves = list(node.state.legal_moves)
            for move in legal_moves:
                if move not in node.children:
                    next_state = node.state.copy()
                    next_state.push(move)
                    # Predykcja sieci (policy, value) dla next_state
                    input_tensor = encode_board(next_state)[np.newaxis, ...]
                    policy_pred, value_pred = model.predict(input_tensor, verbose=0)
                    policy_pred = policy_pred[0]  # (4672,)
                    value_pred = value_pred[0][0]

                    # Maskowanie nielegalnych z perspektywy next_state
                    next_legal_moves = list(next_state.legal_moves)
                    mask = np.zeros_like(policy_pred)
                    for lm in next_legal_moves:
                        idx = MOVE_TO_INDEX[lm.uci()]
                        mask[idx] = 1.0
                    masked_policy = policy_pred * mask
                    # Normalizacja
                    sum_policy = np.sum(masked_policy)
                    if sum_policy > 0:
                        masked_policy /= sum_policy

                    move_index = MOVE_TO_INDEX[move.uci()]
                    prior = masked_policy[move_index]
                    child = node.expand(move, next_state, prior)
                    
                    # Ocena węzła (dla tego jednego dziecka)
                    node = child
                    path.append(node)
                    value = value_pred
                    break
        else:
            # Stan terminalny
            result = node.state.result()
            if result == '1-0':
                value = 1
            elif result == '0-1':
                value = -1
            else:
                value = 0

        # 3. Backpropagation (aktualizacja)
        for n in reversed(path):
            n.update(value)
            value = -value  # zmiana perspektywy (przeciwnik)
