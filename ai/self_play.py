def select_action(root: MCTSNode, temperature=1.0):
    """
    Wybiera ruch z korzenia na podstawie wizyt węzłów dzieci.
    """
    moves = list(root.children.keys())
    visits = np.array([child.visits for child in root.children.values()], dtype=np.float32)
    if len(moves) == 0:
        return None  # brak dzieci, koniec gry

    if temperature < 1e-6:
        # Wybieramy ruch o największej liczbie wizyt
        return moves[np.argmax(visits)]
    else:
        # Losujemy wg rozkładu visits^(1/T)
        visits = visits ** (1.0 / temperature)
        probs = visits / visits.sum()
        return np.random.choice(moves, p=probs)


def self_play_game(model, simulations_per_move=200, temperature=1.0):
    """
    Rozgrywa 1 partię self-play, zwracając listę (stan, policy, wynik).
    """
    board = chess.Board()
    game_data = []

    while not board.is_game_over():
        # MCTS w korzeniu
        root = MCTSNode(board)
        mcts_search(root, model, simulations=simulations_per_move, c_puct=1.0)

        # Policy target = normalizujemy visits dzieci
        moves = list(root.children.keys())
        visits = np.array([child.visits for child in root.children.values()], dtype=np.float32)
        total_visits = visits.sum()

        policy_target = np.zeros(4672, dtype=np.float32)
        for move, child in root.children.items():
            idx = MOVE_TO_INDEX[move.uci()]
            policy_target[idx] = child.visits / total_visits

        # Zapis stanu (X), policy (y_policy)
        state_tensor = encode_board(board)
        game_data.append((state_tensor, policy_target))

        # Wybór ruchu
        best_move = select_action(root, temperature)
        board.push(best_move)

    # Wyznaczenie wyniku
    result = board.result()
    if result == '1-0':
        outcome = 1
    elif result == '0-1':
        outcome = -1
    else:
        outcome = 0

    # Każdy stan w danej partii ma label outcome
    final_data = [(s, p, outcome) for (s, p) in game_data]
    return final_data


from collections import deque

class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def add_samples(self, samples):
        """
        samples to lista (state, policy, value).
        """
        self.buffer.extend(samples)

    def sample_all(self):
        """
        Zwraca wszystkie dane (bez losowego próbkowania).
        Możesz też zaimplementować sample_batch(...)
        """
        return list(self.buffer)

    def __len__(self):
        return len(self.buffer)


def play_match(modelA, modelB, games=4, simulations_per_move=100):
    """
    Rozgrywa mecz pomiędzy modelA i modelB na określoną liczbę gier.
    Po połowie gier modelA gra białymi, a w pozostałych modelB gra białymi.
    Zwraca (scoreA, scoreB).
    Score to suma (1 za wygraną, 0.5 za remis, 0 za porażkę).
    """
    scoreA = 0.0
    scoreB = 0.0
    half = games // 2
    
    for i in range(half):
        # Gra i: modelA białymi, modelB czarnymi
        result = play_single_game(modelA, modelB, simulations_per_move)
        # result > 0 => biały wygrał
        # result = 0 => remis
        # result < 0 => czarny wygrał
        if result > 0:
            scoreA += 1.0
        elif result == 0:
            scoreA += 0.5
            scoreB += 0.5
        else:
            scoreB += 1.0

    for i in range(games - half):
        # Gra i: modelB białymi, modelA czarnymi
        result = play_single_game(modelB, modelA, simulations_per_move)
        # jeżeli result > 0, to biały (modelB) wygrał
        if result > 0:
            scoreB += 1.0
        elif result == 0:
            scoreA += 0.5
            scoreB += 0.5
        else:
            scoreA += 1.0

    return scoreA, scoreB

def play_single_game(model_white, model_black, simulations_per_move=100):
    """
    Rozgrywa 1 partię:
      - model_white steruje białymi,
      - model_black steruje czarnymi,
    Zwraca +1 jeśli białe wygrały, -1 jeśli czarne, 0 jeśli remis.
    """
    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            root = MCTSNode(board)
            mcts_search(root, model_white, simulations=simulations_per_move)
            move = select_action(root, temperature=0.0)  # deterministycznie
        else:
            root = MCTSNode(board)
            mcts_search(root, model_black, simulations=simulations_per_move)
            move = select_action(root, temperature=0.0)  # deterministycznie

        board.push(move)

    result = board.result()
    if result == '1-0':
        return +1
    elif result == '0-1':
        return -1
    else:
        return 0
