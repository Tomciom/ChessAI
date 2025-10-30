# Plik: test_integration.py
# Cel: Weryfikacja poprawności współpracy kluczowych modułów AI po naprawie błędów.

import numpy as np
import tensorflow as tf
import chess

# Importujemy wszystkie niezbędne komponenty naszego silnika
from ai.model import create_chess_model
from ai.mcts import MCTSNode, mcts_search, select_action
from ai.move_mapping import MOVE_TO_INDEX, NUM_ACTIONS

print("=== ROZPOCZYNAM TESTY INTEGRACYJNE RDZENIA AI ===")

# --- TEST 1: Test Dymny (Smoke Test) ---
print("\n--- [TEST 1/3] Uruchamiam Test Dymny... ---")
# Cel: Sprawdzić, czy system potrafi wygenerować jakikolwiek legalny ruch bez awarii.
# Używamy losowego, nietrenowanego modelu.

try:
    board = chess.Board()
    # Tworzymy model z losowymi wagami
    random_model = create_chess_model(num_actions=NUM_ACTIONS)
    
    # Uruchamiamy MCTS na bardzo małą liczbę symulacji
    root = MCTSNode(board.copy())
    mcts_search(root, random_model, simulations=16)
    
    # Wybieramy ruch
    best_move = select_action(root, temperature=0.0)
    
    if best_move in board.legal_moves:
        print(f"[WYNIK TESTU 1] SUKCES: Wygenerowano legalny ruch: {best_move.uci()}")
    else:
        print(f"[WYNIK TESTU 1] PORAŻKA: Wygenerowany ruch {best_move} nie jest legalny!")

except Exception as e:
    print(f"[WYNIK TESTU 1] KRYTYCZNA PORAŻKA: Wystąpił błąd podczas generowania ruchu: {e}")
    # W przypadku błędu, dalsze testy nie mają sensu, więc przerywamy.
    exit()

# --- TEST 2: Test Spójności Logicznej (Test "Sztucznego Mózgu") ---
print("\n--- [TEST 2/3] Uruchamiam Test Spójności Logicznej... ---")
# Cel: Sprawdzić, czy jednoznaczna sugestia sieci jest poprawnie interpretowana.
# Tworzymy "sztuczny mózg", który zawsze chce zagrać jeden, konkretny ruch.

try:
    # Krok 1: Test dla ruchu białych (e2e4)
    board_white = chess.Board()
    move_to_force = chess.Move.from_uci("e2e4")
    index_to_force = MOVE_TO_INDEX[move_to_force]
    
    # Tworzymy "sztuczną" politykę: 100% prawdopodobieństwa na wybranym indeksie
    dummy_policy_white = np.zeros(NUM_ACTIONS, dtype=np.float32)
    dummy_policy_white[index_to_force] = 1.0
    dummy_value_white = np.array([[0.0]], dtype=np.float32)

    # Tworzymy fałszywy model, który zawsze zwraca naszą "sztuczną" politykę
    from tensorflow.keras.layers import Lambda

    # Tworzymy fałszywy model, który dynamicznie obsługuje rozmiar batcha
    inputs = tf.keras.Input(shape=(8, 8, 12), name="board_input")

    # Ta funkcja lambda bierze batch wejściowy `x` i zwraca stałą politykę
    # powieloną tyle razy, ile wynosi rozmiar batcha `x`.
    def create_dummy_policy(x):
        batch_size = tf.shape(x)[0]
        return tf.repeat(tf.constant(dummy_policy_white[np.newaxis, :]), repeats=batch_size, axis=0)

    def create_dummy_value(x):
        batch_size = tf.shape(x)[0]
        return tf.repeat(tf.constant(dummy_value_white), repeats=batch_size, axis=0)

    policy_output = Lambda(create_dummy_policy, name="policy_output")(inputs)
    value_output = Lambda(create_dummy_value, name="value_output")(inputs)
    dummy_model = tf.keras.Model(inputs=inputs, outputs=[policy_output, value_output])
    # Uruchamiamy MCTS
    root_white = MCTSNode(board_white.copy())
    mcts_search(root_white, dummy_model, simulations=16)
    result_move_white = select_action(root_white, temperature=0.0)

    if result_move_white == move_to_force:
        print(f"[WYNIK TESTU 2A - BIAŁE] SUKCES: 'Sztuczny mózg' poprawnie wybrał ruch {move_to_force.uci()}")
    else:
        print(f"[WYNIK TESTU 2A - BIAŁE] PORAŻKA: Oczekiwano {move_to_force.uci()}, otrzymano {result_move_white.uci()}")


    # Krok 2: Test dla ruchu czarnych (e7e5), weryfikujący `flip_move`
    board_black = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    move_to_force_black = chess.Move.from_uci("e7e5")
    # Ważne: sieć ZAWSZE myśli z perspektywy białych, więc nadal "sugeruje" e2e4!
    # Nasz system musi poprawnie przetłumaczyć tę sugestię z powrotem na e7e5.
    
    root_black = MCTSNode(board_black.copy())
    mcts_search(root_black, dummy_model, simulations=16)
    result_move_black = select_action(root_black, temperature=0.0)

    if result_move_black == move_to_force_black:
        print(f"[WYNIK TESTU 2B - CZARNE] SUKCES: 'Sztuczny mózg' i `flip_move` poprawnie wybrały ruch {move_to_force_black.uci()}")
    else:
        print(f"[WYNIK TESTU 2B - CZARNE] PORAŻKA: Oczekiwano {move_to_force_black.uci()}, otrzymano {result_move_black.uci()}")

except Exception as e:
    print(f"[WYNIK TESTU 2] KRYTYCZNA PORAŻKA: Wystąpił błąd podczas testu spójności: {e}")
    exit()

# --- TEST 3: Test Pełnej Rozgrywki ---
print("\n--- [TEST 3/3] Uruchamiam Test Pełnej Rozgrywki... ---")
# Cel: Sprawdzić, czy system jest stabilny i potrafi rozegrać całą partię.

try:
    board = chess.Board()
    game_moves = 0
    # Używamy tego samego losowego modelu z Testu 1
    
    while not board.is_game_over(claim_draw=True) and game_moves < 150: # Ogranicznik na 150 ruchów
        root = MCTSNode(board.copy())
        mcts_search(root, random_model, simulations=16) # Mało symulacji, żeby było szybko
        move = select_action(root, temperature=1.0) # Temperatura > 0, żeby gra była losowa
        
        if move is None or move not in board.legal_moves:
             raise ValueError(f"Silnik wygenerował nieprawidłowy ruch: {move} w pozycji {board.fen()}")
        
        board.push(move)
        game_moves += 1
    
    print(f"[WYNIK TESTU 3] SUKCES: Partia zakończyła się bez błędów po {game_moves} ruchach. Wynik: {board.result(claim_draw=True)}")

except Exception as e:
    print(f"[WYNIK TESTU 3] KRYTYCZNA PORAŻKA: Wystąpił błąd podczas rozgrywki: {e}")

print("\n=== ZAKOŃCZONO TESTY INTEGRACYJNE ===")