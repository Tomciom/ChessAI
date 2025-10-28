import chess
# Poprawione importy: importujemy mapowania z jednego pliku, a funkcję pomocniczą z drugiego
from ai.move_mapping import MOVE_TO_INDEX, INDEX_TO_MOVE
from ai.mcts import flip_move

# --- Zestaw Testowy ---
moves_to_test = [
    'e2e4', 'd7d5', 'e4d5', 'g1f3', 'b1c3', 'f1c4',
    'a1b1', 'd1e2', 'h1h8', 'a8h8', 'e7e8q', 'a7a8r',
    'b7b8b', 'c7c8n'
]

# --- Test 1: Integralność Mapowania (Round-trip) ---
print("=== Rozpoczynam Test Integralności Mapowania Ruchów ===")
print("-" * 60)

successful_tests = 0
failed_tests = 0
missing_in_move_to_index = 0
missing_in_index_to_move = 0

for uci_move in moves_to_test:
    try:
        original_move = chess.Move.from_uci(uci_move)

        if original_move not in MOVE_TO_INDEX:
            print(f"[BŁĄD] Ruch {original_move.uci():<7} nie istnieje w słowniku MOVE_TO_INDEX!")
            missing_in_move_to_index += 1
            failed_tests += 1
            continue

        index = MOVE_TO_INDEX[original_move]

        if index not in INDEX_TO_MOVE:
            print(f"[BŁĄD] Indeks {index:<5} (z ruchu {original_move.uci()}) nie istnieje w słowniku INDEX_TO_MOVE!")
            missing_in_index_to_move += 1
            failed_tests += 1
            continue
            
        returned_move = INDEX_TO_MOVE[index]

        if original_move == returned_move:
            print(f"[ OK ] {original_move.uci():<7} -> Indeks: {index:<5} -> {returned_move.uci():<7}")
            successful_tests += 1
        else:
            print(f"[BŁĄD] {original_move.uci():<7} -> Indeks: {index:<5} -> Otrzymano: {returned_move.uci():<7} (NIEPOPRAWNIE!)")
            failed_tests += 1

    except Exception as e:
        print(f"[KRYTYCZNY BŁĄD] Wyjątek przy przetwarzaniu ruchu {uci_move}: {e}")
        failed_tests += 1

print("-" * 60)
print(f"Wynik Testu 1: {successful_tests} udanych, {failed_tests} nieudanych.")
if missing_in_move_to_index > 0:
    print(f"Brakujących ruchów w MOVE_TO_INDEX: {missing_in_move_to_index}")
if missing_in_index_to_move > 0:
    print(f"Brakujących indeksów w INDEX_TO_MOVE: {missing_in_index_to_move}")
print("=" * 60)


# --- Test 2: Poprawność Odwracania Ruchów (flip_move) ---
print("\n=== Rozpoczynam Test Funkcji flip_move ===")
print("-" * 60)
flip_success = 0
flip_failed = 0

tests_for_flip = {
    'e2e4': 'e7e5',
    'g1f3': 'g8f6',
    'a1b1': 'a8b8',
    'h7h8q': 'h2h1q'
}

for white_uci, black_uci in tests_for_flip.items():
    white_move = chess.Move.from_uci(white_uci)
    expected_black_move = chess.Move.from_uci(black_uci)
    
    # Odwracamy ruch czarnych, powinniśmy dostać ruch białych
    flipped_from_black = flip_move(expected_black_move)

    if flipped_from_black == white_move:
        print(f"[ OK ] flip_move({expected_black_move.uci()}) == {white_move.uci()}")
        flip_success += 1
    else:
        print(f"[BŁĄD] flip_move({expected_black_move.uci()}) zwróciło {flipped_from_black.uci()} zamiast {white_move.uci()}")
        flip_failed += 1

print("-" * 60)
print(f"Wynik Testu 2: {flip_success} udanych, {flip_failed} nieudanych.")
print("=" * 60)