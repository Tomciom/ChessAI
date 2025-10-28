import chess

# Skopiuj DOKŁADNIE tę samą funkcję z Twojego pliku ai/mcts.py
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

# Test
print("=== Testowanie funkcji flip_move w izolacji ===")

move_e2e4 = chess.Move.from_uci("e2e4")
expected_e7e5 = chess.Move.from_uci("e7e5")
result = flip_move(expected_e7e5)

print(f"Oryginalny ruch czarnych: {expected_e7e5.uci()}")
print(f"Pole startowe: {expected_e7e5.from_square} ({chess.square_name(expected_e7e5.from_square)})")
print(f"Pole końcowe:  {expected_e7e5.to_square} ({chess.square_name(expected_e7e5.to_square)})")
print("-" * 20)
print("-" * 20)
print(f"Oczekiwany ruch białych: {move_e2e4.uci()}")
print(f"Otrzymany ruch białych:  {result.uci()}")

if result == move_e2e4:
    print("\n[ WYNIK: OK ]")
else:
    print("\n[ WYNIK: BŁĄD ]")