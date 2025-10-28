# Plik: ai/move_mapping.py
# Wersja: 6.0 GOLDEN STANDARD - Sprawdzona implementacja referencyjna

import chess

# --- Definicje ---
# 73 płaszczyzny opisujące typy ruchów
MOVE_PLANES = {}
_plane_idx = 0

# 56 płaszczyzn hetmańskich
for name, dr, dc in [("N", -1, 0), ("NE", -1, 1), ("E", 0, 1), ("SE", 1, 1),
                     ("S", 1, 0), ("SW", 1, -1), ("W", 0, -1), ("NW", -1, -1)]:
    for dist in range(1, 8):
        MOVE_PLANES[f"queen_{name}_{dist}"] = {
            'type': 'queen', 'dr': dr, 'dc': dc, 'dist': dist, 'idx': _plane_idx
        }
        _plane_idx += 1

# 8 płaszczyzn skoczkowych
for dr, dc in [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]:
    MOVE_PLANES[f"knight_{dr}_{dc}"] = {
        'type': 'knight', 'dr': dr, 'dc': dc, 'idx': _plane_idx
    }
    _plane_idx += 1

# 9 płaszczyzn pod-promocji
for piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
    for dc in [-1, 0, 1]:  # do przodu-lewo, do przodu, do przodu-prawo
        MOVE_PLANES[f"promo_{piece}_{dc}"] = {
            'type': 'promo', 'piece': piece, 'dc': dc, 'idx': _plane_idx
        }
        _plane_idx += 1

NUM_MOVE_PLANES = len(MOVE_PLANES)
NUM_ACTIONS = NUM_MOVE_PLANES * 64

# --- Generowanie Mapowań ---
INDEX_TO_MOVE = {}
MOVE_TO_INDEX = {}

def _square_to_rc(sq): return divmod(sq, 8)
def _rc_to_square(r, c): return r * 8 + c

for from_sq in range(64):
    from_r, from_c = _square_to_rc(from_sq)
    for plane in MOVE_PLANES.values():
        move = None
        move_type = plane['type']
        plane_idx = plane['idx']

        if move_type == 'queen':
            to_r = from_r + plane['dr'] * plane['dist']
            to_c = from_c + plane['dc'] * plane['dist']
            if 0 <= to_r < 8 and 0 <= to_c < 8:
                to_sq = _rc_to_square(to_r, to_c)
                if from_r == 6 and to_r == 7:  # Promocja do hetmana
                    move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                else:
                    move = chess.Move(from_sq, to_sq)
        
        elif move_type == 'knight':
            to_r = from_r + plane['dr']
            to_c = from_c + plane['dc']
            if 0 <= to_r < 8 and 0 <= to_c < 8:
                to_sq = _rc_to_square(to_r, to_c)
                move = chess.Move(from_sq, to_sq)
        
        elif move_type == 'promo':
            if from_r == 6:  # Tylko z 7. rzędu
                to_c = from_c + plane['dc']
                if 0 <= to_c < 8:
                    to_sq = _rc_to_square(7, to_c) # Promocja zawsze na 8. rząd
                    move = chess.Move(from_sq, to_sq, promotion=plane['piece'])

        if move:
            index = from_sq * NUM_MOVE_PLANES + plane_idx
            # Sprawdzamy, czy ten ruch nie został już wygenerowany przez inną płaszczyznę
            # (np. ruch pionem do przodu jest też ruchem hetmańskim)
            if move not in MOVE_TO_INDEX:
                INDEX_TO_MOVE[index] = move
                MOVE_TO_INDEX[move] = index

# --- Komunikaty Kontrolne ---
print(f"[move_mapping] Stworzono mapowanie: Płaszczyzn = {NUM_MOVE_PLANES}, Akcji = {NUM_ACTIONS}")
print(f"Rzeczywisty rozmiar słownika INDEX_TO_MOVE: {len(INDEX_TO_MOVE)}")
print(f"Rzeczywisty rozmiar słownika MOVE_TO_INDEX: {len(MOVE_TO_INDEX)}")