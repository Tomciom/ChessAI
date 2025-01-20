# ai/move_mapping.py

import chess

############################
# Pomocnicze funkcje
############################

def square_to_rc(sq: int):
    """
    Konwersja indexu 0..63 (python-chess) na (row, col),
    przy założeniu row=0 -> rank8, row=7 -> rank1.
    """
    row = 7 - (sq // 8)
    col = sq % 8
    return (row, col)

def rc_to_square(row: int, col: int):
    """
    Odwrotna konwersja: (row, col) -> sq 0..63 (python-chess).
    """
    return (7 - row) * 8 + col


############################
# Definicje dla figur
############################

# 8 kierunków "slajdujących"
SLIDING_DIRECTIONS = [
    (1, 0),   # w dół
    (-1, 0),  # w górę
    (0, 1),   # w prawo
    (0, -1),  # w lewo
    (1, 1),   # w dół-prawo
    (1, -1),  # w dół-lewo
    (-1, 1),  # w górę-prawo
    (-1, -1)  # w górę-lewo
]

# Ruchy skoczka
KNIGHT_OFFSETS = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2)
]

PROMOTION_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

def create_action_planes():
    """
    Zwraca (MOVE_TO_INDEX, INDEX_TO_MOVE, num_actions).
    Płaszczyzny akcji obejmują:
      1. Pionek (ruch o 1, o 2, bicie lewo/prawo, + promocje tychże, + en passant)
      2. Knight moves (8 offsets)
      3. Sliding moves (8 kierunków × 7 dystansów = 56 planes)
      4. Castling (2 planes)

    Daje to >80 planes. Każdy plane ma 64 'from_sq', więc num_actions = n_planes * 64.
    """

    MOVE_TO_INDEX = {}
    INDEX_TO_MOVE = {}

    plane_index = 0

    #############
    # 1. Pionki
    #############

    # 1A. Pionek - single push (bez promocji)
    # Białe: row-1, Czarne: row+1 - rozstrzyga maska legalności
    # W stylu "AlphaZero" generujemy tak jakby "1 up" i "1 down", ale
    #  w praktyce wystarczy 1 plane i sprawdzamy legalność. Aby być czytelnym – zrobimy plane "pionek single push".
    # from_r -> to_r = from_r +/- 1 (po stronie białych to -1, czarnych +1).
    # Ale w praktyce i tak maska zadecyduje. Więc wystarczy JEDEN plane, w którym "delta_r = -1" (z perspektywy białych).
    #  ... Ale dla spójności z oryg. AZ-chess można wziąć 2 oddzielne planes (white push, black push). My uprościmy do 1.

    # Plane: Pawn single push (no promotion)
    for from_sq in range(64):
        # 'delta_r = -1' w kategoriach row=0=top => to_r = from_r - 1
        from_r, from_c = square_to_rc(from_sq)
        to_r = from_r - 1
        to_c = from_c
        if 0 <= to_r < 8 and 0 <= to_c < 8:
            to_sq = rc_to_square(to_r, to_c)
            move = chess.Move(from_sq, to_sq)
            # Sprawdzimy czy to *nie jest* promocja (tzn. to_r != 0/7 dla białych/czarnych).
            # Ale maska i tak to zweryfikuje – jedziemy dalej.
            index = plane_index * 64 + from_sq
            MOVE_TO_INDEX[move] = index
            INDEX_TO_MOVE[index] = move
    plane_index += 1

    # 1B. Pionek - double push (z pozycji wyjściowej)
    # Białe z row=6 do row=4, czarne z row=1 do row=3 (lub odwrotnie zależnie od interpretacji).
    # Znowu 1 plane "pionek double push".
    for from_sq in range(64):
        from_r, from_c = square_to_rc(from_sq)
        to_r = from_r - 2  # białe
        to_c = from_c
        if 0 <= to_r < 8:
            to_sq = rc_to_square(to_r, to_c)
            move = chess.Move(from_sq, to_sq)
            index = plane_index * 64 + from_sq
            MOVE_TO_INDEX[move] = index
            INDEX_TO_MOVE[index] = move
    plane_index += 1

    # 1C. Pionek - bicie ukośne lewo (bez promocji)
    # White: (row-1, col-1), black: (row+1, col+1). 1 plane + mask.
    for from_sq in range(64):
        from_r, from_c = square_to_rc(from_sq)
        to_r = from_r - 1
        to_c = from_c - 1
        if 0 <= to_r < 8 and 0 <= to_c < 8:
            to_sq = rc_to_square(to_r, to_c)
            move = chess.Move(from_sq, to_sq)
            index = plane_index * 64 + from_sq
            MOVE_TO_INDEX[move] = index
            INDEX_TO_MOVE[index] = move
    plane_index += 1

    # 1D. Pionek - bicie ukośne prawo (bez promocji)
    for from_sq in range(64):
        from_r, from_c = square_to_rc(from_sq)
        to_r = from_r - 1
        to_c = from_c + 1
        if 0 <= to_r < 8 and 0 <= to_c < 8:
            to_sq = rc_to_square(to_r, to_c)
            move = chess.Move(from_sq, to_sq)
            index = plane_index * 64 + from_sq
            MOVE_TO_INDEX[move] = index
            INDEX_TO_MOVE[index] = move
    plane_index += 1

    # 1E. En passant (2 planes: lewo i prawo?)
    # Często robi się 2, bo kierunek lewo/prawo. My robimy tak:
    # Plane: "en passant capture left"
    for from_sq in range(64):
        from_r, from_c = square_to_rc(from_sq)
        to_r = from_r - 1
        to_c = from_c - 1
        if 0 <= to_r < 8 and 0 <= to_c < 8:
            to_sq = rc_to_square(to_r, to_c)
            # w Move python-chess en passant to normalne Move + warunek board.is_en_passant(move)
            # ale i tak my generujemy "ten sam" from->to, maska sprawdzi, czy jest en passant legal.
            move = chess.Move(from_sq, to_sq)
            index = plane_index * 64 + from_sq
            MOVE_TO_INDEX[move] = index
            INDEX_TO_MOVE[index] = move
    plane_index += 1

    # Plane: "en passant capture right"
    for from_sq in range(64):
        from_r, from_c = square_to_rc(from_sq)
        to_r = from_r - 1
        to_c = from_c + 1
        if 0 <= to_r < 8 and 0 <= to_c < 8:
            to_sq = rc_to_square(to_r, to_c)
            move = chess.Move(from_sq, to_sq)
            index = plane_index * 64 + from_sq
            MOVE_TO_INDEX[move] = index
            INDEX_TO_MOVE[index] = move
    plane_index += 1

    # 1F. Promocje (ruch do przodu + bicie ukośne) – 4 figury promowane

    # (i) Single push z promocją
    for promo_piece in PROMOTION_PIECES:
        for from_sq in range(64):
            from_r, from_c = square_to_rc(from_sq)
            to_r = from_r - 1
            to_c = from_c
            if 0 <= to_r < 8 and 0 <= to_c < 8:
                to_sq = rc_to_square(to_r, to_c)
                move = chess.Move(from_sq, to_sq, promotion=promo_piece)
                index = plane_index * 64 + from_sq
                MOVE_TO_INDEX[move] = index
                INDEX_TO_MOVE[index] = move
        plane_index += 1

    # (ii) Bicie ukośne lewo z promocją
    for promo_piece in PROMOTION_PIECES:
        for from_sq in range(64):
            from_r, from_c = square_to_rc(from_sq)
            to_r = from_r - 1
            to_c = from_c - 1
            if 0 <= to_r < 8 and 0 <= to_c < 8:
                to_sq = rc_to_square(to_r, to_c)
                move = chess.Move(from_sq, to_sq, promotion=promo_piece)
                index = plane_index * 64 + from_sq
                MOVE_TO_INDEX[move] = index
                INDEX_TO_MOVE[index] = move
        plane_index += 1

    # (iii) Bicie ukośne prawo z promocją
    for promo_piece in PROMOTION_PIECES:
        for from_sq in range(64):
            from_r, from_c = square_to_rc(from_sq)
            to_r = from_r - 1
            to_c = from_c + 1
            if 0 <= to_r < 8 and 0 <= to_c < 8:
                to_sq = rc_to_square(to_r, to_c)
                move = chess.Move(from_sq, to_sq, promotion=promo_piece)
                index = plane_index * 64 + from_sq
                MOVE_TO_INDEX[move] = index
                INDEX_TO_MOVE[index] = move
        plane_index += 1

    #############
    # 2. Knight moves (8 offsets)
    #############
    for (dr, dc) in KNIGHT_OFFSETS:
        for from_sq in range(64):
            from_r, from_c = square_to_rc(from_sq)
            to_r = from_r + dr
            to_c = from_c + dc
            if 0 <= to_r < 8 and 0 <= to_c < 8:
                to_sq = rc_to_square(to_r, to_c)
                move = chess.Move(from_sq, to_sq)
                index = plane_index * 64 + from_sq
                MOVE_TO_INDEX[move] = index
                INDEX_TO_MOVE[index] = move
        plane_index += 1

    #############
    # 3. Sliding moves (8 kierunków × 7 pól)
    #############
    for (dr, dc) in SLIDING_DIRECTIONS:
        for steps in range(1, 8):
            for from_sq in range(64):
                from_r, from_c = square_to_rc(from_sq)
                to_r = from_r + dr*steps
                to_c = from_c + dc*steps
                if 0 <= to_r < 8 and 0 <= to_c < 8:
                    to_sq = rc_to_square(to_r, to_c)
                    move = chess.Move(from_sq, to_sq)
                    index = plane_index * 64 + from_sq
                    MOVE_TO_INDEX[move] = index
                    INDEX_TO_MOVE[index] = move
            plane_index += 1

    #############
    # 4. Castling (2 planes: O-O, O-O-O)
    #############
    # Białe O-O (e1->g1), Czarne O-O (e8->g8)
    for from_sq in [chess.E1, chess.E8]:  
        # e1 -> g1, e8 -> g8
        to_sq = from_sq + 2  # W python-chess e1=4, g1=6, e8=60, g8=62 (szczegóły sprawdź)
        move = chess.Move(from_sq, to_sq)
        index = plane_index * 64 + from_sq
        MOVE_TO_INDEX[move] = index
        INDEX_TO_MOVE[index] = move
    plane_index += 1

    # Białe O-O-O (e1->c1), Czarne O-O-O (e8->c8)
    for from_sq in [chess.E1, chess.E8]:
        # e1->c1 = 2, e8->c8=58
        to_sq = from_sq - 2
        move = chess.Move(from_sq, to_sq)
        index = plane_index * 64 + from_sq
        MOVE_TO_INDEX[move] = index
        INDEX_TO_MOVE[index] = move
    plane_index += 1

    num_planes = plane_index
    num_actions = num_planes * 64

    print(f"[move_mapping] Planes = {num_planes}, total actions = {num_actions}")

    return MOVE_TO_INDEX, INDEX_TO_MOVE, num_actions


# Generujemy mapowanie (możesz to zrobić raz w kodzie głównym)
MOVE_TO_INDEX, INDEX_TO_MOVE, NUM_ACTIONS = create_action_planes()
