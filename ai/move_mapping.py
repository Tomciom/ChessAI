import chess

def square_to_rc(sq: int):
    row = 7 - (sq // 8)
    col = sq % 8
    return (row, col)

def rc_to_square(row: int, col: int):
    return (7 - row) * 8 + col

SLIDING_DIRECTIONS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1)
]

# Ruchy skoczka
KNIGHT_OFFSETS = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2)
]

PROMOTION_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

def create_action_planes():

    MOVE_TO_INDEX = {}
    INDEX_TO_MOVE = {}

    plane_index = 0

    for from_sq in range(64):
        from_r, from_c = square_to_rc(from_sq)
        to_r = from_r - 1
        to_c = from_c
        if 0 <= to_r < 8 and 0 <= to_c < 8:
            to_sq = rc_to_square(to_r, to_c)
            move = chess.Move(from_sq, to_sq)
            index = plane_index * 64 + from_sq
            MOVE_TO_INDEX[move] = index
            INDEX_TO_MOVE[index] = move
    plane_index += 1

    for from_sq in range(64):
        from_r, from_c = square_to_rc(from_sq)
        to_r = from_r - 2
        to_c = from_c
        if 0 <= to_r < 8:
            to_sq = rc_to_square(to_r, to_c)
            move = chess.Move(from_sq, to_sq)
            index = plane_index * 64 + from_sq
            MOVE_TO_INDEX[move] = index
            INDEX_TO_MOVE[index] = move
    plane_index += 1

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

    for from_sq in [chess.E1, chess.E8]:
        to_sq = from_sq + 2
        move = chess.Move(from_sq, to_sq)
        index = plane_index * 64 + from_sq
        MOVE_TO_INDEX[move] = index
        INDEX_TO_MOVE[index] = move
    plane_index += 1

    for from_sq in [chess.E1, chess.E8]:
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

MOVE_TO_INDEX, INDEX_TO_MOVE, NUM_ACTIONS = create_action_planes()
