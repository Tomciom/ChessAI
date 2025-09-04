import chess

QUEEN_DIRECTIONS = [(dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if not (dr == 0 and dc == 0)] 
KNIGHT_DIRECTIONS = [(dr, dc) for dr in [-2, -1, 1, 2] for dc in [-2, -1, 1, 2] if abs(dr) != abs(dc)] 

NUM_QUEEN_MOVES = 7 * len(QUEEN_DIRECTIONS)
NUM_KNIGHT_MOVES = len(KNIGHT_DIRECTIONS)
NUM_PROMOTIONS = 3 * 3 

NUM_PLANES = NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES + NUM_PROMOTIONS 
NUM_ACTIONS = NUM_PLANES * 64 

MOVE_TO_INDEX = {}
INDEX_TO_MOVE = {}

def _rc_to_square(r, c):
    return (7 - r) * 8 + c

def _square_to_rc(sq):
    return 7 - (sq // 8), sq % 8

plane = 0
for dr, dc in QUEEN_DIRECTIONS:
    for length in range(1, 8):
        for from_sq in range(64):
            r, c = _square_to_rc(from_sq)
            to_r, to_c = r + dr * length, c + dc * length
            if 0 <= to_r < 8 and 0 <= to_c < 8:
                to_sq = _rc_to_square(to_r, to_c)
                index = plane * 64 + from_sq
                
                move = chess.Move(from_sq, to_sq)
                MOVE_TO_INDEX[move] = index
                INDEX_TO_MOVE[index] = move

                if chess.square_rank(from_sq) == 6 and chess.square_rank(to_sq) == 7:
                     promo_move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                     MOVE_TO_INDEX[promo_move] = index

        plane += 1

for dr, dc in KNIGHT_DIRECTIONS:
    for from_sq in range(64):
        r, c = _square_to_rc(from_sq)
        to_r, to_c = r + dr, c + dc
        if 0 <= to_r < 8 and 0 <= to_c < 8:
            to_sq = _rc_to_square(to_r, to_c)
            index = plane * 64 + from_sq
            move = chess.Move(from_sq, to_sq)
            MOVE_TO_INDEX[move] = index
            INDEX_TO_MOVE[index] = move
    plane += 1

PROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
PROMO_DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1)] 

for piece in PROMOTION_PIECES:
    for dr, dc in PROMO_DIRECTIONS:
         for from_sq in range(8, 16):
            r, c = _square_to_rc(from_sq)
            to_r, to_c = r + dr, c + dc
            if 0 <= to_c < 8:
                to_sq = _rc_to_square(to_r, to_c)
                index = plane * 64 + from_sq
                move = chess.Move(from_sq, to_sq, promotion=piece)
                MOVE_TO_INDEX[move] = index
                INDEX_TO_MOVE[index] = move
         plane += 1

print(f"[move_mapping] Stworzono mapowanie: Płaszczyzn = {NUM_PLANES}, Akcji = {NUM_ACTIONS}")
print(f"Rzeczywisty rozmiar słownika MOVE_TO_INDEX: {len(MOVE_TO_INDEX)}")