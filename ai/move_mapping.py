import chess

MOVE_TO_INDEX = {}
INDEX_TO_MOVE = {}

def create_move_mapping():
    index = 0
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            # Ruch bez promocji
            if from_sq != to_sq:
                move = chess.Move(from_sq, to_sq)
                MOVE_TO_INDEX[move.uci()] = index
                INDEX_TO_MOVE[index] = move
                index += 1
            # Ruchy z promocją
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                MOVE_TO_INDEX[promo_move.uci()] = index
                INDEX_TO_MOVE[index] = promo_move
                index += 1

create_move_mapping()
