import numpy as np
import chess

def encode_board(board: chess.Board):
    """
    Konwertuje obiekt chess.Board na tensor (8, 8, 12).
    Można dodać kanały: castling rights, halfmove clock, itp.
    """
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        row = 7 - (square // 8)
        col = square % 8
        channel = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
        tensor[row, col, channel] = 1.0
    return tensor