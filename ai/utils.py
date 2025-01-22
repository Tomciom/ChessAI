import numpy as np
import chess

def encode_board(board: chess.Board):
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        row = 7 - (square // 8)
        col = square % 8
        channel = piece.piece_type - 1
        if piece.color == chess.WHITE:
            tensor[row, col, channel] = 1.0
        else:
            tensor[row, col, channel + 6] = 1.0

    return tensor

def encode_board_perspective(board: chess.Board) -> np.ndarray:
    if board.turn == chess.WHITE:
        return encode_board(board)
    else:
        flipped_tensor = np.zeros((8, 8, 12), dtype=np.float32)

        for square, piece in board.piece_map().items():
            flipped_square = 63 - square
            new_row = 7 - (flipped_square // 8)
            new_col = flipped_square % 8

            channel = piece.piece_type - 1

            if piece.color == chess.WHITE:
                flipped_tensor[new_row, new_col, channel + 6] = 1.0
            else:
                flipped_tensor[new_row, new_col, channel] = 1.0

        return flipped_tensor