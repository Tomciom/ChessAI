import chess

class MoveGenerator:
    def __init__(self, board):
        """
        Initialize the move generator with a chess board.
        
        Args:
            board (chess.Board): The current state of the chess board.
        """
        self.board = board

    def get_legal_moves(self):
        """
        Get all legal moves for the current board position.
        
        Returns:
            list: A list of moves in UCI format.
        """
        return [move.uci() for move in self.board.legal_moves]

    def get_legal_captures(self):
        """
        Get all legal capture moves for the current board position.
        
        Returns:
            list: A list of capture moves in UCI format.
        """
        return [move.uci() for move in self.board.legal_moves if self.board.is_capture(move)]

    def get_checks(self):
        """
        Get all moves that put the opponent in check.
        
        Returns:
            list: A list of moves in UCI format that result in a check.
        """
        checks = []
        for move in self.board.legal_moves:
            self.board.push(move)
            if self.board.is_check():
                checks.append(move.uci())
            self.board.pop()
        return checks

    def get_promotion_moves(self):
        """
        Get all legal pawn promotion moves.
        
        Returns:
            list: A list of promotion moves in UCI format.
        """
        return [move.uci() for move in self.board.legal_moves if move.promotion]

    def get_best_move(self):
        """
        Get the best move based on a simple evaluation (material balance).
        
        Returns:
            str: The best move in UCI format.
        """
        best_move = None
        best_score = -float('inf')

        for move in self.board.legal_moves:
            self.board.push(move)
            score = self.evaluate_material_balance()
            self.board.pop()

            if score > best_score:
                best_score = score
                best_move = move.uci()

        return best_move

    def evaluate_material_balance(self):
        """
        Simple evaluation of the board based on material balance.
        
        Returns:
            int: The material score of the current position.
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King is not evaluated
        }

        score = 0
        for piece_type in piece_values:
            score += len(self.board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            score -= len(self.board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

        return score


if __name__ == "__main__":
    # Example usage
    board = chess.Board()
    generator = MoveGenerator(board)

    print("Legal moves:", generator.get_legal_moves())
    print("Legal captures:", generator.get_legal_captures())
    print("Checks:", generator.get_checks())
    print("Promotion moves:", generator.get_promotion_moves())
    print("Best move (simple evaluation):", generator.get_best_move())
