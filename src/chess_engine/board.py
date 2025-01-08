import chess
import chess.pgn
from io import StringIO

class ChessBoard:
    def __init__(self):
        """Initialize a new chess board and an empty game history."""
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.node = self.game

    def reset_board(self):
        """Reset the board to the initial state and clear game history."""
        self.board.reset()
        self.game = chess.pgn.Game()
        self.node = self.game

    def get_legal_moves(self):
        """Return a list of all legal moves in the current position."""
        return list(self.board.legal_moves)

    def make_move(self, move):
        """
        Make a move on the board.
        
        Args:
            move (str): The move in UCI format (e.g., 'e2e4').
            
        Returns:
            bool: True if the move was successful, False otherwise.
        """
        try:
            uci_move = chess.Move.from_uci(move)
            if uci_move in self.board.legal_moves:
                self.board.push(uci_move)
                self.node = self.node.add_variation(uci_move)
                return True
            else:
                print(f"Illegal move: {move}")
                return False
        except ValueError:
            print(f"Invalid move format: {move}")
            return False

    def undo_move(self):
        """Undo the last move."""
        if self.board.move_stack:
            self.board.pop()
            self.node = self.node.parent
        else:
            print("No moves to undo.")

    def is_game_over(self):
        """Check if the game is over."""
        return self.board.is_game_over()

    def get_result(self):
        """Get the result of the game if it's over."""
        if self.board.is_checkmate():
            return "Checkmate"
        elif self.board.is_stalemate():
            return "Stalemate"
        elif self.board.is_insufficient_material():
            return "Draw (Insufficient Material)"
        elif self.board.is_seventyfive_moves():
            return "Draw (75-move rule)"
        elif self.board.is_fivefold_repetition():
            return "Draw (Fivefold Repetition)"
        return "Ongoing"

    def save_game_to_pgn(self, filename="game.pgn"):
        """
        Save the game to a PGN file.
        
        Args:
            filename (str): The name of the file to save.
        """
        with open(filename, "w") as f:
            f.write(str(self.game))
        print(f"Game saved to {filename}")

    def display_board(self):
        """Print the board in ASCII format."""
        print(self.board)

    def get_pgn_string(self):
        """Return the current game as a PGN string."""
        return str(self.game)
    
    def get_fen(self):
        return self.board.fen()


if __name__ == "__main__":
    # Example usage
    chess_board = ChessBoard()
    chess_board.display_board()
    chess_board.make_move("e2e4")
    chess_board.make_move("e7e5")
    chess_board.make_move("g1f3")
    chess_board.display_board()
    print("\nPGN output:")
    print(chess_board.get_pgn_string())
    chess_board.save_game_to_pgn()
