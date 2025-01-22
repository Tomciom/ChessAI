from flask import Flask, render_template, request, jsonify
import chess

from ai.model import create_chess_model, load_model
from ai.mcts import MCTSNode, mcts_search
from ai.utils import encode_board
from ai.self_play import select_action
from ai.move_mapping import NUM_ACTIONS


app = Flask(__name__)

board = chess.Board()

try:
    model = load_model('best_model.weights.h5', num_actions=NUM_ACTIONS)
    print("Model załadowany.")
except Exception as e:
    print("Nie udało się załadować modelu:", e)
    model = create_chess_model(num_actions=NUM_ACTIONS)
    print("Utworzono nowy model.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def handle_move():
    global board
    data = request.get_json()
    move = data.get('move')
    print("Ruch odebrany od front-endu:", move)

    try:
        uci_move = chess.Move.from_uci(move.replace('-', ''))
        if uci_move in board.legal_moves:
            board.push(uci_move)
            is_checkmate = board.is_checkmate()
            response = {
                "status": "success",
                "new_fen": board.fen(),
                "checkmate": is_checkmate
            }
            print("Nowy FEN:", board.fen())
        else:
            response = {
                "status": "error",
                "message": "Illegal move or not your turn"
            }
    except Exception as e:
        response = {
            "status": "error",
            "message": str(e)
        }

    return jsonify(response)

@app.route('/possible_moves', methods=['POST'])
def possible_moves():
    global board
    data = request.get_json()
    square = data.get('square')

    try:
        square_index = chess.parse_square(square)
        moves = [move for move in board.legal_moves if move.from_square == square_index]
        response = {
            "status": "success",
            "moves": [chess.square_name(move.to_square) for move in moves]
        }
    except Exception as e:
        response = {
            "status": "error",
            "message": str(e)
        }

    return jsonify(response)

@app.route('/game_state', methods=['GET'])
def game_state():
    global board
    response = {
        "status": "success",
        "currentTurn": "white" if board.turn == chess.WHITE else "black",
        "fen": board.fen()
    }
    return jsonify(response)

@app.route('/restart', methods=['POST'])
def restart():
    global board
    board = chess.Board()
    response = {
        "status": "success",
        "fen": board.fen(),
        "currentTurn": "white" if board.turn == chess.WHITE else "black"
    }
    return jsonify(response)

@app.route('/ai_move', methods=['GET'])
def ai_move():
    """
    Endpoint wykonujący ruch AI przy użyciu MCTS i modelu TensorFlow.
    """
    global board, model
    if board.is_game_over():
        return jsonify({"status": "error", "message": "Gra już zakończona."})

    root = MCTSNode(board.copy())

    mcts_search(root, model, simulations=25, c_puct=1.0)

    best_move = select_action(root, temperature=0)

    board.push(best_move)

    response = {
        "status": "success",
        "new_fen": board.fen(),
        "ai_move": best_move.uci(),
        "checkmate": board.is_checkmate()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
