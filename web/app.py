from flask import Flask, render_template, request, jsonify
import chess

app = Flask(__name__)

# Initialize a global chess board
board = chess.Board()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def handle_move():
    global board
    data = request.get_json()
    move = data.get('move')

    try:
        uci_move = chess.Move.from_uci(move.replace('-', ''))
        if uci_move in board.legal_moves:
            board.push(uci_move)  # Wykonaj ruch
            is_checkmate = board.is_checkmate()  # Sprawdź, czy nastąpił szach-mat
            response = {
                "status": "success",
                "new_fen": board.fen(),
                "checkmate": is_checkmate
            }
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
        # Convert the square (e.g., 'e2') to the internal board index
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
    board = chess.Board()  # Reset planszy do pozycji startowej
    response = {
        "status": "success",
        "fen": board.fen(),
        "currentTurn": "white" if board.turn == chess.WHITE else "black"
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
