from flask import Flask, render_template, request, jsonify
import chess
import chess.syzygy
import os
import json

from ai.model import create_chess_model, load_model
from ai.mcts import MCTSNode, mcts_search, select_action
from ai.move_mapping import NUM_ACTIONS
# Importujemy "czyste" funkcje z knowledge
from ai.knowledge import get_endgame_move, get_opening_move

app = Flask(__name__)

board = chess.Board()

try:
    model_path = 'alphazero_trained_model_v1.weights.h5'
    if not os.path.exists(model_path):
        model_path = 'lichess_pretrained_model.weights.h5'
    
    if not os.path.exists(model_path):
        print("Nie znaleziono żadnych wytrenowanych wag. Tworzenie nowego modelu.")
        model = create_chess_model(num_actions=NUM_ACTIONS)
    else:
        print(f"Ładowanie modelu z: {model_path}")
        model = load_model(model_path, num_actions=NUM_ACTIONS)

except Exception as e:
    print(f"Nie udało się załadować modelu: {e}. Tworzenie nowego modelu.")
    model = create_chess_model(num_actions=NUM_ACTIONS)

opening_book = {}
try:
    with open(os.path.join(os.path.dirname(__file__), '..', 'ai', 'opening_book.json'), 'r') as f:
        opening_book = json.load(f)
    print("Księga otwarć załadowana pomyślnie.")
except Exception as e:
    print(f"Nie udało się załadować księgi otwarć: {e}")

tablebase = None
try:
    tb_path = os.path.join(os.path.dirname(__file__), '..', 'tablebases')
    if os.path.exists(tb_path) and any(f.endswith(('.rtbw', '.rtbz')) for f in os.listdir(tb_path)):
        tablebase = chess.syzygy.open_tablebase(tb_path)
        print("Bazy zakończeń załadowane pomyślnie.")
except Exception as e:
    print(f"Nie udało się załadować baz zakończeń: {e}")


@app.route('/')
def index():
    return render_template('index.html')

def get_game_over_status(board: chess.Board):
    """Sprawdza, czy gra jest zakończona i zwraca status."""
    if board.is_checkmate():
        winner = "Białe" if board.turn == chess.BLACK else "Czarne"
        return f"Szach-mat! Wygrały {winner}."
    if board.is_stalemate():
        return "Pat! Gra zakończona remisem."
    if board.is_insufficient_material():
        return "Niewystarczający materiał do mata. Remis."
    if board.can_claim_threefold_repetition():
        return "Trzykrotne powtórzenie pozycji. Remis."
    if board.can_claim_fifty_moves():
        return "Reguła 50 ruchów. Remis."
    return None

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
            game_over_message = get_game_over_status(board)
            response = {
                "status": "success",
                "new_fen": board.fen(),
                "game_over": game_over_message is not None,
                "game_over_message": game_over_message
            }
            print("Nowy FEN:", board.fen())
        else:
            response = {"status": "error", "message": "Illegal move or not your turn"}
    except Exception as e:
        response = {"status": "error", "message": str(e)}
    return jsonify(response)

@app.route('/ai_move', methods=['GET'])
def ai_move():
    global board, model, opening_book, tablebase
    if board.is_game_over():
        return jsonify({"status": "error", "message": "Gra już zakończona."})

    best_move = None
    
    opening_move = get_opening_move(board, opening_book)
    if opening_move:
        print("[AI] Ruch znaleziony w księdze otwarć.")
        best_move = opening_move
    
    if best_move is None:
        endgame_move = get_endgame_move(board, tablebase)
        if endgame_move:
            print("[AI] Ruch znaleziony w bazie zakończeń.")
            best_move = endgame_move
    
    if best_move is None:
        print("[AI] Uruchamiam przeszukiwanie MCTS...")
        root = MCTSNode(board.copy())
        mcts_search(root, model, simulations=800, c_puct=2.5, mcts_batch_size=16) 
        best_move = select_action(root, temperature=0)

    if best_move is None:
        return jsonify({"status": "error", "message": "Nie udało się znaleźć ruchu."})

    board.push(best_move)

    game_over_message = get_game_over_status(board)
    response = {
        "status": "success",
        "new_fen": board.fen(),
        "ai_move": best_move.uci(),
        "game_over": game_over_message is not None,
        "game_over_message": game_over_message
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
        response = {"status": "success", "moves": [chess.square_name(move.to_square) for move in moves]}
    except Exception as e:
        response = {"status": "error", "message": str(e)}
    return jsonify(response)

@app.route('/game_state', methods=['GET'])
def game_state():
    global board
    response = {"status": "success", "currentTurn": "white" if board.turn == chess.WHITE else "black", "fen": board.fen()}
    return jsonify(response)

@app.route('/restart', methods=['POST'])
def restart():
    global board
    board = chess.Board()
    response = {"status": "success", "fen": board.fen(), "currentTurn": "white" if board.turn == chess.WHITE else "black"}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
