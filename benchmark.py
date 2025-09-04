# benchmark.py

import chess
import chess.engine
import chess.pgn
import os
import random
import json
from tqdm import tqdm
from datetime import datetime
import time

# Importujemy nasze moduły AI
from ai.model import create_chess_model, load_model
from ai.mcts import MCTSNode, mcts_search, select_action
from ai.move_mapping import NUM_ACTIONS
from ai.knowledge import get_endgame_move

# --- Konfiguracja Benchmarku ---
CONFIG = {
    "chessai_model_path_final": "alphazero_trained_model_v1.weights.h5",
    "chessai_model_path_pretrained": "lichess_pretrained_model.weights.h5",
    "stockfish_opponent_path": "./stockfish_modern_copy",
    "stockfish_oracle_path": "./stockfish_modern",
    "num_games": 10,
    "simulations_per_move": 250,
    "mcts_batch_size": 16,

    "stockfish_opponent_options": {
        "Skill Level": 0,        
        "Hash": 1,               
        "Threads": 1,           
        "EvalFile": "",        
        "EvalFileSmall": ""      
    },

    "stockfish_opponent_limit": {
        "depth": 1,         
        "nodes": 1             
    },

    "stockfish_oracle_level": {"depth": 10},
    "pgn_output_file": "match_results.pgn"
}


DEFINED_OPENINGS = {
    # 1. e4 e5 2. Nf3 Nc6 3. Bb5
    "Partia Hiszpańska (Ruy Lopez)": "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    # 1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6
    "Obrona Sycylijska (Najdorf)": "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    # 1. e4 e5 2. f4 exf4
    "Gambit Królewski (Przyjęty)": "rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR w KQkq - 0 2",
    # 1. d4 d5 2. Bf4
    "System Londyński": "rnbqkbnr/ppp1pppp/8/3p4/3P1B2/8/PPP1PPPP/RN1QKBNR b KQkq - 1 2",
    # 1. e4 c6 2. d4 d5
    "Obrona Caro-Kann": "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2"
}



def load_resources():
    print("Ładowanie zasobów...")
    model_path = CONFIG["chessai_model_path_final"]
    if not os.path.exists(model_path):
        model_path = CONFIG["chessai_model_path_pretrained"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Nie znaleziono żadnego modelu AI do testów!")
    print(f"Używam modelu AI: {model_path}")
    chessai_model = load_model(model_path, num_actions=NUM_ACTIONS)
    tablebase = None
    try:
        tb_path = 'tablebases'
        if os.path.exists(tb_path) and any(f.endswith(('.rtbw', '.rtbz')) for f in os.listdir(tb_path)):
            tablebase = chess.syzygy.open_tablebase(tb_path)
    except Exception as e: print(f"Ostrzeżenie: Nie załadowano baz zakończeń: {e}")
    print("Zasoby załadowane.")
    return chessai_model, tablebase

def get_chessai_move(board, model, tablebase):
    endgame_move = get_endgame_move(board, tablebase)
    if endgame_move: return endgame_move
    root = MCTSNode(board.copy())
    mcts_search(root, model, simulations=CONFIG["simulations_per_move"], c_puct=1.25, mcts_batch_size=CONFIG["mcts_batch_size"])
    return select_action(root, temperature=0.0)

def analyze_game_quality(game, oracle_engine):
    """
    Analizuje partię i oblicza zaawansowane metryki jakości gry dla CHESSAI.
    Przyjmuje obiekt chess.pgn.Game.
    """
    quality_metrics = {
        "top1_correct": 0, "top3_correct": 0, "total_moves": 0, "centipawn_loss_sum": 0
    }
  
    board = game.board()
    is_chessai_white = "CHESSAI" in game.headers.get("White", "")

    for node in game.mainline():
        move = node.move
        
        is_chessai_turn = (board.turn == chess.WHITE and is_chessai_white) or \
                          (board.turn == chess.BLACK and not is_chessai_white)

        if is_chessai_turn:
            try:
                analysis = oracle_engine.analyse(board, chess.engine.Limit(**CONFIG["stockfish_oracle_level"]), multipv=3)
                
                score_before_info = analysis[0].get("score")
                if score_before_info is None: continue
                
                score_before = score_before_info.white().score(mate_score=10000)
                oracle_moves = [info["pv"][0] for info in analysis if "pv" in info and info["pv"]]
                
                if move == oracle_moves[0]:
                    quality_metrics["top1_correct"] += 1
                if move in oracle_moves:
                    quality_metrics["top3_correct"] += 1
                
                board.push(move)
                
                score_after_info = oracle_engine.analyse(board, chess.engine.Limit(depth=1))["score"]
                score_after = score_after_info.white().score(mate_score=10000)
                
                board.pop()
                
                actual_loss = score_before - score_after if not is_chessai_turn else score_after - score_before 
                centipawn_loss = max(0, actual_loss)
                quality_metrics["centipawn_loss_sum"] += centipawn_loss
                
                quality_metrics["total_moves"] += 1

            except Exception as e:
                print(f"Ostrzeżenie: Błąd silnika-wyroczni lub analizy: {e}. Pomijam ruch.")
        
        if move in board.legal_moves:
            board.push(move)
        else:
            print(f"Krytyczny błąd analizy: ruch {move} jest nielegalny w pozycji {board.fen()}. Przerywam analizę tej partii.")
            break

            
    total = quality_metrics["total_moves"]
    if total > 0:
        return {
            "accuracy_top1": quality_metrics["top1_correct"] / total * 100,
            "accuracy_top3": quality_metrics["top3_correct"] / total * 100,
            "acpl": quality_metrics["centipawn_loss_sum"] / total
        }
    return {"accuracy_top1": 0.0, "accuracy_top3": 0.0, "acpl": 0.0}

def run_benchmark():
    start_time = time.time()
    chessai_model, tablebase = load_resources()

    try:
        opponent_engine = chess.engine.SimpleEngine.popen_uci(CONFIG["stockfish_opponent_path"])
        oracle_engine = chess.engine.SimpleEngine.popen_uci(CONFIG["stockfish_oracle_path"])

        print("Konfiguruję siłę gry Stockfisha-przeciwnika...")
        opponent_engine.configure(CONFIG["stockfish_opponent_options"])
        print(f"Ustawiono opcje przeciwnika: {CONFIG['stockfish_opponent_options']}")

    except Exception as e:
        print(f"BŁĄD: Problem z silnikiem Stockfish. Sprawdź ścieżki i czy obsługuje opcje UCI. Błąd: {e}")
        return

    scores = {"wins": 0, "losses": 0, "draws": 0}
    all_quality_metrics = []
    pgn_file = open(CONFIG["pgn_output_file"], "w", encoding="utf-8")
    
    opening_items = list(DEFINED_OPENINGS.items())
    games_to_play = []
    for i in range(CONFIG["num_games"]):
        opening_name, start_fen = opening_items[i % len(opening_items)]
        chessai_is_white = (i // len(opening_items)) % 2 == 0
        games_to_play.append((opening_name, start_fen, chessai_is_white))
    random.shuffle(games_to_play)

    for i, (opening_name, start_fen, chessai_is_white) in enumerate(tqdm(games_to_play, desc="Rozgrywanie partii")):
        game = chess.pgn.Game()
        game.headers["Event"] = "CHESSAI Benchmark"
        game.headers["Opening"] = opening_name
        game.headers["FEN"] = start_fen
        game.headers["White"] = "CHESSAI" if chessai_is_white else f"Stockfish (Limit: {CONFIG['stockfish_opponent_limit']})"
        game.headers["Black"] = f"Stockfish (Limit: {CONFIG['stockfish_opponent_limit']})" if chessai_is_white else "CHESSAI"

        board = chess.Board(start_fen)
        game.setup(board)
        node = game
        while not board.is_game_over():
            if (board.turn == chess.WHITE and chessai_is_white) or (board.turn == chess.BLACK and not chessai_is_white):
                move = get_chessai_move(board, chessai_model, tablebase)
            else:
                try:
                    result = opponent_engine.play(board, chess.engine.Limit(**CONFIG["stockfish_opponent_limit"]))
                    move = result.move
                except Exception as e:
                    print(f"Błąd silnika-przeciwnika: {e}. Przerywam partię.")
                    break
            if move is None: break
            node = node.add_variation(move)
            board.push(move)
            
        game.headers["Result"] = board.result()
        result_val = board.result()
        if (result_val == "1-0" and chessai_is_white) or (result_val == "0-1" and not chessai_is_white): scores["wins"] += 1
        elif (result_val == "0-1" and chessai_is_white) or (result_val == "1-0" and not chessai_is_white): scores["losses"] += 1
        else: scores["draws"] += 1
        
        print(f"\nPartia {i+1} ({opening_name}) zakończona. Analizowanie jakości gry...")
        quality = analyze_game_quality(game, oracle_engine)
        all_quality_metrics.append(quality)
        for key, val in quality.items():
            game.headers[f"CHESSAI_{key}"] = f"{val:.2f}"
        print(game, file=pgn_file, end="\n\n")

    total_time = time.time() - start_time
    print("\n\n" + "="*60)
    print(" WYNIKI MECZU (ZDEFINIOWANE OTWARCIA) ".center(60, "="))
    print("="*60)
    print(f"Liczba partii: {len(games_to_play)}")
    print(f"Przeciwnik: Stockfish (Limit: {CONFIG['stockfish_opponent_limit']})")
    print(f"Wyrocznia: Nowoczesny Stockfish (Poziom: {CONFIG['stockfish_oracle_level']})")
    print(f"Całkowity czas benchmarku: {total_time:.2f} s ({total_time/60:.2f} min)")
    print("-"*60)
    win_rate, loss_rate, draw_rate = (scores[k] / len(games_to_play) * 100 for k in ['wins', 'losses', 'draws'])
    print(f"Zwycięstwa CHESSAI: {scores['wins']:<4} ({win_rate:5.1f}%)")
    print(f"Porażki CHESSAI:   {scores['losses']:<4} ({loss_rate:5.1f}%)")
    print(f"Remisy:            {scores['draws']:<4} ({draw_rate:5.1f}%)")
    if all_quality_metrics:
        avg_acc1 = sum(q['accuracy_top1'] for q in all_quality_metrics) / len(all_quality_metrics)
        avg_acc3 = sum(q['accuracy_top3'] for q in all_quality_metrics) / len(all_quality_metrics)
        avg_acpl = sum(q['acpl'] for q in all_quality_metrics) / len(all_quality_metrics)
        print("\n--- Średnia jakość gry CHESSAI ---")
        print(f"Top-1 Accuracy: {avg_acc1:.2f}%")
        print(f"Top-3 Accuracy: {avg_acc3:.2f}%")
        print(f"Śr. utrata centypionów (ACPL): {avg_acpl:.2f}")
    print("-"*60)
    print(f"Wszystkie partie zostały zapisane do pliku: {CONFIG['pgn_output_file']}")
    print("="*60)

    opponent_engine.quit()
    oracle_engine.quit()
    pgn_file.close()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_benchmark()