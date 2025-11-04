import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np
import tensorflow as tf

tf.config.run_functions_eagerly(True)

import multiprocessing
from tqdm import tqdm
import chess
import chess.syzygy
import json
import pickle

# Ustawienie memory growth dla GPU, aby uniknąć problemów z pamięcią
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Ustawiono memory_growth=True dla {len(gpus)} GPU.")
    except RuntimeError as e:
        print(e)

# Importowanie modułów z projektu
from ai.model import create_chess_model, save_model, load_model
from ai.self_play import self_play_game, ReplayBuffer
from ai.move_mapping import NUM_ACTIONS
from ai.mcts import MCTSNode, mcts_search, select_action
from ai.knowledge import get_endgame_move, get_opening_move 

# Definicje workerów do multiprocessing
def run_self_play_worker(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf # Re-import jest często potrzebny
    model_weights, simulations_per_move, mcts_batch_size, num_actions = args
    worker_model = create_chess_model(num_actions=num_actions)
    worker_model.set_weights(model_weights)
    return self_play_game(worker_model, simulations_per_move, mcts_batch_size=mcts_batch_size)

def run_comparison_worker(args):
    model_A_weights, model_B_weights, simulations_per_move, mcts_batch_size, num_actions, a_is_white = args
    
    tb = None
    try:
        path = 'tablebases'
        if os.path.exists(path) and any(f.endswith(('.rtbw', '.rtbz')) for f in os.listdir(path)):
            tb = chess.syzygy.open_tablebase(path)
    except Exception:
        pass

    model_A = create_chess_model(num_actions=num_actions)
    model_A.set_weights(model_A_weights)
    
    model_B = create_chess_model(num_actions=num_actions)
    model_B.set_weights(model_B_weights)

    model_white, model_black = (model_A, model_B) if a_is_white else (model_B, model_A)
    
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        move = None
        
        endgame_move = get_endgame_move(board, tb)
        if endgame_move:
            move = endgame_move
        else:
            current_model = model_white if board.turn == chess.WHITE else model_black
            root = MCTSNode(board.copy())
            mcts_search(root, current_model, simulations=simulations_per_move, mcts_batch_size=mcts_batch_size)
            move = select_action(root, temperature=0.0) # W ewaluacji zawsze gramy "chciwie"

        if move is None: break
        board.push(move)

    result = board.result(claim_draw=True)
    if result == '1-0': return 1 if a_is_white else -1
    elif result == '0-1': return -1 if a_is_white else 1
    else: return 0


# Główna pętla treningowa
def train_alphazero_style(
    num_iterations=100,
    games_per_iteration=100,
    simulations_per_move=100,
    replay_buffer_max=50000,
    training_batch_size=4096,
    comparison_games=20,
    comparison_threshold=0.55,
    save_path="best_model.weights.h5",
    num_workers=max(1, multiprocessing.cpu_count() - 4),
    mcts_batch_size=8,
    pretrained_weights_path=None
):
    replay_buffer_path = "replay_buffer.pkl"
    if os.path.exists(replay_buffer_path):
        print(f"Znaleziono istniejący bufor. Wczytywanie z {replay_buffer_path}...")
        with open(replay_buffer_path, 'rb') as f:
            replay_buffer = pickle.load(f)
        print(f"Bufor wczytany. Aktualny rozmiar: {len(replay_buffer)}")
    else:
        replay_buffer = ReplayBuffer(replay_buffer_max)

    if os.path.exists(save_path):
        print(f"Kontynuuję trening. Ładowanie istniejącego modelu z {save_path}")
        best_model = load_model(save_path, num_actions=NUM_ACTIONS)
    elif pretrained_weights_path and os.path.exists(pretrained_weights_path):
        print(f"Rozpoczynam nowy trening. Ładowanie wstępnie wytrenowanego modelu z {pretrained_weights_path}")
        best_model = load_model(pretrained_weights_path, num_actions=NUM_ACTIONS)
    else:
        print("Nie znaleziono żadnych wag. Tworzenie nowego, losowego modelu.")
        best_model = create_chess_model(num_actions=NUM_ACTIONS)


    for iteration in range(num_iterations):
        print(f"\n=== Iteracja {iteration+1}/{num_iterations} ===")
        print(f"-> Generowanie {games_per_iteration} gier na {num_workers} rdzeniach...")
        
        current_weights = best_model.get_weights()
        tasks = [(current_weights, simulations_per_move, mcts_batch_size, NUM_ACTIONS) for _ in range(games_per_iteration)]
        
        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(run_self_play_worker, tasks), total=len(tasks)))

        for game_data in results:
            replay_buffer.add_samples(game_data)
        
        print(f"Gry wygenerowane. Rozmiar bufora: {len(replay_buffer)}")

        if len(replay_buffer) < training_batch_size:
            print(f"-> Za mało danych do treningu ({len(replay_buffer)}/{training_batch_size}), kontynuuję zbieranie.")
            continue

        print("-> Tworzenie kandydata i trening...")
        candidate_model = tf.keras.models.clone_model(best_model)
        candidate_model.build((None, 8, 8, 12))
        candidate_model.set_weights(best_model.get_weights())

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=num_iterations * 5,
        alpha=0.001
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Kompilujemy kandydata z tym samym optymalizatorem
        candidate_model.compile(
            optimizer=optimizer,
            loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
            loss_weights={'policy': 1.0, 'value': 1.0},
            jit_compile=False
        )

        # --- SEKCJA Z POPRAWKĄ ---
        all_data = replay_buffer.sample_all() 
        np.random.shuffle(all_data)
        training_data = all_data[:training_batch_size]
        if not training_data:
            print("Brak danych treningowych, pomijam krok treningu.")
            continue

        X = np.array([s for (s, p, v) in training_data])
        Y_policy = np.array([p for (s, p, v) in training_data])
        Y_value = np.array([v for (s, p, v) in training_data])

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X, {"policy": Y_policy, "value": Y_value})
        )

        BATCH_SIZE_FIT = 256
        train_dataset = train_dataset.shuffle(buffer_size=len(X)).batch(BATCH_SIZE_FIT)

        candidate_model.fit(
            train_dataset,
            epochs=7, # Tutaj zmieniasz liczbę epok
            verbose=1
        )
        # --- KONIEC SEKCJI Z POPRAWKĄ ---

        print(f"-> Porównanie kandydata z best_model w {comparison_games} grach...")
        
        best_weights = best_model.get_weights()
        cand_weights = candidate_model.get_weights()
        
        eval_tasks = [(cand_weights, best_weights, simulations_per_move, mcts_batch_size, NUM_ACTIONS, i % 2 == 0) for i in range(comparison_games)]
        
        with multiprocessing.Pool(num_workers) as pool:
            match_results = list(tqdm(pool.imap(run_comparison_worker, eval_tasks), total=len(eval_tasks)))
            
        score_cand = sum(1 for r in match_results if r == 1) + sum(0.5 for r in match_results if r == 0)
        total_points = comparison_games
        score_best = total_points - score_cand
        
        print(f"Wynik meczu: candidate={score_cand}, best={score_best} (z {total_points} gier)")

        ratio = score_cand / total_points if total_points > 0 else 0
        
        if ratio >= comparison_threshold:
            print(f"-> Kandydat zaakceptowany jako nowy best_model (ratio: {ratio:.2f})!")
            best_model.set_weights(candidate_model.get_weights())
            save_model(best_model, save_path)
            print(f"Nowy najlepszy model zapisany w {save_path}")
        else:
            print(f"-> Kandydat odrzucony (ratio: {ratio:.2f}).")

        print("-> Zapisywanie Replay Buffer na dysk...")
        temp_replay_buffer_path = replay_buffer_path + ".tmp"
        try:
            with open(temp_replay_buffer_path, 'wb') as f:
                pickle.dump(replay_buffer, f)
            # Jeśli zapis się udał, zamieniamy stary plik na nowy
            os.replace(temp_replay_buffer_path, replay_buffer_path)
            print("Bufor zapisany pomyślnie.")
        except Exception as e:
            print(f"BŁĄD podczas zapisywania bufora: {e}")

    return best_model


if __name__ == "__main__":
    # Niezbędne dla multiprocessing w niektórych systemach
    multiprocessing.set_start_method('spawn', force=True)

    # --- KONFIGURACJA HIPERPARAMETRÓW ---
    PRETRAINED_PATH = "lichess_pretrained_model.weights.h5" 
    SAVE_PATH = "alphazero_trained_model_v1.weights.h5"
    
    NUM_ITERATIONS = 100
    GAMES_PER_ITERATION = 120
    SIMULATIONS_PER_MOVE = 1600
    REPLAY_BUFFER_MAX = 150000
    TRAINING_BATCH_SIZE = 4096
    COMPARISON_GAMES = 24
    COMPARISON_THRESHOLD = 0.55
    MCTS_BATCH_SIZE = 64

    final_model = train_alphazero_style(
        num_iterations=NUM_ITERATIONS,
        games_per_iteration=GAMES_PER_ITERATION,
        simulations_per_move=SIMULATIONS_PER_MOVE,
        replay_buffer_max=REPLAY_BUFFER_MAX,
        training_batch_size=TRAINING_BATCH_SIZE,
        comparison_games=COMPARISON_GAMES,
        comparison_threshold=COMPARISON_THRESHOLD,
        save_path=SAVE_PATH,
        pretrained_weights_path=PRETRAINED_PATH,
        mcts_batch_size=MCTS_BATCH_SIZE
    )