import chess
import chess.pgn
import numpy as np
import os

from ai.utils import encode_board_perspective
from ai.move_mapping import MOVE_TO_INDEX, NUM_ACTIONS
from ai.mcts import flip_move

def save_chunk(output_path, states, policies, values, chunk_num):
    """Zapisuje pojedynczy kawałek danych."""
    chunk_path = output_path.replace('.npz', f'_chunk_{chunk_num}.npz')
    print(f"Zapisywanie {len(states)} pozycji do kawałka: {chunk_path}")
    np.savez_compressed(chunk_path, states=states, policies=policies, values=values)

def process_pgn_file_in_chunks(
    pgn_path: str, 
    output_path: str, 
    max_games: int = 500000, 
    chunk_size_games: int = 10000
):
    """
    Przetwarza plik PGN w kawałkach, aby uniknąć problemów z pamięcią.
    """
    print(f"Rozpoczynam przetwarzanie pliku: {pgn_path}")
    
    all_states, all_policies, all_values = [], [], []
    games_processed = 0
    chunk_num = 0

    with open(pgn_path) as pgn_file:
        while True:
            if games_processed >= max_games:
                print(f"Osiągnięto globalny limit {max_games} gier.")
                break

            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                result = game.headers.get("Result")
                if result == "1-0": outcome = 1.0
                elif result == "0-1": outcome = -1.0
                else: continue

                board = game.board()
                game_history = []

                for move in game.mainline_moves():
                    state_tensor = encode_board_perspective(board)
                    policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    
                    move_for_policy = move if board.turn == chess.WHITE else flip_move(move)
                    if move_for_policy in MOVE_TO_INDEX:
                        idx = MOVE_TO_INDEX[move_for_policy]
                        policy_target[idx] = 1.0
                    else:
                        board.push(move)
                        continue
                    
                    game_history.append((state_tensor, policy_target))
                    board.push(move)

                player_perspective_outcome = 1.0
                for state, policy in game_history:
                    value = outcome * player_perspective_outcome
                    all_states.append(state)
                    all_policies.append(policy)
                    all_values.append(value)
                    player_perspective_outcome *= -1
                
                games_processed += 1
                
                if games_processed % chunk_size_games == 0:
                    save_chunk(output_path, all_states, all_policies, all_values, chunk_num)
                    all_states, all_policies, all_values = [], [], []
                    chunk_num += 1

            except Exception as e:
                print(f"Błąd przy przetwarzaniu partii: {e}. Pomijam.")
                continue

    if all_states:
        save_chunk(output_path, all_states, all_policies, all_values, chunk_num)

    print("\nZakończono przetwarzanie i zapisywanie wszystkich kawałków.")


def merge_chunks(output_path, num_chunks):
    """Łączy wszystkie pliki-kawałki w jeden duży plik .npz."""
    print("Rozpoczynam łączenie kawałków...")
    
    all_states, all_policies, all_values = [], [], []
    
    for i in range(num_chunks):
        chunk_path = output_path.replace('.npz', f'_chunk_{i}.npz')
        print(f"Ładowanie {chunk_path}...")
        try:
            with np.load(chunk_path) as data:
                all_states.append(data['states'])
                all_policies.append(data['policies'])
                all_values.append(data['values'])
            os.remove(chunk_path)
        except FileNotFoundError:
            print(f"Ostrzeżenie: nie znaleziono pliku {chunk_path}. Pomijam.")
            continue
            
    print("Konkatenacja tablic...")
    final_states = np.concatenate(all_states, axis=0)
    final_policies = np.concatenate(all_policies, axis=0)
    final_values = np.concatenate(all_values, axis=0)
    
    print(f"Zapisywanie ostatecznego, połączonego pliku: {output_path}")
    np.savez_compressed(output_path, states=final_states, policies=final_policies, values=final_values)
    print("Zakończono łączenie. Pliki-kawałki zostały usunięte.")


if __name__ == "__main__":
    FILTERED_PGN_PATH = "lichess_final_filtered.pgn"
    OUTPUT_NPZ_PATH = "lichess_data.npz"
    MAX_GAMES_TO_PROCESS = 25000
    CHUNK_SIZE = 5000

    if not os.path.exists(FILTERED_PGN_PATH):
        print(f"BŁĄD: Plik {FILTERED_PGN_PATH} nie został znaleziony.")
    else:
        process_pgn_file_in_chunks(
            FILTERED_PGN_PATH, 
            OUTPUT_NPZ_PATH, 
            MAX_GAMES_TO_PROCESS, 
            CHUNK_SIZE
        )
        

        # num_chunks = (MAX_GAMES_TO_PROCESS + CHUNK_SIZE - 1) // CHUNK_SIZE
        # merge_chunks(OUTPUT_NPZ_PATH, num_chunks)
        print("\nPrzetwarzanie na kawałki zakończone.")
        print("Teraz należy zmodyfikować skrypt treningowy, aby wczytywał dane kawałek po kawałku.")