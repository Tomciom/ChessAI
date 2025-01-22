import os
import numpy as np
import tensorflow as tf

from ai.model import create_chess_model, save_model, load_model
from ai.self_play import self_play_game, ReplayBuffer, play_match
from ai.move_mapping import NUM_ACTIONS

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

def train_alphazero_style(
    num_iterations=10,
    games_per_iteration=10,
    simulations_per_move=200,
    replay_buffer_max=50000,
    comparison_games=4,
    comparison_threshold=0.55,
    save_path="best_model.weights.h5"
):
    if os.path.exists(save_path):
        best_model = load_model(save_path, num_actions=NUM_ACTIONS)
    else:
        best_model = create_chess_model(num_actions=NUM_ACTIONS)

    replay_buffer = ReplayBuffer(replay_buffer_max)

    for iteration in range(num_iterations):
        print(f"\n=== Iteracja {iteration+1}/{num_iterations} ===")

        print("-> Generowanie partii self-play...")
        new_data = []
        for _ in range(games_per_iteration):
            game_data = self_play_game(
                best_model,
                simulations_per_move=simulations_per_move,
                temperature=1.0 if iteration < 5 else 0.1
            )
            new_data.extend(game_data)

        replay_buffer.add_samples(new_data)
        print(f"Aktualny rozmiar bufora: {len(replay_buffer)}")

        print("-> Tworzenie kandydata i trening...")
        candidate_model = tf.keras.models.clone_model(best_model)
        candidate_model.build((None, 8, 8, 12))
        candidate_model.set_weights(best_model.get_weights())
        candidate_model.compile(
            optimizer='adam',
            loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
            loss_weights={'policy': 1.0, 'value': 1.0}
        )

        all_data = replay_buffer.sample_all()
        X = np.array([s for (s, p, v) in all_data])
        Y_policy = np.array([p for (s, p, v) in all_data])
        Y_value = np.array([v for (s, p, v) in all_data]).reshape(-1, 1)

        candidate_model.fit(
            X,
            {"policy": Y_policy, "value": Y_value},
            epochs=5,
            batch_size=64,
            verbose=1
        )

        print("-> Porównanie kandydata z best_model...")
        score_cand, score_best = play_match(candidate_model, best_model,
                                            games=comparison_games,
                                            simulations_per_move=simulations_per_move)
        print(f"Wynik meczu: candidate={score_cand}, best={score_best} (z {comparison_games} gier)")

        total = score_cand + score_best
        ratio = score_cand / total if total > 0 else 0
        if ratio >= comparison_threshold:
            print("-> Kandydat zaakceptowany jako nowy best_model!")
            best_model.set_weights(candidate_model.get_weights())
        else:
            print("-> Kandydat odrzucony, pozostajemy przy starym best_model.")

        save_model(best_model, save_path)

    return best_model


if __name__ == "__main__":
    final_model = train_alphazero_style(
        num_iterations=25,
        games_per_iteration=5,
        simulations_per_move=25,
        replay_buffer_max=10000,
        comparison_games=4,
        comparison_threshold=0.55,
        save_path="best_model.weights.h5"
    )