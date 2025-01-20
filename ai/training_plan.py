def train_alphazero_style(
    num_iterations=10,
    games_per_iteration=10,
    simulations_per_move=200,
    replay_buffer_max=50000,
    comparison_games=4,
    comparison_threshold=0.55
):
    """
    Pętla treningowa w stylu AlphaZero.
    num_iterations: ile razy powtarzamy (self-play -> trening -> ewaluacja).
    games_per_iteration: ile partii generować w każdej iteracji (self-play).
    simulations_per_move: ile symulacji MCTS na ruch.
    replay_buffer_max: maksymalny rozmiar bufora powtórek.
    comparison_games: ile gier w meczu porównawczym nowy vs stary model.
    comparison_threshold: minimalny procent punktów, by zaakceptować nowy model.
    """

    # Tworzymy model (nasz "best_model") i replay buffer
    best_model = create_chess_model(
        input_shape=(8, 8, 12),
        num_actions=4672,
        num_filters=64,   # możesz zwiększać
        num_res_blocks=8  # możesz zwiększać
    )

    replay_buffer = ReplayBuffer(max_size=replay_buffer_max)

    for iteration in range(num_iterations):
        print(f"\n=== Iteracja {iteration+1}/{num_iterations} ===")

        # 1. Self-play
        print("-> Generowanie partii self-play...")
        new_data = []
        for g in range(games_per_iteration):
            game_data = self_play_game(
                best_model,
                simulations_per_move=simulations_per_move,
                temperature=1.0 if iteration<5 else 0.1  # np. wyższa temp. w początkowych iteracjach
            )
            new_data.extend(game_data)

        # Dodajemy do bufora
        replay_buffer.add_samples(new_data)
        print(f"Aktualny rozmiar bufora: {len(replay_buffer)}")

        # 2. Trening "kandydata" na bazie starego
        print("-> Tworzenie kandydata i trening...")
        candidate_model = tf.keras.models.clone_model(best_model)
        candidate_model.build((None, 8, 8, 12))
        candidate_model.set_weights(best_model.get_weights())
        candidate_model.compile(
            optimizer='adam',
            loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
            loss_weights={'policy': 1.0, 'value': 1.0}
        )

        # Pobieramy całą zawartość bufora
        all_data = replay_buffer.sample_all()
        X = np.array([state for (state, policy, value) in all_data])
        Y_policy = np.array([policy for (state, policy, value) in all_data])
        Y_value = np.array([value for (state, policy, value) in all_data]).reshape(-1, 1)

        # Trening (np. 1-5 epok)
        candidate_model.fit(
            X,
            {"policy": Y_policy, "value": Y_value},
            epochs=1,
            batch_size=64,
            verbose=1
        )

        # 3. Porównanie (kandydat vs best_model)
        print("-> Porównanie kandydata z dotychczasowym modelem...")
        score_candidate, score_best = play_match(candidate_model, best_model, games=comparison_games, simulations_per_move=simulations_per_move)
        print(f"Wynik meczu: candidate={score_candidate}, best={score_best} (z {comparison_games} gier)")

        # 4. Decyzja: czy aktualizować best_model?
        total = score_candidate + score_best
        if total > 0:
            ratio = score_candidate / total
        else:
            ratio = 0.0

        if ratio >= comparison_threshold:
            print("-> Kandydat zaakceptowany jako nowy best_model!")
            best_model.set_weights(candidate_model.get_weights())
        else:
            print("-> Kandydat ODRZUCONY, pozostajemy przy starym best_model.")

        # (Opcjonalnie) Zapis best_model
        best_model.save(f"best_model_iter_{iteration+1}.h5")

    return best_model


if __name__ == "__main__":
    final_model = train_alphazero_style(
        num_iterations=3,          # zwiększaj przy dłuższych eksperymentach
        games_per_iteration=5,     # docelowo 20-100+
        simulations_per_move=200,  # docelowo 800-1600+ dla lepszej jakości
        replay_buffer_max=10000,
        comparison_games=4,
        comparison_threshold=0.55
    )
