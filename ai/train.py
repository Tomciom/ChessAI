import numpy as np
from ai.model import create_chess_model, save_model, load_model
from ai.self_play import self_play_game

# Inicjalizacja lub załadowanie modelu
model = create_chess_model()

# Parametry treningu
num_iterations = 25
games_per_iteration = 5
simulations_per_move = 25

def train_model(model, num_iterations=10, games_per_iteration=5, simulations_per_move=100):
    for iteration in range(num_iterations):
        print(f"\n=== Iteracja {iteration+1} ===")
        training_examples = []

        # Generowanie partii self-play
        for _ in range(games_per_iteration):
            game_data = self_play_game(
                model,
                simulations_per_move=simulations_per_move,
                temperature=1.0
            )
            training_examples.extend(game_data)

        # Przygotowanie do treningu
        X = np.array([state for (state, policy, value) in training_examples])
        Y_policy = np.array([policy for (state, policy, value) in training_examples])
        Y_value = np.array([value for (state, policy, value) in training_examples]).reshape(-1, 1)

        # Trening
        history = model.fit(
            X, 
            {"policy": Y_policy, "value": Y_value},
            epochs=1, 
            batch_size=32
        )
