import numpy as np
import tensorflow as tf
import os
import glob
import random

from ai.model import create_chess_model
from ai.move_mapping import NUM_ACTIONS

def create_data_generator(data_dir, batch_size):
    """
    Tworzy generator, który wczytuje dane kawałek po kawałku i dostarcza je w batchach.
    """
    chunk_files = sorted(glob.glob(os.path.join(data_dir, '*_chunk_*.npz')))
    if not chunk_files:
        raise FileNotFoundError(f"Nie znaleziono plików-kawałków w folderze: {data_dir}")
    
    print(f"Znaleziono {len(chunk_files)} kawałków danych.")
    
    while True:
        random.shuffle(chunk_files)
        
        for chunk_file in chunk_files:
            try:
                with np.load(chunk_file) as data:
                    states = data['states']
                    policies = data['policies']
                    values = data['values']
                
                indices = np.arange(len(states))
                np.random.shuffle(indices)
                
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    yield states[batch_indices], {"policy": policies[batch_indices], "value": values[batch_indices]}

            except Exception as e:
                print(f"Błąd podczas wczytywania lub przetwarzania {chunk_file}: {e}")
                continue


def train_on_lichess_chunks(
    data_dir=".",
    save_path="lichess_pretrained_model.weights.h5",
    epochs=10,
    batch_size=256
):
    """
    Trenuje model w trybie nadzorowanym na danych w kawałkach.
    """
    print("Tworzenie nowego modelu do treningu nadzorowanego...")
    model = create_chess_model(num_actions=NUM_ACTIONS)

    chunk_files = sorted(glob.glob(os.path.join(data_dir, '*_chunk_*.npz')))
    if not chunk_files:
        print(f"BŁĄD: Nie znaleziono plików-kawałków w folderze: {data_dir}")
        return
        
    total_positions = 0
    for chunk_file in chunk_files:
        with np.load(chunk_file) as data:
            total_positions += len(data['states'])
            
    steps_per_epoch = total_positions // batch_size
    print(f"Łączna liczba pozycji: {total_positions}")
    print(f"Rozmiar batcha: {batch_size}")
    print(f"Liczba kroków na epokę: {steps_per_epoch}")

    train_generator = create_data_generator(data_dir, batch_size)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        save_weights_only=True,
        monitor='loss', 
        mode='min',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3, 
        verbose=1,
        restore_best_weights=True
    )

    print("\n--- Rozpoczynam trening nadzorowany na kawałkach ---")
    model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    print(f"\n--- Trening zakończony ---")
    print(f"Najlepszy model (wstępnie wytrenowany) został zapisany w pliku: {save_path}")

if __name__ == "__main__":
    DATA_DIRECTORY = "." 
    OUTPUT_WEIGHTS_FILE = "lichess_pretrained_model.weights.h5"
    EPOCHS = 20
    BATCH_SIZE = 512

    train_on_lichess_chunks(
        data_dir=DATA_DIRECTORY,
        save_path=OUTPUT_WEIGHTS_FILE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )