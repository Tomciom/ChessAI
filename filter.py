import sys
import time

# --- Konfiguracja ---
INPUT_FILENAME = 'lichess_db_standard_rated_2025-01.pgn'
OUTPUT_FILENAME = 'lichess_db_elo_2000_plus.pgn'
ELO_THRESHOLD = 2000
# --------------------

def parse_elo_from_line(line):
    """Wyciąga wartość ELO z linii tagu PGN."""
    try:
        elo_str = line.split('"')[1]
        if elo_str.isdigit():
            return int(elo_str)
    except (IndexError, ValueError):
        return 0
    return 0

def process_game_buffer(buffer):
    """Analizuje bufor z grą i decyduje, czy ją zapisać."""
    white_elo, black_elo = 0, 0
    
    # Przeszukaj tagi w poszukiwaniu ELO - wystarczy sprawdzić nagłówki
    for line in buffer:
        if line.startswith('[WhiteElo'):
            white_elo = parse_elo_from_line(line)
        elif line.startswith('[BlackElo'):
            black_elo = parse_elo_from_line(line)
        
        # Optymalizacja: jeśli oba ELO są już znalezione, przerwij pętlę
        # Jest to bezpieczne, bo ruchy nigdy nie zaczynają się od '[WhiteElo'
        if white_elo and black_elo:
            break
            
    # Sprawdź, czy obaj gracze spełniają kryterium ELO
    if white_elo > ELO_THRESHOLD and black_elo > ELO_THRESHOLD:
        return True
    
    return False

def main():
    """Główna funkcja przetwarzająca plik PGN."""
    print(f"Rozpoczynam przetwarzanie pliku: {INPUT_FILENAME}")
    print(f"Minimalne ELO dla obu graczy: {ELO_THRESHOLD}")
    print(f"Plik wyjściowy: {OUTPUT_FILENAME}")
    
    games_processed = 0
    games_written = 0
    start_time = time.time()
    
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8', errors='ignore') as infile, \
             open(OUTPUT_FILENAME, 'w', encoding='utf-8') as outfile:
            
            game_buffer = []
            for line in infile:
                # Każda partia zaczyna się od '[Event'. Jeśli napotkamy ten tag,
                # a bufor nie jest pusty, to znaczy, że właśnie skończyliśmy
                # wczytywać poprzednią partię i możemy ją przetworzyć.
                if line.startswith('[Event') and game_buffer:
                    # Przetwórz poprzednią grę znajdującą się w buforze
                    if process_game_buffer(game_buffer):
                        outfile.writelines(game_buffer)
                        # PGN wymaga pustej linii między partiami.
                        # Sprawdzamy, czy ostatnia linia w buforze już jest pusta.
                        if game_buffer[-1].strip() != "":
                           outfile.write('\n')
                        games_written += 1
                    
                    games_processed += 1
                    
                    # Wyświetlaj postęp
                    if games_processed % 100000 == 0:
                        elapsed_time = time.time() - start_time
                        print(f"Przetworzono: {games_processed} gier | Zapisano: {games_written} gier | Czas: {elapsed_time:.2f} s")
                    
                    # Zresetuj bufor i dodaj do niego pierwszą linię nowej gry
                    game_buffer = [line]
                else:
                    # Dodawaj kolejne linie do bufora
                    game_buffer.append(line)
            
            # Po zakończeniu pętli, w buforze zostanie ostatnia partia z pliku.
            # Musimy ją również przetworzyć.
            if game_buffer:
                if process_game_buffer(game_buffer):
                    outfile.writelines(game_buffer)
                    games_written += 1
                games_processed += 1

    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku '{INPUT_FILENAME}'. Upewnij się, że znajduje się on w tym samym folderze co skrypt.")
        sys.exit(1)

    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n--- Zakończono ---")
    print(f"Całkowita liczba przetworzonych partii: {games_processed}")
    print(f"Liczba partii zapisanych do pliku (ELO > {ELO_THRESHOLD}): {games_written}")
    print(f"Całkowity czas przetwarzania: {total_time:.2f} sekund ({total_time/60:.2f} minut)")

if __name__ == '__main__':
    main()