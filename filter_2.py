import sys
import time

# --- Konfiguracja ---
# CZYTAMY Z PLIKU, KTÓRY WŁAŚNIE TWORZYSZ
INPUT_FILENAME = 'lichess_db_elo_2000_plus.pgn' 
# TWORZYMY OSTATECZNY, WYSOKIEJ JAKOŚCI PLIK
OUTPUT_FILENAME = './ChessAI/lichess_final_filtered.pgn'
ELO_THRESHOLD = 2300  # Podnosimy próg ELO
MIN_MOVES_THRESHOLD = 20 # Minimum 20 pół-ruchów
# --------------------

# Funkcja parse_elo_from_line pozostaje bez zmian
def parse_elo_from_line(line):
    try:
        elo_str = line.split('"')[1]
        if elo_str.isdigit():
            return int(elo_str)
    except (IndexError, ValueError):
        return 0
    return 0

# Funkcja process_game_buffer z nowymi warunkami
def process_game_buffer(buffer):
    white_elo, black_elo = 0, 0
    move_count = 0
    
    # Sprawdzamy też, czy gra nie została przerwana
    termination = ""

    for line in buffer:
        if line.startswith('[WhiteElo'):
            white_elo = parse_elo_from_line(line)
        elif line.startswith('[BlackElo'):
            black_elo = parse_elo_from_line(line)
        elif line.startswith('[Termination'):
            termination = line.split('"')[1]
        elif line.strip() and line.strip()[0].isdigit():
            move_count += line.count(' ')
            
    # Sprawdź wszystkie warunki
    if (white_elo > ELO_THRESHOLD and 
        black_elo > ELO_THRESHOLD and
        move_count >= MIN_MOVES_THRESHOLD and
        termination != "Abandoned"): # Ignorujemy przerwane partie
        return True
    
    return False

# Funkcja main pozostaje praktycznie taka sama
def main():
    print(f"Rozpoczynam drugi etap filtrowania pliku: {INPUT_FILENAME}")
    print(f"Nowe kryteria: ELO > {ELO_THRESHOLD}, Ruchy >= {MIN_MOVES_THRESHOLD}")
    print(f"Plik wyjściowy: {OUTPUT_FILENAME}")
    
    # ... (reszta funkcji main jest identyczna jak w Twoim skrypcie)
    # ... (skopiuj ją z Twojego oryginalnego skryptu)
    games_processed = 0
    games_written = 0
    start_time = time.time()
    
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8', errors='ignore') as infile, \
             open(OUTPUT_FILENAME, 'w', encoding='utf-8') as outfile:
            
            game_buffer = []
            for line in infile:
                if line.startswith('[Event') and game_buffer:
                    if process_game_buffer(game_buffer):
                        outfile.writelines(game_buffer)
                        outfile.write('\n')
                        games_written += 1
                    
                    games_processed += 1
                    
                    if games_processed % 100000 == 0:
                        elapsed_time = time.time() - start_time
                        print(f"Przetworzono: {games_processed} gier | Zapisano: {games_written} gier | Czas: {elapsed_time:.2f} s")
                    
                    game_buffer = [line]
                else:
                    game_buffer.append(line)
            
            if game_buffer:
                if process_game_buffer(game_buffer):
                    outfile.writelines(game_buffer)
                    games_written += 1
                games_processed += 1

    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku '{INPUT_FILENAME}'.")
        sys.exit(1)

    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n--- Zakończono drugi etap filtrowania ---")
    print(f"Całkowita liczba przetworzonych partii: {games_processed}")
    print(f"Liczba partii zapisanych do pliku finalnego: {games_written}")
    print(f"Całkowity czas przetwarzania: {total_time:.2f} sekund")


if __name__ == '__main__':
    main()