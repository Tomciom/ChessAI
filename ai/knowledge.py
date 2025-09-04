import chess
import chess.syzygy
import os
import json
import random



def get_endgame_move(board: chess.Board, tablebase: chess.syzygy.Tablebase) -> chess.Move | None:
    """
    Sprawdza bazę zakończeń używając przekazanego obiektu tablebase.
    """
    if tablebase is None:
        return None
    
    try:
        dtz = tablebase.probe_dtz(board)

        best_move = None
        min_abs_wdl = 2

        for move in board.legal_moves:
            board.push(move)
            wdl = tablebase.probe_wdl(board)
            board.pop()

            if abs(wdl) < min_abs_wdl:
                 min_abs_wdl = abs(wdl)
                 best_move = move
        
        return best_move

    except (KeyError, IndexError):
        return None

def get_opening_move(board: chess.Board, opening_book: dict) -> chess.Move | None:
    """
    Sprawdza księgę otwarć używając przekazanego słownika.
    """
    if opening_book is None:
        return None
        
    fen_key = board.fen()
    
    if fen_key in opening_book:
        recommended_uci_moves = opening_book[fen_key]
        legal_moves = [chess.Move.from_uci(m) for m in recommended_uci_moves if chess.Move.from_uci(m) in board.legal_moves]
        
        if legal_moves:
            return random.choice(legal_moves)
            
    return None