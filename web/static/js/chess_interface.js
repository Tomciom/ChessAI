document.addEventListener("DOMContentLoaded", function () {
    console.log("chess_interface.js został załadowany");

    let currentTurn;
    let selectedSquare = null;
    let possibleMoves = [];

    // Inicjalizacja szachownicy
    const board = Chessboard('board1', {
        position: 'start', // chwilowo ustawiamy pozycję startową; nadpiszemy ją stanem z serwera
        pieceTheme: function (piece) {
            return `static/img/chesspieces/wikipedia/${piece}.png`;
        }
    });

    // Pobierz stan gry z serwera po załadowaniu strony
    fetch('/game_state')
        .then(response => response.json())
        .then(data => {
            if(data.status === 'success') {
                currentTurn = data.currentTurn;  // ustawienie aktualnej tury z serwera
                board.position(data.fen);        // ustawienie pozycji na szachownicy zgodnie z FEN z serwera
            } else {
                alert("Nie udało się załadować stanu gry.");
            }
        })
        .catch(error => console.error('Error przy pobieraniu stanu gry:', error));

    // Delegacja zdarzeń - nasłuchujemy kliknięć na elementach .square-55d63
    $('#board1').on('click', '.square-55d63', function() {
        // Używamy klasy, aby uzyskać kwadrat, ponieważ elementy szachownicy mogą nie mieć atrybutu data-square.
        // Zakładam, że klasy pól mają format .square-a1, .square-b2 itd.
        const classes = $(this).attr('class').split(/\s+/);
        const squareClass = classes.find(c => /^square-[a-h][1-8]$/.test(c));
        const square = squareClass ? squareClass.replace('square-', '') : null;
        
        if(!square) return;  // jeżeli nie znaleziono pola, zakończ działanie

        const piece = board.position()[square];  // Pobierz figurę z klikniętego pola
        console.log("Kliknięto pole:", square, "Figura:", piece);
        
        // Logika po kliknięciu:
        if (selectedSquare) {
            if (possibleMoves.includes(square)) {
                handleMove(selectedSquare, square);
            }
            clearHighlights();
            selectedSquare = null;
            possibleMoves = [];
        } else {
            if (piece && isCurrentPlayerPiece(piece)) {
                selectedSquare = square;
                fetchPossibleMoves(square);
            }
        }
    });

    function fetchPossibleMoves(square) {
        fetch(`/possible_moves`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ square: square }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                possibleMoves = data.moves;
                highlightSquare(square);
                highlightPossibleMoves(data.moves);
            } else {
                alert(data.message);
            }
        })
        .catch(error => console.error('Error:', error));
    }

    function handleMove(source, target) {
        const move = `${source}-${target}`;
        fetch('/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ move: move }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'error') {
                alert(data.message);
            } else {
                board.position(data.new_fen);  // Ustaw nową pozycję na szachownicy
                currentTurn = (currentTurn === 'white') ? 'black' : 'white';
                if(data.checkmate) {
                    // Opóźnienie alertu, aby dać czas na odświeżenie szachownicy
                    setTimeout(() => {
                        alert("Szach mat! Gra zakończona!");
                    }, 250);
                }
            }
            clearHighlights();
        })
        .catch(error => console.error('Error:', error));
    }
    
    
    function isCurrentPlayerPiece(piece) {
        return (currentTurn === 'white' && piece.startsWith('w')) ||
               (currentTurn === 'black' && piece.startsWith('b'));
    }

    function highlightSquare(square) {
        const el = document.querySelector(`.square-${square}`);
        if (el) el.classList.add('highlight');
    }

    function highlightPossibleMoves(moves) {
        moves.forEach(move => {
            const el = document.querySelector(`.square-${move}`);
            if (el) el.classList.add('possible-move');
        });
    }

    function clearHighlights() {
        document.querySelectorAll('.highlight, .possible-move')
                .forEach(el => el.classList.remove('highlight', 'possible-move'));
    }
});

// Obsługa kliknięcia przycisku Restart
document.getElementById('restartButton').addEventListener('click', function() {
    fetch('/restart', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(response => response.json())
    .then(data => {
        if(data.status === 'success') {
            // Odśwież stronę po pomyślnym restarcie
            window.location.reload();
        } else {
            alert("Nie udało się zrestartować gry.");
        }
    })
    .catch(error => console.error('Błąd przy restarcie:', error));
});


