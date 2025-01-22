document.addEventListener("DOMContentLoaded", function () {
    console.log("chess_interface.js został załadowany");

    let currentTurn;
    let selectedSquare = null;
    let possibleMoves = [];

    const board = Chessboard('board1', {
        position: 'start',
        pieceTheme: function (piece) {
            return `static/img/chesspieces/wikipedia/${piece}.png`;
        }
    });

    fetch('/game_state')
        .then(response => response.json())
        .then(data => {
            if(data.status === 'success') {
                currentTurn = data.currentTurn;
                board.position(data.fen);
            } else {
                alert("Nie udało się załadować stanu gry.");
            }
        })
        .catch(error => console.error('Error przy pobieraniu stanu gry:', error));

    $('#board1').on('click', '.square-55d63', function() {
        const classes = $(this).attr('class').split(/\s+/);
        const squareClass = classes.find(c => /^square-[a-h][1-8]$/.test(c));
        const square = squareClass ? squareClass.replace('square-', '') : null;
        
        if(!square) return;
        const piece = board.position()[square];
        console.log("Kliknięto pole:", square, "Figura:", piece);
        
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
        let move = `${source}-${target}`;

        const piece = board.position()[source];
        if (piece) {
            if (piece === 'wP' && target[1] === '8') {
                move += 'q';
            }
            else if (piece === 'bP' && target[1] === '1') {
                move += 'q';
            }
        }

        console.log("Ruch wysyłany na serwer:", move);

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
                board.position(data.new_fen);
                currentTurn = (currentTurn === 'white') ? 'black' : 'white';
                if(data.checkmate) {
                    setTimeout(() => { alert("Szach mat! Gra zakończona!"); }, 250);
                } else {
                    fetch('/ai_move')
                      .then(response => response.json())
                      .then(aiData => {
                          if(aiData.status === 'success'){
                              board.position(aiData.new_fen);
                              currentTurn = (currentTurn === 'white') ? 'black' : 'white';
                              if(aiData.checkmate) {
                                  setTimeout(() => { alert("Szach mat! Gra zakończona!"); }, 250);
                              }
                          } else {
                              alert(aiData.message);
                          }
                      });
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

document.getElementById('restartButton').addEventListener('click', function() {
    fetch('/restart', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(response => response.json())
    .then(data => {
        if(data.status === 'success') {
            window.location.reload();
        } else {
            alert("Nie udało się zrestartować gry.");
        }
    })
    .catch(error => console.error('Błąd przy restarcie:', error));
});


