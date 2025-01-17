// Inicjalizacja obiektu gry za pomocą chess.js
var game = new Chess();

// Konfiguracja szachownicy
var config = {
  draggable: true,   // Umożliwia przeciąganie figur
  position: 'start', // Początkowa pozycja
  onDragStart: function(source, piece, position, orientation) {
    // Zablokuj ruch, jeśli gra zakończona
    if (game.game_over()) return false;
  },
  onDrop: function(source, target) {
    // Próba wykonania ruchu
    var move = game.move({
      from: source,
      to: target,
      promotion: 'q' // automatyczna promocja na hetmana
    });
    
    // Jeśli ruch nielegalny, cofnięcie
    if (move === null) return 'snapback';
  },
  onSnapEnd: function() {
    // Aktualizacja pozycji szachownicy po zakończeniu ruchu
    board.position(game.fen());
  }
};

// Utworzenie szachownicy w elemencie #board
var board = Chessboard('board', config);

console.log("Interfejs szachownicy załadowany.");
