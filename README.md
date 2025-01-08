# ChessAI

ChessAI is a project aimed at developing a chess-playing artificial intelligence using Reinforcement Learning (RL) and Monte Carlo Tree Search (MCTS). Initially, the user interface will be local (pygame-chess), with a web-based version using Chessboard.js planned for later stages.

---

## Project Goal

The goal is to develop an AI that:
- Learns chess through game simulations.
- Achieves the skill level of an amateur chess player.
- Visualizes its learning progress over time.

---

## Technology and Tools

- **AI and Machine Learning**: TensorFlow, NumPy, Pandas  
- **Chess Logic**: python-chess  
- **User Interface (GUI)**: pygame-chess  
- **Planned Web Interface**: Chessboard.js + Flask/FastAPI (future stages)  
- **Result Visualization**: Matplotlib, TensorBoard  
- **Testing**: Pytest  

---

## Project Structure

ChessAI/
│
├── src/
│   ├── ai/
│   │   ├── reinforcement_learning.py    # RL algorithms
│   │   ├── mcts.py                      # Monte Carlo Tree Search implementation
│   │
│   ├── chess_engine/
│   │   ├── board.py                     # Chessboard representation
│   │   ├── move_generator.py            # Generating legal moves
│   │
│   ├── interface/
│       ├── pygame_ui.py                 # GUI based on pygame
│
├── data/
│   ├── training_data/                   # Recorded games for analysis
│   ├── models/                          # Saved AI models
│
├── tests/
│   ├── test_ai.py                       # AI testing
│   ├── test_ui.py                       # Interface testing
│
├── main.py                              # Main program entry point
│
└── README.md                            # Repository documentation


---

## Installation

1. Clone the repository:

    git clone https://github.com/yourusername/ChessAI.git
    cd ChessAI
    

2. Create a virtual environment and install required packages:
    
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    

3. Run the user interface for a simple game against the AI:

    python main.py
    

---

## Milestones

1. Implementation of the basic RL algorithm.
2. Integration of MCTS with the AI.
3. Development of a simple GUI using pygame.
4. Testing and optimization of the algorithm.
5. Implementation of a web interface with Chessboard.js.

---

## Future Plans

- Transition to Chessboard.js for an improved user experience.
- Deploy a web application for online gameplay.
- Further enhance the AI and test it against chess engines (e.g., Stockfish).

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

