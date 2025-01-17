---

# ChessAI

ChessAI is a project aimed at developing a chess-playing AI using Reinforcement Learning (RL) and Monte Carlo Tree Search (MCTS). The AI learns by simulating games against itself, aspiring to reach a level comparable to an amateur player. Initially, the project features a local interface built with Pygame, with plans to implement a web-based GUI using Chessboard.js in the future.

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
- **Web Interface**: Chessboard.js + Flask/FastAPI (future stages)  
- **Result Visualization**: Matplotlib, TensorBoard  
- **Testing**: Pytest  

---

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Tomciom/ChessAI.git
    cd ChessAI
    ```

2. Create a virtual environment and install required packages:
    
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. Run the user interface for a simple game against the AI:

    ```bash
    python main.py
    ```

---

## Milestones

1. Implementation of the basic RL algorithm.
2. Integration of MCTS with the AI.
3. Implementation of a web interface with Chessboard.js.
4. Testing and optimization of the algorithm.

---

## Future Plans

- Further enhance the AI and test it against chess engines (e.g., Stockfish).

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
