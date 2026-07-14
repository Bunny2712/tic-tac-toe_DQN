# tic-tac-toe_DQN

A tic-tac-toe agent trained from scratch using Deep Q-Learning (DQN) and self-play — no hardcoded strategy, no minimax, no lookup tables. The agent learns to play optimally purely through trial, error, and reward.
 
**Live demo:** https://bunny2712.github.io/tic-tac-toe_DQN/
 
Play against it directly in your browser — the trained PyTorch weights are ported to JavaScript and run entirely client-side, so every move you see is the actual network evaluating the board in real time.
 
---
 
## Overview
 
This project implements a Deep Q-Network that learns to play tic-tac-toe by playing thousands of games against itself, gradually improving its strategy through reinforcement learning. The final model:
 
- Beats a random-move opponent **~98% of the time**, with **zero losses**, over 100 evaluation games
- Plays as both X and O using a single shared policy (see [Canonical State Trick](#canonical-state-trick) below)
- Runs inference in the browser with no backend — the trained weights are extracted from the `.pth` checkpoint and reimplemented as a plain JavaScript forward pass
## How It Works
 
### Architecture
 
A small fully-connected network maps a board state directly to a value for each of the 9 possible moves:
 
```
Input (9)  →  Linear(9, 256)  →  ReLU
           →  Linear(256, 256) →  ReLU
           →  Linear(256, 9)          →  Q-values (9)
```
 
~70,000 parameters total. Each of the 9 outputs is the network's estimated value (Q-value) of placing a mark in that cell — the agent simply picks the legal move with the highest value.
 
### Canonical State Trick
 
Rather than training separate strategies for "playing as X" and "playing as O," the board is always presented to the network from the current player's perspective:
 
```python
def get_canonical_state(board, player):
    return board * player
```
 
This means the network always sees `1` for its own pieces and `-1` for the opponent's, regardless of which symbol it's actually playing. One network, one policy, works for both sides.
 
### Training Loop (DQN + Self-Play)
 
- **Environment:** the agent plays against itself, alternating turns
- **Exploration:** ε-greedy, decaying from `1.0 → 0.05` (×0.9995 per step)
- **Experience replay:** a buffer of 5,000 transitions, sampled in batches of 64
- **Target network:** a separate slowly-updated copy (soft update, τ = 0.01, every 20 episodes) stabilizes the learning targets
- **Reward shaping:** `+10` for a win, `0` for a draw — since tic-tac-toe is zero-sum, the opponent's best response is subtracted from the bootstrapped target, so a great move for them is penalized for the agent
- **Loss:** Smooth L1 (Huber) loss, with gradient clipping at 1.0
- **Optimizer:** Adam, learning rate `1e-3`
Illegal moves are masked out with `-∞` before taking the argmax, so the agent can never select an occupied cell — at inference time it always plays legally, with no exploration noise (ε = 0).
 
### Results
 
| Metric | Value |
|---|---|
| Training episodes | 5,000 |
| Win rate vs. random opponent | ~98% |
| Draw rate vs. random opponent | ~2% |
| Loss rate vs. random opponent | 0% |
| Parameters | 70,665 |
 
## Project Structure
 
```
tic-tac-toe_DQN/
├── index.html                          # Live interactive demo (GitHub Pages)
├── TicTacToe_DQN_Improved.ipynb        # Training notebook (environment, DQN, training loop, evaluation)
├── best_model.pth                      # Trained model weights (PyTorch state_dict)
└── README.md
```
 
## Try It Yourself
 
**Play it live:** https://bunny2712.github.io/tic-tac-toe_DQN/
 
**Or run it locally:**
1. Download `index.html` from this repo
2. Open it directly in any modern browser — no install, no server required
**Or retrain it:**
1. Open `TicTacToe_DQN_Improved.ipynb` in Jupyter or Google Colab
2. Run all cells — training takes roughly 1–2 minutes on a single GPU (or CPU, just slower)
3. The notebook saves the trained weights as `best_model.pth`
## Tech Stack
 
- **Training:** Python, PyTorch, NumPy
- **Demo:** HTML, CSS, vanilla JavaScript (the trained weights are extracted from the `.pth` file and the forward pass is reimplemented in JS — no server, no ML framework needed to run the demo)
## Possible Extensions
 
- Generalize the environment to larger boards (`N×N`, connect-`k`)
- Compare against a minimax baseline to formally verify optimal play
- Add a difficulty slider by exposing ε at inference time
- Train with prioritized experience replay
## Author
 
Built as part of a college internship project exploring reinforcement learning fundamentals through a small, fully self-contained example — training, evaluation, and deployment all included.
