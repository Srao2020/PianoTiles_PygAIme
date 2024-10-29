# Falling Tiles AI 
I chose this game because I love playing piano tiles and it has a relatively simple game structure that can be made more difficult. 

## Overview

This project is an AI-based tile-clicking game developed using Python and Pygame. The goal of the game is to click on falling tiles before they reach the bottom of the screen. The game features a Deep Q-Network (DQN) agent that learns how to master the game over time through reinforcement learning. As the game progresses, the AI agent improves its decision-making abilities, optimizing for higher scores by taking the correct actions.

## Features

- **AI Agent**: Implemented using Deep Q-Network (DQN) to learn the game rules and master the gameplay.
- **Difficulty Scaling**: The game increases in difficulty based on the player's or AI's score.
- **Reward System**: A detailed reward system based on correct or incorrect tile clicks, streak bonuses, and score progression.
- **Real-Time Learning**: The agent uses an epsilon-greedy strategy for exploration and exploitation, with epsilon decay over episodes.

## How to Play

### Human Player
- Click on the falling tiles before they reach the bottom of the screen. 
- Your score increases for every correct tile clicked. 
- The game ends when a tile reaches the bottom.

### AI Player
- The AI agent decides which tile to click using a learned Q-value policy.
- The agent improves over time, learning which actions maximize rewards.

## Project Structure

- **`main.py`**: Main game logic that includes the game loop, tile management, and AI agent integration.
- **`dqn_agent.py`**: The Deep Q-Network (DQN) agent class, which includes the neural network architecture, experience replay, action selection, and learning logic.
- **`utils.py`**: Helper functions for the game such as drawing tiles, handling colors, and playing sound effects.
- **`sounds/`**: Directory containing sound effects used in the game (e.g., for tile clicks, game over).
  
## Dependencies

To run this project, you'll need the following libraries:

- `pygame`: For creating the game window and handling game logic.
- `torch`: PyTorch library for implementing the Deep Q-Network.
- `numpy`: For array operations in the state and reward functions.

## AI Tile-Clicking Agent

This AI agent uses a Deep Q-Network (DQN) to learn optimal actions to maximize its score in a tile-clicking game. The state of the game is represented by the positions of the tiles and the current difficulty level, which are fed into the DQN. The agent receives rewards based on correct or incorrect actions and adjusts its behavior over multiple episodes to improve performance.

### Agent Details
- **State Size**: 27 (3 features per tile, up to 9 tiles)
- **Action Size**: 4 (corresponding to the 4 rows in the game)
- **Neural Network**: 3 fully connected layers with ReLU activations
- **Epsilon-Greedy**: Starts with full exploration and gradually reduces exploration (epsilon decay) as the agent learns

### Game Rules
- The player must click on falling tiles before they reach the bottom of the screen.
- The score increases by 1 for each correct tile clicked.
- The game becomes progressively harder as the score increases.
- The AI agent learns to click the correct tiles and maximizes its reward over time.

### Future Enhancements
- **Human-AI Interaction**: Allow human players to compete against the AI in real-time.
- **Enhanced State Representation**: Add more features (e.g., more tile colors, speed) to improve the agent's learning.
- **More Game Levels**: Implement multiple levels of increasing difficulty (more tiles, more double clicks, different backgrounds) for both the AI and human player.

## Project Collaboration
- **dmitchelldm74/Tiles-pygame** for pygame game inspiration (built the current game off of this)
- ChatGPT for error handling
