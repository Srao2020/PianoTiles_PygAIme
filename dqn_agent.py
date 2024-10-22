import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Initialize the DQNNetwork with three fully connected layers:
        - Input layer to hidden layer 1 (64 neurons)
        - Hidden layer 1 to hidden layer 2 (64 neurons)
        - Hidden layer 2 to output layer (number of actions)

        Args:
        - input_size: Number of input features (state size)
        - output_size: Number of output neurons (action size)
        """
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        """
        Forward pass through the neural network.
        - Apply ReLU activation to the first two layers.
        - Output raw Q-values from the final layer (no activation).

        Args:
        - x: Input state

        Returns:
        - Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size, alpha=0.001, gamma=0.99, epsilon=0.1):
        """
        Initialize the DQN agent with its key parameters:
        - state_size: Number of features in the state.
        - action_size: Number of possible actions.
        - alpha: Learning rate for the Adam optimizer.
        - gamma: Discount factor for future rewards.
        - epsilon: Exploration rate for epsilon-greedy action selection.

        The agent maintains a memory buffer for experience replay and uses a DQN model to estimate Q-values.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

        self.memory = []
        self.batch_size = 64
        self.memory_limit = 1000

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience (state, action, reward, next_state, done) in the memory buffer.
        If the memory exceeds its limit, remove the oldest experience (FIFO).

        Args:
        - state: The current state.
        - action: The action taken.
        - reward: The reward received.
        - next_state: The next state after the action.
        - done: Whether the episode is finished (True/False).
        """
        if len(self.memory) > self.memory_limit:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def get_state(self, tiles, screen_w, screen_h, difficulty):
        """
        Generate a state representation from the game environment.
        Normalizes tile positions and difficulty and ensures consistent state size by padding with zeroes.

        Args:
        - tiles: List of tile positions (x, y).
        - screen_w: Screen width for normalization.
        - screen_h: Screen height for normalization.
        - difficulty: Difficulty level.

        Returns:
        - Normalized state as a NumPy array.
        """
        state = []
        for t in tiles:
            state.append(t[1] / screen_w)  # Normalized horizontal position
            state.append(t[2] / screen_h)  # Normalized vertical position
            state.append(difficulty / 10)  # Difficulty level

        # Assuming up to 9 tiles, padding with 0s if necessary
        max_tiles = 9
        state_size_needed = max_tiles * 3  # 3 features per tile (x position, y position, difficulty)
        while len(state) < state_size_needed:
            state.extend([0, 0, 0])  # Add padding for missing tiles

        state = state[:state_size_needed]  # Ensure the state has exactly state_size features
        return np.array(state)

    def act(self, state):
        """
        Choose an action using epsilon-greedy strategy:
        - With probability epsilon, select a random action (exploration).
        - Otherwise, select the action with the highest predicted Q-value (exploitation).

        Args:
        - state: The current state.

        Returns:
        - Selected action (integer).
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return np.argmax(q_values.cpu().data.numpy())

    def learn(self, state, action, reward):
        """
        Sample a batch of experiences from memory and perform Q-learning updates:
        - Use the experiences to compute Q-value targets and minimize the loss.
        - Update the DQN model based on the computed loss.

        This function ensures the model learns from past experiences and improves over time.
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            reward_tensor = torch.FloatTensor([reward]).to(self.device)

            q_values = self.model(state_tensor)
            next_q_values = self.model(next_state_tensor)

            target = reward_tensor + self.gamma * torch.max(next_q_values)

            loss = self.loss_fn(q_values[action], target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Save the trained model parameters to a file.
    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)

# Load the trained model parameters from a file.
    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name))