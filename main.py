import sys, random, pygame
from dqn_agent import DQNAgent
from utils import *

pygame.init()
pygame.mixer.init()

# Constants
FPS = 60  # Frames per second
size = width, height = 450, 650  # Screen size
screen_w = width
screen_h = height
half_w = screen_w / 2
rows = 4  # Number of rows
row_w = screen_w / rows  # Width of each row
tile_w = row_w - 2  # Width of each tile
tile_h = tile_w * 2  # Height of each tile
tile_spacing = tile_h + 10  # Added spacing to prevent overlapping tiles

# Initialize the screen
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Tiles AI Game")
clock = pygame.time.Clock()

# Game variables
difficulty = 2  # Starting difficulty level
score = 0  # Player score
tiles = []  # List of tiles
game = False  # Game state flag
hscore = 0  # Track the high score
episode_num = 0  # Episode counter for tracking AI episodes
total_reward = 0  # Track total reward in each episode
streak = 0  # Track streak of consecutive correct AI actions

# Initialize DQN agent with state size (27) and action size (4)
agent = DQNAgent(state_size=27, action_size=4, epsilon=1.0)  # Start with full exploration


# Highscore-related functions
def highscore(method):
    """
    Handle high score operations including saving the high score and determining
    background and foreground colors based on score.

    Args:
        method (str): Determines the operation ("SAVE", "BGCOLOR", or "FGCOLOR")
    """
    global score, hscore
    if method == "SAVE":
        if score > hscore:
            hscore = score
    elif method == "BGCOLOR":
        return rgb.WHITE if score < hscore or score == 0 else rgb.BLACK
    elif method == "FGCOLOR":
        return rgb.BLACK if score < hscore or score == 0 else rgb.WHITE


# Game over display
def display_game_over():
    """
    Display the 'Game Over' screen with the final score.
    """
    screen.fill(rgb.BLACK)
    label(screen, "monospace", 50, "Game Over", rgb.WHITE, (0, 0), center=(half_w, screen_h / 2 - 50))
    label(screen, "monospace", 40, f"Your Score: {score}", rgb.WHITE, (0, 0), center=(half_w, screen_h / 2 + 50))
    pygame.display.flip()
    pygame.time.wait(2000)  # Pause for 2 seconds before exiting or resetting


# Start or reset the game
def start_game():
    """
    Reset the game variables and prepare for a new episode.
    """
    global game, tiles, score, difficulty, total_reward, streak
    game = True
    score = 0  # Reset score
    difficulty = 2  # Reset difficulty
    total_reward = 0  # Reset total reward
    streak = 0  # Reset streak
    tiles = [[random_color(), 2, -tile_h * 2], [random_color(), 2 + row_w * 1, -tile_h * 2]]  # Initial tiles
    [add_tile() for _ in range(7)]  # Add initial set of tiles
    play_sound("begin.ogg")  # Play game start sound


# End the current game
def end_game():
    """
    Handle the end of the game, including saving the high score, applying penalties,
    and logging the episode statistics.
    """
    global game, episode_num, total_reward
    play_sound("end.ogg")  # Play game over sound
    game = False
    highscore("SAVE")  # Save high score if applicable
    display_game_over()  # Show game over message
    agent.save_model('dqn_model.pth')  # Save the AI model

    # Apply a minor penalty for game over
    total_reward -= 1

    # Log the final score and total reward of the episode
    print(f"Episode {episode_num}: Score = {score}, Total Reward = {total_reward}")

    # Increment the episode count and restart the game
    episode_num += 1
    start_game()


# Add a new tile to the game
def add_tile():
    """
    Add a new tile in a random row above the screen at a properly spaced position.
    """
    row = random.randint(0, rows - 1)
    y_position = -tile_h  # Start above the screen
    if tiles:
        last_tile_y = max(t[2] for t in tiles)  # Get the lowest tile
        y_position = last_tile_y - tile_spacing  # Space new tile above
    tiles.append([random_color(), 2 + row_w * row, y_position])


# Calculate reward for the AI agent
def calculate_reward(action_reward, score, streak):
    """
    Calculate the reward based on the action (correct or wrong), including streak bonuses
    for consecutive correct actions and score bonuses for every 50 points.

    Args:
        action_reward (int): Reward from the AI's action (+1 for correct, -1 for wrong).
        score (int): Current score of the player.
        streak (int): Current streak of consecutive correct actions.

    Returns:
        int: The calculated reward.
    """
    reward = action_reward  # Start with action reward

    # Gradual reward progression based on the score
    if score > 0:
        reward += (score // 10)  # Increase reward based on score (for every 10 points)

    # Bonus for achieving every 50 points
    if score % 50 == 0 and action_reward > 0:
        reward += 10

    # Add streak bonus for consecutive correct actions
    if action_reward > 0:
        reward += streak * 0.5  # Increase reward with longer streaks

    return reward


# Handle AI "clicking" a tile
def ai_click_tile(action):
    """
    Simulate the AI clicking a tile based on its chosen action (row). If the action is correct,
    the tile in the selected row is removed and a new one is added.

    Args:
        action (int): The row selected by the AI agent.

    Returns:
        int: +1 for correct action, -1 for incorrect action.
    """
    global score, streak
    for i, t in enumerate(tiles):
        if int(t[1] / row_w) == action:  # Check if the tile is in the selected row
            del tiles[i]  # Remove the tile
            play_sound("click.ogg")  # Play click sound
            add_tile()  # Add a new tile
            score += 1  # Increase score
            streak += 1  # Increment streak for consecutive correct actions
            return 1  # Correct action, tile clicked
    streak = 0  # Reset streak on incorrect action
    return 0  # Wrong action, no tile in that row


# Draw the game elements
def draw_game():
    """
    Draw the game elements including the background, tiles, and score. Adjust difficulty based
    on the current score.
    """
    global difficulty
    bg_color = highscore("BGCOLOR")
    screen.fill(bg_color)
    draw_vertical_lines(screen, rows, row_w, screen_h, bg_color)
    difficulty = min(10, 2 + (score // 20))  # Gradual difficulty progression

    # Draw and move each tile
    for t in tiles:
        pygame.draw.rect(screen, t[0], [t[1], t[2], tile_w, tile_h], 0)
        t[2] += difficulty  # Move tiles down
        if t[2] > screen_h:
            end_game()  # End game if a tile reaches the bottom

    # Display the current score
    label(screen, "monospace", 100, str(score), highscore("FGCOLOR"), (row_w / 3.25, screen_h - 100))


# Main game loop with AI agent
start_game()
while True:
    if game:
        # Get the current state of the game
        state = agent.get_state(tiles, screen_w, screen_h, difficulty)

        # AI decides which row to "click"
        action = agent.act(state)

        # Simulate AI clicking the tile
        action_reward = ai_click_tile(action)

        # Calculate the reward (with streak and score bonuses)
        reward = calculate_reward(action_reward, score, streak)

        # Train the AI based on the feedback
        agent.learn(state, action, reward)

        # Apply epsilon decay for AI exploration-exploitation balance
        agent.epsilon = max(agent.epsilon * 0.999, 0.05)

        # Update total reward for the episode
        total_reward += reward

        # Draw the game
        draw_game()

    # Handle user exit event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    # Refresh the screen and control FPS
    pygame.display.flip()
    clock.tick(FPS)
