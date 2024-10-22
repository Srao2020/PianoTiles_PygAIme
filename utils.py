import pygame
import random


# Define RGB color values as class constants for easy reference
class rgb:
    BLACK = (0, 0, 0)  # Black color (R:0, G:0, B:0)
    WHITE = (255, 255, 255)  # White color (R:255, G:255, B:255)
    BLUE = (0, 0, 255)  # Blue color (R:0, G:0, B:255)
    RED = (255, 0, 0)  # Red color (R:255, G:0, B:0)
    GREEN = (0, 255, 0)  # Green color (R:0, G:255, B:0)
    YELLOW = (255, 255, 0)  # Yellow color (R:255, G:255, B:0)


# Define a function to initialize a font object in pygame
def font(font_name, size):
    """
    Return a pygame font object for rendering text on the screen.

    Args:
    - font_name: The name of the font (string).
    - size: The size of the font (integer).

    Returns:
    - A pygame Font object with the specified font name and size.
    """
    return pygame.font.SysFont(font_name, int(size))


# Define a function to render and display text labels on the screen
def label(screen, font_name, size, text, color, pos, center=False):
    """
    Render a text label on the screen using a specified font and color.

    Args:
    - screen: The surface on which to draw the label.
    - font_name: The name of the font to use.
    - size: The size of the font.
    - text: The string of text to render.
    - color: The RGB color tuple for the text.
    - pos: The position tuple (x, y) for the label on the screen.
    - center: If True, center the text at the given position (optional).
    """
    lbl = font(font_name, size).render(text, 1, color)  # Render the text with the given font and color
    screen.blit(lbl, pos if not center else lbl.get_rect(center=center))  # Draw the text at specified position


# Function to randomly select a color from predefined RGB values
def random_color():
    """
    Return a random color from a predefined set of RGB values (BLUE, RED, GREEN, YELLOW).

    Returns:
    - A random RGB color tuple from the choices.
    """
    return random.choice([rgb.BLUE, rgb.RED, rgb.GREEN, rgb.YELLOW])


# Function to check if a tile is clicked based on the mouse position
def is_tile_clicked(tile):
    """
    Check if a tile is clicked by comparing the tile's position to the current mouse position.

    Args:
    - tile: A tuple containing the tile's x and y position.

    Returns:
    - True if the mouse is within the bounds of the tile, False otherwise.
    """
    x, y = pygame.mouse.get_pos()  # Get the current mouse position (x, y)
    # Check if the mouse coordinates are within the tile's boundaries
    return tile[1] < x < tile[1] + 100 and tile[2] < y < tile[2] + 200


# Function to play a sound file using pygame's mixer
def play_sound(filename):
    """
    Play a sound file from a specified directory.

    Args:
    - filename: The name of the sound file to play (string).
    """
    sound_path = "/Users/25rao/Desktop/mlai/Project_2/sounds/" + filename  # Construct the file path
    pygame.mixer.music.load(sound_path)  # Load the sound file
    pygame.mixer.music.play()  # Play the sound


# Function to draw vertical lines that separate the rows on the screen
def draw_vertical_lines(screen, rows, row_w, screen_h, bg_color):
    """
    Draw vertical lines across the screen to separate rows. The line color is dynamic, changing
    based on the background color.

    Args:
    - screen: The surface on which to draw the lines.
    - rows: The number of vertical lines to draw (corresponds to the number of rows).
    - row_w: The width of each row (distance between lines).
    - screen_h: The height of the screen (vertical span of the lines).
    - bg_color: The background color of the screen (affects line color).
    """
    # Determine the color of the lines based on the background color:
    # If the background is black, lines will be white; otherwise, lines will be black.
    line_color = rgb.WHITE if bg_color == rgb.BLACK else rgb.BLACK

    # Draw vertical lines using pygame's line drawing function
    for x in range(0, rows + 1):
        # Draw a line from (x * row_w, 0) to (x * row_w, screen_h)
        pygame.draw.line(screen, line_color, (row_w * x, 0), (row_w * x, screen_h), 2)
