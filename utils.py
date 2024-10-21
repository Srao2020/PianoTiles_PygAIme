import pygame
import random

class rgb:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)

def font(font_name, size):
    return pygame.font.SysFont(font_name, int(size))

def label(screen, font_name, size, text, color, pos, center=False):
    lbl = font(font_name, size).render(text, 1, color)
    screen.blit(lbl, pos if not center else lbl.get_rect(center=center))

def random_color():
    return random.choice([rgb.BLUE, rgb.RED, rgb.GREEN, rgb.YELLOW])

def is_tile_clicked(tile):
    x, y = pygame.mouse.get_pos()
    return tile[1] < x < tile[1] + 100 and tile[2] < y < tile[2] + 200

def play_sound(filename):
    sound_path = "/Users/25rao/Desktop/mlai/Project_2/sounds/" + filename
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play()


def draw_vertical_lines(screen, rows, row_w, screen_h, bg_color):
    """
    Draw vertical lines that separate the rows. The color of the lines changes
    dynamically based on the background color.
    """
    # If background color is black, set the line color to white; otherwise, use black
    line_color = rgb.WHITE if bg_color == rgb.BLACK else rgb.BLACK

    for x in range(0, rows + 1):
        pygame.draw.line(screen, line_color, (row_w * x, 0), (row_w * x, screen_h), 2)
