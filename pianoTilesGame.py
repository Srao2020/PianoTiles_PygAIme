import sys, random, os, time
import pygame

pygame.init()
pygame.mixer.init()
FPS = 60
size=width, height=450, 650
screen_w = width
screen_h = height
half_w = screen_w / 2
pygame.display.set_caption("Tiles")
difficulty = 2
score = 0
try:
    hscore = int(open('.hs','r').readline())
except:
    hscore = 0
    open('.hs','w').write(str(hscore))
rows = 4
row_w = screen_w / rows
row_h = screen_h
tile_w = row_w - 2
half_t_w = tile_w / 2
tile_h = tile_w * 2
blinking_text = True
y = 10

class rgb:
    BLACK = (0,0,0)
    WHITE = (255,255,255)
    BLUE = (0,0,255)
    RED = (255,0,0)
    GREEN = (0,255,0)
    YELLOW = (255,255,0)

screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

def font(font_name, size):
    return pygame.font.SysFont(font_name, int(size))

def label(font_name, size, text, color, pos, center=False):
    global screen
    lbl = font(font_name, size).render(text, 1, color)
    if not center:
        screen.blit(lbl, pos)
    else:
        screen.blit(lbl, lbl.get_rect(center=center))

def add_tile():
    global tiles, rows, row_w
    row = random.randint(int(0), int(rows - 1))
    color = rrgb()
    color = color if color != rgb.BLACK and color != rgb.WHITE else random.choice([rgb.BLUE, rgb.YELLOW, rgb.GREEN, rgb.RED])
    def getY():
        global tile_h
        n = -random.randint(int(tile_h), int(1000))
        for t in tiles:
            if n == t[2] - tile_h - 1 and n != t[2]:
                return n
        try:
            return getY()
        except:
            return t[2] - tile_h - 1
    y = getY()
    tiles.append([color, 2 + row_w * row, y])

def rrgb():
    def c():
        return random.randint(0, 255)
    return (c(), c(), c()) if blinking_text == True else rgb.BLACK

def play(filename_with_sound):
    pygame.mixer.music.load(open("/Users/25rao/Desktop/mlai/Project_2/sounds/%s" % (filename_with_sound), "rb"))
    pygame.mixer.music.play()

def start_game():
    global game, hscore_msg, tiles
    game = True
    play("begin.ogg")
    hscore_msg = False
    tiles = [[rgb.BLUE, 2, -tile_h * 2], [rgb.GREEN, 2 + row_w * 1, -tile_h * 2]]
    [add_tile() for x in range(0, 7)]

def end_game():
    global game, score, hscore, hscore_msg, tiles
    play("end.ogg")
    game = False
    highscore("SAVE")
    # Display the "Game Over" message with the score
    display_game_over()

def display_game_over():
    screen.fill(rgb.BLACK)
    label("monospace", 50, "Game Over", rgb.WHITE, (0, 0), center=(half_w, screen_h / 2 - 50))  # Smaller size
    label("monospace", 40, "Your Score: " + str(score), rgb.WHITE, (0, 0), center=(half_w, screen_h / 2 + 50))  # Score display
    pygame.display.flip()
    pygame.time.wait(2000)  # Display for 2 seconds before resetting

def highscore(method):
    global score, hscore, hscore_msg
    if method == "SAVE":
        if score > hscore:
            hscore = score
            hscore_msg = True
            open('.hs', 'w').write(str(score))
    elif method == "BGCOLOR":
        if score < hscore or score == 0:
            return rgb.WHITE
        else:
            return rgb.BLACK
    elif method == "FGCOLOR":
        if score < hscore or score == 0:
            return rgb.BLACK
        else:
            return rgb.WHITE

def click_tile():
    global mouse_position, tiles, score, tile_w, tile_h
    x, y = mouse_position
    i = 0
    click_on_tile = False
    for t in tiles:
        if x > t[1] and x < t[1] + tile_w and y > t[2] and y < t[2] + tile_h:
            click_on_tile = True
            del tiles[i]
            play("click.ogg")
            add_tile()
            score += 1
        i += 1
    if not click_on_tile:
        end_game()

def draw_vertical_lines():
    global rows, screen, row_w, row_h
    for x in range(0, rows + 1):
        pygame.draw.line(screen, highscore("FGCOLOR"), (row_w * x, 0), (row_w * x, row_h), 2)

tiles = []
game = False
hscore_msg = False

# Start the game immediately when the code runs
start_game()

while True:
    screen.fill(highscore("BGCOLOR"))
    draw_vertical_lines()

    if game == True:
        for t in tiles:
            pygame.draw.rect(screen, t[0], [t[1], t[2], tile_w, tile_h], 0)
            if t[2] < row_h and t[2] + tile_h != row_h:
                t[2] = t[2] + difficulty
            else:
                end_game()
        label("monospace", 100, str(score), highscore("FGCOLOR"), (row_w / 3.25, screen_h - 100))

    for event in pygame.event.get():
        mouse_buttons = pygame.mouse.get_pressed()
        mouse_position = pygame.mouse.get_pos()
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if game == True:
                click_tile()

    pygame.display.flip()
    clock.tick(FPS)