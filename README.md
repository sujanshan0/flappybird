from sitecustomize import pythonpath
import os
import sys
import pickle
import random

import pygame
import neat


WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730

PIPE_GAP = 160        
PIPE_VEL = 5          
BASE_VEL = 5          
FPS = 100             

TARGET_SCORE = 1000   
MAX_GENERATIONS = 50  


FITNESS_PER_FRAME = 0.1      
FITNESS_PER_PIPE = 5         
FITNESS_COLLISION = -1      

WINNER_FILE = "winner.pkl"
CONFIG_FILE = "config-feedforward.txt"


pygame.init()
STAT_FONT = pygame.font.SysFont("comicsans", 50)
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird – zelflerend")

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs", "bg.png")).convert_alpha(), (600, 900))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird" + str(x) + ".png")))
               for x in range(1, 4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")).convert_alpha())

gen = 0  


class TargetReached(Exception):
    """Wordt opgegooid zodra een vogel TARGET_SCORE haalt; stopt de NEAT-training."""

    def __init__(self, genome):
        super().__init__("Doelscore bereikt")
        self.genome = genome


class Bird:
    """
    De vogel. Beweegt onder invloed van de zwaartekracht en springt wanneer het
    neurale netwerk dat besluit.
    """
    MAX_ROTATION = 25
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        """Laat de vogel omhoog springen."""
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        """Verplaats de vogel één frame (valversnelling plus eindsnelheid)."""
        self.tick_count += 1
        displacement = self.vel * self.tick_count + 0.5 * 3 * self.tick_count ** 2

        
        if displacement >= 16:
            displacement = (displacement / abs(displacement)) * 16
        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:  
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:                                              
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        """Teken de vogel, met vleugelanimatie en kanteling."""
        self.img_count += 1

        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:  
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        blit_rotate_center(win, self.img, (self.x, self.y), self.tilt)

    def get_mask(self):
        """Geeft het pixelmasker van het huidige vogelbeeld (voor botsingscontrole)."""
        return pygame.mask.from_surface(self.img)


class Pipe:
    """Een paar pijpen (boven en onder) met een opening van PIPE_GAP pixels."""

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img
        self.passed = False
        self.set_height()

    def set_height(self):
        """Kies willekeurig de hoogte van de opening en bereken de posities van de pijpen."""
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + PIPE_GAP

    def move(self):
        self.x -= PIPE_VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        """Geeft True als de vogel de bovenste of onderste pijp raakt (pixel-precies)."""
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        return bool(bird_mask.overlap(bottom_mask, bottom_offset) or
                    bird_mask.overlap(top_mask, top_offset))


class Base:
    """De bewegende grond onderaan het scherm."""
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= BASE_VEL
        self.x2 -= BASE_VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blit_rotate_center(surf, image, topleft, angle):
    """Draai een afbeelding om zijn middelpunt en teken hem op het scherm."""
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)
    surf.blit(rotated_image, new_rect.topleft)


def bird_inputs(bird, pipe):
    """
    De drie invoerwaarden voor het neurale netwerk (zie config-feedforward.txt: num_inputs = 3):
      1. de hoogte van de vogel
      2. de afstand van de vogel tot de bovenkant van de opening
      3. de afstand van de vogel tot de onderkant van de opening
    """
    return (bird.y, abs(bird.y - pipe.height), abs(bird.y - pipe.bottom))


def draw_window(win, birds, pipes, base, score, generation, pipe_ind):
    """Teken het hele scherm: achtergrond, pijpen, grond, vogels en de statistieken."""
    win.blit(bg_img, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for bird in birds:
        bird.draw(win)

    score_label = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))

    gen_label = STAT_FONT.render("Gens: " + str(generation), 1, (255, 255, 255))
    win.blit(gen_label, (10, 10))

    alive_label = STAT_FONT.render("Alive: " + str(len(birds)), 1, (255, 255, 255))
    win.blit(alive_label, (10, 50))

    pygame.display.update()


def eval_genomes(genomes, config):
    """
    Laat alle vogels van één generatie tegelijk spelen en bepaal hun fitness.

    FITNESS FUNCTION (toelichting in eigen woorden, voor de beoordeling)
    --------------------------------------------------------------------
    De fitness is het cijfer waarmee NEAT bepaalt welke vogels mogen
    "ouders" worden van de volgende generatie. Hoe hoger de fitness, hoe
    groter de kans dat het brein van die vogel wordt doorgegeven.
    Deze functie bestaat uit drie beloningen:

      * +0,1 per frame dat een vogel leeft (FITNESS_PER_FRAME):
        een vogel die lang overleeft, is beter dan een die snel crasht.
        Dit geeft een leersignaal, ook vóórdat een vogel een pijp haalt.
      * +5 per gepasseerde pijp (FITNESS_PER_PIPE):
        dit is het echte doel van het spel. Alle levende vogels krijgen
        deze beloning, omdat de pijpen voor de hele groep gelijk zijn.
      * -1 bij botsing met een pijp (FITNESS_COLLISION):
        een straf zodat een vogel die tegen een pijp vliegt minder
        waard is dan een vogel die (even lang) veilig blijft.
        Een val op de grond of tegen het plafond geeft geen straf,
        omdat die vogel toch al stopt met leven.

    Het eindcijfer van een vogel is dus ongeveer:
        fitness = 0,1 * aantal frames + 5 * aantal pijpen - 1 * (botsing)

    TRAININGSSTOP
    -------------
    Zodra een vogel TARGET_SCORE (1000) pijpen heeft gepasseerd, is het doel
    bereikt. We gooien dan de uitzondering TargetReached op, die de NEAT-training
    stopt en de vogel teruggeeft zodat hij kan worden opgeslagen.
    """
    global gen
    gen += 1

    nets = []
    birds = []
    ge = []  

    for _, genome in genomes:
        genome.fitness = 0
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        birds.append(Bird(230, 350))
        ge.append(genome)

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0
    clock = pygame.time.Clock()

    while len(birds) > 0:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        
        pipe_ind = 0
        if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_ind = 1

        
        for i, bird in enumerate(birds):
            ge[i].fitness += FITNESS_PER_FRAME
            bird.move()
            output = nets[i].activate(bird_inputs(bird, pipes[pipe_ind]))
            if output[0] > 0.5:
                bird.jump()

        base.move()
        for pipe in pipes:
            pipe.move()

        
        survivors = []
        for i, bird in enumerate(birds):
            if any(pipe.collide(bird) for pipe in pipes):
                ge[i].fitness += FITNESS_COLLISION
            elif bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                pass  
            else:
                survivors.append(i)

        nets = [nets[i] for i in survivors]
        ge = [ge[i] for i in survivors]
        birds = [birds[i] for i in survivors]

        if not birds:
            break

        
        add_pipe = False
        for pipe in pipes:
            if not pipe.passed and pipe.x < birds[0].x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            for genome in ge:
                genome.fitness += FITNESS_PER_PIPE
            pipes.append(Pipe(WIN_WIDTH))

            if score >= TARGET_SCORE:
                raise TargetReached(ge[0])

        
        pipes = [p for p in pipes if p.x + p.PIPE_TOP.get_width() >= 0]

        draw_window(WIN, birds, pipes, base, score, gen, pipe_ind)


def train(config_path):
    """Train de NEAT-populatie en sla de slimste vogel op in WINNER_FILE."""
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    try:
        winner = p.run(eval_genomes, MAX_GENERATIONS)
        print("\nMaximaal aantal generaties bereikt. Beste genoom opgeslagen.")
    except TargetReached as done:
        winner = done.genome
        print("\nDoel bereikt: score " + str(TARGET_SCORE) + " in generatie " + str(gen) + ".")

    with open(WINNER_FILE, "wb") as f:
        pickle.dump(winner, f)
    print("Slimste vogel opgeslagen in " + WINNER_FILE)


def play_with_best_bird(winner_path, config):
    """Het eindspel: de opgeslagen vogel speelt zelfstandig tot hij tegen iets aanvliegt."""
    with open(winner_path, "rb") as f:
        winner = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    bird = Bird(230, 350)
    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0
    clock = pygame.time.Clock()

    playing = True
    while playing:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                playing = False

        pipe_ind = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_ind = 1

        bird.move()
        output = net.activate(bird_inputs(bird, pipes[pipe_ind]))
        if output[0] > 0.5:
            bird.jump()

        base.move()
        for pipe in pipes:
            pipe.move()

        crashed = any(pipe.collide(bird) for pipe in pipes) or \
            bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50
        if crashed:
            print("Game over! Score: " + str(score))
            playing = False
            break

        
        for pipe in pipes:
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                score += 1
                pipes.append(Pipe(WIN_WIDTH))
        pipes = [p for p in pipes if p.x + p.PIPE_TOP.get_width() >= 0]

        draw_window(WIN, [bird], pipes, base, score, 0, pipe_ind)


def run_endgame(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    if not os.path.exists(WINNER_FILE):
        print("Geen " + WINNER_FILE + " gevonden. Voer eerst 'python flappy_bird_neat.py train' uit.")
        return
    play_with_best_bird(WINNER_FILE, config)


if __name__ == "__main__":
    
    local_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(local_dir)  
    config_path = os.path.join(local_dir, CONFIG_FILE)

    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        train(config_path)
    elif mode == "play":
        run_endgame(config_path)
    else:
        print("Gebruik: python flappy_bird_neat.py [train|play]")

    pygame.quit()
