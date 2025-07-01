"""
Configuratie constanten voor BewegendeAnimaties fMRI-stijl GIF generator.
"""

# Achtergrond configuratie
BACKGROUND_IMAGE_PATH = "background/brain_background.png"  # Pad naar hersenachtergrond
BACKGROUND_DEFAULT_SIZE = (800, 600)  # Standaard afmetingen als geen achtergrond

# Ovaal regio configuratie (hersenregio waar animatie plaatsvindt)
OVAL_CENTER = (400, 300)  # Centrum van het ovaal (x, y)
OVAL_WIDTH = 300  # Breedte van het ovaal (verhoogd van 200 naar 300)
OVAL_HEIGHT = 225  # Hoogte van het ovaal (verhoogd van 150 naar 225)

# Animatie algemene instellingen
ANIMATION_DURATION = 3.0  # Duur van animatie in seconden
FRAMES_PER_SECOND = 15  # FPS voor GIF animaties
TOTAL_FRAMES = int(ANIMATION_DURATION * FRAMES_PER_SECOND)  # Totaal aantal frames

# fMRI kleur configuratie
FMRI_COLORS = {
    'primary': (255, 140, 0),    # Oranje
    'secondary': (255, 69, 0),   # Rood-oranje  
    'accent': (255, 215, 0),     # Goud-geel
    'highlight': (255, 255, 0)   # Geel
}

# Gloed effect instellingen
GLOW_RADIUS = 10  # Radius van gloed effect
GLOW_INTENSITY = 0.7  # Intensiteit van gloed (0.0 - 1.0)

# Bewegend mannetje configuratie
MOVING_FIGURE_SIZE = 15  # Grootte van bewegend figuur
MOVING_FIGURE_SPEED = 1.0  # Snelheid multiplier

# Realistisch lopend mannetje configuratie
WALKING_FIGURE_SIZE = 20  # Grootte van stick figure mannetje
WALKING_SPEED = 3.5  # Bewegingssnelheid in pixels per frame (verhoogd van 2.0 naar 3.5)
WALKING_DIRECTION_CHANGE_CHANCE = 0.05  # Kans op richtingsverandering per frame (0.0-1.0)
WALKING_RANDOM_VARIATION = 0.3  # Willekeurige variatie in beweging (0.0-1.0)
WALKING_POSE_CHANGE_SPEED = 3  # Snelheid van pose verandering (frames per pose)
WALKING_BOUNDARY_MARGIN = 15  # Marge van ovaal rand in pixels

# Random walk configuratie
RANDOM_WALK_STEP_SIZE = 1.5  # Basis stapgrootte voor random walk
RANDOM_WALK_MOMENTUM = 0.7  # Momentum factor (0.0-1.0, hoger = meer rechtdoor)
RANDOM_WALK_DIRECTION_NOISE = 0.2  # Ruis in richting (0.0-1.0)

# Vlekken animatie configuratie
SPOT_MIN_SIZE = 5  # Minimale grootte van vlekken
SPOT_MAX_SIZE = 25  # Maximale grootte van vlekken
SPOT_COUNT = 8  # Aantal vlekken per frame

# Tekst animatie configuratie
TEXT_CONTENT = "fMRI"  # Standaard tekst voor animatie
TEXT_FONT_SIZE = 48  # Font grootte
TEXT_COLOR = FMRI_COLORS['primary']  # Tekst kleur

# Output configuratie
OUTPUT_DIR = "output"  # Directory voor gegenereerde GIFs
OUTPUT_FILENAMES = {
    'moving_figure': 'bewegend_mannetje.gif',
    'walking_figure': 'bewegend_mannetje_vrij_lopend.gif',
    'spots_appearing': 'vlekken_verschijnend.gif', 
    'spots_disappearing': 'vlekken_verdwijnend.gif',
    'text_appearing': 'tekst_verschijnend.gif'
}

# GIF optimalisatie instellingen
GIF_OPTIMIZE = True  # Optimaliseer GIF bestandsgrootte
GIF_LOOP = 0  # 0 = oneindige loop