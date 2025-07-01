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

# fMRI kleur configuratie (origineel)
FMRI_COLORS = {
    'primary': (255, 140, 0),    # Oranje
    'secondary': (255, 69, 0),   # Rood-oranje  
    'accent': (255, 215, 0),     # Goud-geel
    'highlight': (255, 255, 0)   # Geel
}

# Uitgebreide fMRI kleurschalen (neuroimaging standaarden)
FMRI_COLOR_SCHEMES = {
    'hot': {
        'name': 'Hot (Warm Colors)',
        'description': 'Klassieke hot colormap - zwart via rood naar wit',
        'colors': [
            (0, 0, 0),        # Zwart (geen activiteit)
            (64, 0, 0),       # Donkerrood
            (128, 0, 0),      # Rood
            (192, 0, 0),      # Helderrood
            (255, 0, 0),      # Puur rood
            (255, 64, 0),     # Rood-oranje
            (255, 128, 0),    # Oranje
            (255, 192, 0),    # Geel-oranje
            (255, 255, 0),    # Geel
            (255, 255, 128),  # Lichtgeel
            (255, 255, 192),  # Zeer lichtgeel
            (255, 255, 255)   # Wit (maximale activiteit)
        ],
        'negative_colors': [
            (0, 0, 64),       # Donkerblauw
            (0, 0, 128),      # Blauw
            (0, 64, 192),     # Lichtblauw
            (0, 128, 255),    # Helder blauw
            (64, 192, 255),   # Zeer lichtblauw
            (128, 224, 255)   # Cyaan-wit
        ]
    },
    'cool': {
        'name': 'Cool (Cold Colors)', 
        'description': 'Cool colormap - cyaan via blauw naar magenta',
        'colors': [
            (0, 255, 255),    # Cyaan (lage activiteit)
            (0, 224, 255),    # Licht cyaan
            (0, 192, 255),    # Cyaan-blauw
            (0, 160, 255),    # Lichtblauw
            (0, 128, 255),    # Blauw
            (0, 96, 255),     # Donkerblauw
            (0, 64, 255),     # Diep blauw
            (32, 32, 255),    # Blauw-paars
            (64, 0, 255),     # Paars
            (128, 0, 255),    # Magenta-paars
            (192, 0, 255),    # Licht magenta
            (255, 0, 255)     # Magenta (hoge activiteit)
        ],
        'negative_colors': [
            (255, 255, 0),    # Geel
            (255, 192, 0),    # Oranje-geel
            (255, 128, 0),    # Oranje
            (255, 64, 0),     # Rood-oranje
            (255, 0, 0),      # Rood
            (192, 0, 0)       # Donkerrood
        ]
    },
    'jet': {
        'name': 'Jet (Rainbow)',
        'description': 'Jet colormap - blauw via groen/geel naar rood',
        'colors': [
            (0, 0, 128),      # Donkerblauw (lage activiteit)
            (0, 0, 192),      # Blauw
            (0, 0, 255),      # Helder blauw
            (0, 64, 255),     # Blauw-cyaan
            (0, 128, 255),    # Cyaan-blauw
            (0, 192, 255),    # Licht cyaan
            (0, 255, 255),    # Cyaan
            (0, 255, 192),    # Cyaan-groen
            (0, 255, 128),    # Groen-cyaan
            (0, 255, 64),     # Lichtgroen
            (0, 255, 0),      # Groen
            (64, 255, 0),     # Geel-groen
            (128, 255, 0),    # Lime
            (192, 255, 0),    # Geel-lime
            (255, 255, 0),    # Geel
            (255, 192, 0),    # Geel-oranje
            (255, 128, 0),    # Oranje
            (255, 64, 0),     # Rood-oranje
            (255, 0, 0),      # Rood
            (192, 0, 0),      # Donkerrood
            (128, 0, 0)       # Zeer donkerrood (hoge activiteit)
        ],
        'negative_colors': [
            (128, 0, 128),    # Paars
            (160, 0, 160),    # Licht paars
            (192, 0, 192),    # Magenta-paars
            (224, 0, 224),    # Licht magenta
            (255, 0, 255),    # Magenta
            (255, 64, 255)    # Licht magenta-wit
        ]
    },
    'viridis': {
        'name': 'Viridis (Perceptually Uniform)',
        'description': 'Moderne perceptueel uniforme colormap - paars via blauw/groen naar geel',
        'colors': [
            (68, 1, 84),      # Donkerpaars (lage activiteit)
            (72, 35, 116),    # Paars-blauw
            (64, 67, 135),    # Blauw-paars
            (52, 94, 141),    # Donkerblauw
            (41, 120, 142),   # Blauw
            (32, 144, 140),   # Blauw-groen
            (34, 167, 132),   # Groen-blauw
            (68, 190, 112),   # Groen
            (121, 209, 81),   # Lichtgroen
            (189, 222, 38),   # Geel-groen
            (253, 231, 37)    # Geel (hoge activiteit)
        ],
        'negative_colors': [
            (128, 0, 128),    # Magenta
            (160, 32, 128),   # Magenta-rood
            (192, 64, 128),   # Roze-magenta
            (224, 96, 128),   # Roze
            (255, 128, 128),  # Lichtroze
            (255, 160, 160)   # Zeer lichtroze
        ]
    }
}

# Standaard kleurschema selectie
DEFAULT_COLOR_SCHEME = 'hot'  # Kan worden gewijzigd naar 'cool', 'jet', of 'viridis'

# Kleur mapping configuratie
COLOR_MAPPING = {
    'intensity_levels': 12,        # Aantal intensiteitsniveaus voor kleurmapping
    'smooth_transitions': True,    # Vloeiende overgangen tussen kleuren
    'use_negative_colors': True,   # Gebruik negatieve kleuren voor deactivatie
    'temporal_variation': True,    # Tijdelijke kleurvariatie in animaties
    'gradient_steps': 256         # Aantal stappen in kleurgradiënten
}

# Gloed effect instellingen (uitgebreid)
GLOW_RADIUS = 10  # Radius van gloed effect
GLOW_INTENSITY = 0.7  # Intensiteit van gloed (0.0 - 1.0)
ENHANCED_GLOW_RADIUS = 15  # Verhoogde gloed radius voor nieuwe effecten
ENHANCED_GLOW_INTENSITY = 0.9  # Verhoogde gloed intensiteit

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

# Vlekken animatie configuratie (uitgebreid)
SPOT_MIN_SIZE = 5  # Minimale grootte van vlekken
SPOT_MAX_SIZE = 25  # Maximale grootte van vlekken
SPOT_COUNT = 8  # Aantal vlekken per frame
SPOT_COLOR_VARIATION = True  # Variatie in vlekkleur gebaseerd op grootte/tijd
SPOT_INTENSITY_MAPPING = True  # Map vlekgrootte naar kleurintensiteit

# Tekst animatie configuratie (uitgebreid)
TEXT_CONTENT = "fMRI"  # Standaard tekst voor animatie
TEXT_FONT_SIZE = 48  # Font grootte
TEXT_COLOR = FMRI_COLORS['primary']  # Tekst kleur (oranje)
TEXT_POSITION = "center"  # Positionering binnen ovaal ("center", "left", "right", "top", "bottom")
TEXT_ANIMATION_SPEED = 0.15  # Snelheid van letter onthulling (seconden per letter)
TEXT_GLOW_RADIUS = 15  # Gloed radius voor tekst (groter dan standaard voor meer effect)
TEXT_GLOW_INTENSITY = 0.8  # Gloed intensiteit voor tekst
TEXT_PULSE_SPEED = 2.0  # Snelheid van pulsing effect (cycli per seconde)
TEXT_FONT_PATH = None  # Pad naar custom font (None = standaard font)
TEXT_COLOR_SCHEME_BASED = True  # Gebruik kleurschema voor tekstkleuren

# Dynamische kleurverandering configuratie
DYNAMIC_COLORS = {
    'movement_based': True,       # Kleurverandering gebaseerd op bewegingssnelheid
    'time_based': True,          # Kleurverandering gebaseerd op tijd
    'activity_based': True,      # Kleurverandering gebaseerd op activiteitsniveau
    'pulsing_enabled': True,     # Pulserende kleureffecten
    'gradient_animation': True,  # Geanimeerde kleurgradiënten
    'intensity_variation': 0.3   # Variatie in kleurintensiteit (0.0-1.0)
}

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