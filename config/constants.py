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

# HEATMAP ENHANCEMENT CONFIGURATIE (NIEUW)
HEATMAP_ENHANCEMENT = {
    'enabled': True,
    
    # Intensiteit mapping
    'intensity_mapping': 'logarithmic',  # 'linear', 'logarithmic', 'exponential'
    'intensity_levels': 12,              # Aantal intensiteitsniveaus
    
    # Kleur saturatie en helderheid
    'color_saturation': 1.3,             # Verhoogde saturatie voor levendige kleuren
    'adaptive_brightness': True,         # Adaptieve helderheid per regio
    'brightness_enhancement': 1.1,       # Helderheid versterking factor
    
    # Vloeiende overgangen
    'smooth_gradients': True,            # Vloeiende kleurovergangen
    'gradient_transition_width': 3,      # Breedte van overgangszone
    'gradient_method': 'gaussian',       # 'gaussian', 'linear', 'cubic'
    
    # Gaussische blur voor heatmap-effect
    'gaussian_blur_sigma': 1.5,          # Blur sterkte voor heatmap-effect
    'blur_preserve_edges': True,         # Behoud randen tijdens blur
    
    # Hotspot enhancement
    'hotspot_enhancement': True,         # Voeg realistische hotspots toe
    'hotspot_count': 3,                  # Aantal hotspots per frame
    'hotspot_intensity_range': (0.6, 0.9), # (min, max) intensiteit voor hotspots
    'hotspot_size_range': (15, 40),      # (min, max) radius voor hotspots
    
    # Rand verzachting
    'edge_softening': 0.8,               # Sterkte van rand verzachting (0.0-1.0)
    'edge_softening_method': 'gaussian', # 'gaussian', 'bilateral', 'morphological'
    
    # Diepte simulatie
    'depth_simulation': True,            # Pseudo-3D effect voor diepte
    'depth_factor': 0.3,                 # Sterkte van diepte effect
    'depth_method': 'gradient',          # 'gradient', 'shadow', 'highlight'
    
    # Blending modes
    'blending_mode': 'screen',           # 'normal', 'screen', 'overlay', 'multiply'
    'blending_opacity': 0.8,             # Opacity voor heatmap blending
    
    # Transparantie effecten
    'transparency_levels': [0.7, 0.8, 0.9, 1.0], # Verschillende transparantie niveaus
    'intensity_based_transparency': True, # Transparantie gebaseerd op intensiteit
    
    # Contrast enhancement
    'contrast_enhancement': 1.2,         # Contrast versterking factor
    'adaptive_contrast': True,           # Adaptieve contrast verbetering
    'contrast_preserve_range': True,     # Behoud kleur bereik
    
    # Ruimtelijke gradiënten
    'spatial_gradients': True,           # Verbeter ruimtelijke gradiënten
    'gradient_strength': 0.7,            # Sterkte van gradiënt effect
    'gradient_direction': 'radial',      # 'radial', 'horizontal', 'vertical'
    
    # Wetenschappelijke kleurmapping
    'scientific_color_mapping': True,    # Gebruik wetenschappelijk accurate kleuren
    'zscore_based_mapping': True,        # Z-score gebaseerde kleurmapping
    'statistical_thresholding': True,    # Pas statistische thresholding toe
    
    # Temporele effecten
    'temporal_smoothing': True,          # Temporele smoothing tussen frames
    'temporal_correlation': 0.8,         # Correlatie tussen frames
    'temporal_variation': True,          # Tijdelijke kleurvariatie
    
    # Performance optimalisatie
    'enable_caching': True,              # Cache berekeningen
    'optimization_level': 'balanced',    # 'speed', 'quality', 'balanced'
    'parallel_processing': False,        # Parallelle verwerking (experimenteel)
    
    # Debug en validatie
    'debug_mode': False,                 # Debug informatie
    'validation_enabled': True,          # Valideer parameters
    'before_after_comparison': False,    # Genereer voor/na vergelijking
}

# fMRI Realisme configuratie (BESTAAND)
FMRI_REALISM = {
    # Voxel-achtige structuur
    'voxel_size': 2,              # Voxel grootte voor textuur (pixels)
    'voxel_enabled': True,        # Schakel voxel effecten in/uit
    'voxel_opacity': 0.3,         # Transparantie van voxel raster (0.0-1.0)
    
    # Noise patterns (realistische ruis)
    'noise_level': 0.1,           # Basis ruis niveau (0.0-1.0)
    'noise_type': 'gaussian',     # Type ruis ('gaussian', 'uniform', 'salt_pepper')
    'noise_temporal': True,       # Tijdelijke ruis variatie
    'noise_spatial': True,        # Ruimtelijke ruis variatie
    
    # Spatial smoothing
    'smoothing_kernel': 3,        # Kernel grootte voor spatial smoothing
    'smoothing_enabled': True,    # Schakel smoothing in/uit
    'smoothing_iterations': 1,    # Aantal smoothing iteraties
    'smoothing_preserve_edges': True,  # Behoud randen tijdens smoothing
    
    # Statistical thresholding
    'threshold_level': 0.3,       # Activatie drempel (0.0-1.0)
    'threshold_enabled': True,    # Schakel thresholding in/uit
    'threshold_type': 'soft',     # Type thresholding ('hard', 'soft')
    'threshold_fade': 0.1,        # Fade zone voor soft thresholding
    
    # Cluster-based activatie
    'cluster_min_size': 5,        # Minimale cluster grootte (pixels)
    'cluster_enabled': True,      # Schakel cluster detectie in/uit
    'cluster_connectivity': 8,    # Connectiviteit voor clustering (4 of 8)
    'cluster_merge_distance': 3,  # Afstand voor cluster merging
    
    # Intensity variations
    'intensity_levels': 5,        # Aantal intensiteitsniveaus binnen activatie
    'intensity_variation': 0.2,   # Variatie in intensiteit (0.0-1.0)
    'intensity_gradient': True,   # Gradiënt binnen activatiegebieden
    'intensity_temporal': True,   # Tijdelijke intensiteitsvariatie
    
    # Temporal dynamics
    'temporal_smoothing': True,   # Temporele smoothing tussen frames
    'temporal_correlation': 0.8,  # Correlatie tussen opeenvolgende frames (0.0-1.0)
    'temporal_drift': 0.05,       # Langzame drift in activatie (0.0-1.0)
    'temporal_fluctuations': True, # Realistische temporele fluctuaties
    
    # Baseline fluctuations
    'baseline_level': 0.1,        # Basis activatie niveau (0.0-1.0)
    'baseline_variation': 0.05,   # Variatie in baseline (0.0-1.0)
    'baseline_temporal': True,    # Tijdelijke baseline variatie
    'baseline_spatial': True,     # Ruimtelijke baseline variatie
    
    # Edge enhancement
    'edge_enhancement': True,     # Schakel rand verbetering in
    'edge_glow_radius': 2,        # Radius voor rand gloed
    'edge_glow_intensity': 0.4,   # Intensiteit van rand gloed (0.0-1.0)
    'edge_detection_threshold': 0.2,  # Drempel voor rand detectie
    
    # Gradient boundaries
    'gradient_boundaries': True,  # Zachte gradiënt grenzen
    'gradient_width': 5,          # Breedte van gradiënt zone (pixels)
    'gradient_falloff': 'gaussian', # Type falloff ('linear', 'gaussian', 'exponential')
    'gradient_strength': 0.7,     # Sterkte van gradiënt effect (0.0-1.0)
    
    # Multi-level intensiteit
    'multi_level_enabled': True,  # Verschillende intensiteitsniveaus binnen vorm
    'level_count': 3,             # Aantal intensiteitsniveaus
    'level_spacing': 0.3,         # Afstand tussen niveaus (0.0-1.0)
    'level_blending': 'smooth',   # Blending tussen niveaus ('sharp', 'smooth')
    
    # Anatomical realism
    'anatomical_variation': True, # Anatomische variatie in activatie
    'anatomical_asymmetry': 0.1,  # Asymmetrie factor (0.0-1.0)
    'anatomical_hotspots': True,  # Hotspot gebieden met hogere activatie
    'anatomical_gradients': True, # Anatomische gradiënten
    
    # Z-score mapping simulatie
    'zscore_mapping': True,       # Simuleer z-score gebaseerde kleuren
    'zscore_range': (-3.0, 6.0),  # Z-score bereik (min, max)
    'zscore_threshold': 2.3,      # Significantie drempel (z-score)
    'zscore_colorbar': False,     # Toon kleurenbalk (voor toekomstige uitbreiding)
    
    # Contrast enhancement
    'contrast_enhancement': True, # Verbeter contrast
    'contrast_factor': 1.2,       # Contrast versterking factor
    'contrast_adaptive': True,    # Adaptieve contrast verbetering
    'contrast_preserve_range': True, # Behoud kleur bereik
    
    # Performance instellingen
    'enable_caching': True,       # Cache berekeningen voor performance
    'parallel_processing': False, # Parallelle verwerking (experimenteel)
    'optimization_level': 'balanced', # Optimalisatie niveau ('speed', 'quality', 'balanced')
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