"""
Vlekken Verdwijnend Animatie voor BewegendeAnimaties.

Deze module implementeert een animatie waarbij vlekken geleidelijk verdwijnen
van vol naar leeg - het omgekeerde proces van vlekken_verschijnend.py.
De animatie simuleert afnemende hersenactiviteit, inhibitie of herstelprocessen
en gebruikt enhanced fMRI-stijl kleurschalen met dynamische kleureffecten.
"""

import math
import random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from utils.image_utils import (
    load_background_image, create_oval_mask, apply_oval_mask,
    get_oval_bounds, point_in_oval, generate_random_position_in_oval
)
from utils.color_utils import (
    get_fmri_color, create_fmri_gradient, add_glow_effect,
    create_pulsing_color, map_value_to_color, create_activity_based_color,
    apply_temporal_color_variation, create_intensity_mapped_color
)
from utils.gif_utils import create_gif_from_frames
from config.constants import (
    SPOT_MIN_SIZE, SPOT_MAX_SIZE, SPOT_COUNT, ANIMATION_DURATION,
    FRAMES_PER_SECOND, TOTAL_FRAMES, OUTPUT_FILENAMES,
    OVAL_CENTER, OVAL_WIDTH, OVAL_HEIGHT, DEFAULT_COLOR_SCHEME,
    SPOT_COLOR_VARIATION, SPOT_INTENSITY_MAPPING, DYNAMIC_COLORS
)


def generate_spot_positions(count=None, seed=None):
    """
    Genereert startposities voor vlekken binnen het ovaal.
    
    Args:
        count (int): Aantal vlekken (standaard uit constants)
        seed (int): Random seed voor reproduceerbare resultaten
        
    Returns:
        list: Lijst van (x, y) posities voor vlekken
    """
    if count is None:
        count = SPOT_COUNT
    
    if seed is not None:
        random.seed(seed)
    
    positions = []
    for _ in range(count):
        # Genereer willekeurige positie binnen ovaal
        position = generate_random_position_in_oval(margin=20)
        positions.append(position)
    
    return positions


def calculate_spot_shrinkage(frame_number, total_frames, spot_index, total_spots):
    """
    Berekent de grootte van een vlek voor een specifiek frame tijdens verdwijning.
    
    Args:
        frame_number (int): Huidige frame nummer (0-based)
        total_frames (int): Totaal aantal frames
        spot_index (int): Index van de vlek
        total_spots (int): Totaal aantal vlekken
        
    Returns:
        float: Grootte van de vlek (1.0 = maximale grootte, 0.0 = verdwenen)
    """
    # Bereken voortgang van animatie (0.0 - 1.0)
    animation_progress = frame_number / (total_frames - 1) if total_frames > 1 else 0
    
    # Vlekken verdwijnen geleidelijk - elke vlek start verdwijning op een ander moment
    spot_start_time = (spot_index / total_spots) * 0.4  # Eerste 40% van animatie starten vlekken met verdwijnen
    spot_shrink_duration = 0.8  # Elke vlek verdwijnt over 80% van de tijd
    
    # Bereken lokale voortgang voor deze vlek
    if animation_progress < spot_start_time:
        return 1.0  # Vlek nog op maximale grootte
    
    local_progress = (animation_progress - spot_start_time) / spot_shrink_duration
    local_progress = max(0.0, min(1.0, local_progress))
    
    # Gebruik easing functie voor natuurlijke verdwijning
    # Sigmoid curve voor smooth start en eind
    if local_progress <= 0:
        return 1.0
    elif local_progress >= 1:
        return 0.0
    else:
        # Sigmoid easing - omgekeerd voor verdwijning
        sigmoid = 1 / (1 + math.exp(-10 * (local_progress - 0.5)))
        return 1.0 - sigmoid  # Omgekeerd: start bij 1.0, eindig bij 0.0


def calculate_disappearing_intensity(shrinkage_factor, spot_index, total_spots, time_factor):
    """
    Berekent de kleurintensiteit van een verdwijnende vlek.
    
    Args:
        shrinkage_factor (float): Grootte factor van de vlek (1.0-0.0, verdwijnend)
        spot_index (int): Index van de vlek
        total_spots (int): Totaal aantal vlekken
        time_factor (float): Tijd factor voor dynamische effecten
        
    Returns:
        float: Intensiteitswaarde voor kleurmapping (0.0-1.0)
    """
    if not SPOT_INTENSITY_MAPPING:
        return shrinkage_factor  # Gebruik grootte als intensiteit
    
    # Basis intensiteit gebaseerd op resterende grootte
    base_intensity = shrinkage_factor
    
    # Voeg variatie toe gebaseerd op vlek positie
    spot_variation = (spot_index / total_spots) * 0.2
    
    # Voeg tijdelijke variatie toe (minder dan bij verschijnende vlekken)
    if DYNAMIC_COLORS.get('time_based', True):
        time_variation = math.sin(time_factor * 2 * math.pi + spot_index) * 0.15
        base_intensity += time_variation
    
    # Voor verdwijnende vlekken: hoe kleiner, hoe intenser de resterende kleur
    # Dit simuleert concentratie van activiteit
    if shrinkage_factor > 0:
        concentration_factor = 1.0 + (1.0 - shrinkage_factor) * 0.3
        base_intensity *= concentration_factor
    
    # Combineer alle factoren
    final_intensity = base_intensity + spot_variation
    
    # Clamp tussen 0 en 1
    return max(0.0, min(1.0, final_intensity))


def create_organic_spot(size, color=None, irregularity=0.3):
    """
    Creëert een natuurlijk uitziende vlek vorm.
    
    Args:
        size (int): Basis grootte van de vlek
        color (tuple): RGB kleur (standaard fMRI primary)
        irregularity (float): Mate van onregelmatigheid (0.0-1.0)
        
    Returns:
        PIL.Image: Afbeelding van organische vlek
    """
    if color is None:
        color = get_fmri_color('primary')
    
    # Maak afbeelding iets groter dan vlek voor irregulariteit
    img_size = int(size * 1.5)
    spot_img = Image.new('RGBA', (img_size, img_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(spot_img)
    
    center = (img_size // 2, img_size // 2)
    
    # Genereer punten voor organische vorm
    num_points = 12  # Aantal punten voor de vorm
    points = []
    
    for i in range(num_points):
        angle = (i / num_points) * 2 * math.pi
        
        # Basis radius met irregulariteit
        base_radius = size / 2
        variation = random.uniform(-irregularity, irregularity) * base_radius
        radius = base_radius + variation
        
        # Bereken punt
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    
    # Teken gevulde polygon voor organische vorm
    draw.polygon(points, fill=color + (255,))
    
    # Voeg zachte randen toe met blur
    spot_img = spot_img.filter(ImageFilter.GaussianBlur(radius=1))
    
    return spot_img


def get_disappearing_spot_color(spot_index, total_spots, shrinkage_factor, time_factor, color_scheme=None):
    """
    Bepaalt de kleur van een verdwijnende vlek gebaseerd op verschillende factoren.
    
    Args:
        spot_index (int): Index van de vlek
        total_spots (int): Totaal aantal vlekken
        shrinkage_factor (float): Grootte factor van de vlek (1.0-0.0)
        time_factor (float): Tijd factor voor dynamische effecten
        color_scheme (str): Naam van het kleurschema
        
    Returns:
        tuple: RGB kleur tuple
    """
    if color_scheme is None:
        color_scheme = DEFAULT_COLOR_SCHEME
    
    if SPOT_COLOR_VARIATION:
        # Bereken intensiteit gebaseerd op verdwijning en tijd
        intensity = calculate_disappearing_intensity(shrinkage_factor, spot_index, total_spots, time_factor)
        
        # Map intensiteit naar kleur uit kleurschema
        base_color = create_intensity_mapped_color(intensity, color_scheme)
        
        # Voeg tijdelijke variatie toe (minder dan bij verschijnende vlekken)
        if DYNAMIC_COLORS.get('time_based', True):
            base_color = apply_temporal_color_variation(base_color, time_factor, variation_intensity=0.2)
        
        return base_color
    else:
        # Gebruik standaard fMRI gradient
        color_gradient = create_fmri_gradient(steps=max(3, total_spots // 2), color_scheme=color_scheme)
        color_index = spot_index % len(color_gradient)
        return color_gradient[color_index]


def render_disappearing_spots_frame(frame_number, spot_positions, background_size, color_scheme=None):
    """
    Rendert één frame van de verdwijnende vlekken animatie.
    
    Args:
        frame_number (int): Frame nummer om te renderen
        spot_positions (list): Lijst van vlek posities
        background_size (tuple): (width, height) van achtergrond
        color_scheme (str): Naam van het kleurschema
        
    Returns:
        PIL.Image: Gerenderd frame
    """
    if color_scheme is None:
        color_scheme = DEFAULT_COLOR_SCHEME
    
    # Maak transparante overlay voor vlekken
    overlay = Image.new('RGBA', background_size, (0, 0, 0, 0))
    
    total_spots = len(spot_positions)
    time_factor = frame_number / TOTAL_FRAMES
    
    for spot_index, position in enumerate(spot_positions):
        # Bereken grootte voor deze vlek in dit frame (verdwijning)
        shrinkage_factor = calculate_spot_shrinkage(
            frame_number, TOTAL_FRAMES, spot_index, total_spots
        )
        
        if shrinkage_factor <= 0:
            continue  # Vlek volledig verdwenen
        
        # Bereken werkelijke grootte
        min_size = SPOT_MIN_SIZE
        max_size = SPOT_MAX_SIZE
        current_size = int(min_size + (max_size - min_size) * shrinkage_factor)
        
        # Bepaal kleur gebaseerd op verdwijning en andere factoren
        base_color = get_disappearing_spot_color(
            spot_index, total_spots, shrinkage_factor, time_factor, color_scheme
        )
        
        # Voeg pulsing effect toe voor meer dynamiek (minder intensief dan bij verschijnende vlekken)
        if DYNAMIC_COLORS.get('pulsing_enabled', True):
            pulse_position = (time_factor + spot_index * 0.1) % 1.0
            spot_color = create_pulsing_color(base_color, pulse_position * 0.7, color_scheme)  # Minder pulsing
        else:
            spot_color = base_color
        
        # Creëer organische vlek
        spot = create_organic_spot(current_size, spot_color)
        
        # Bereken positie om vlek te centreren
        spot_x = position[0] - spot.size[0] // 2
        spot_y = position[1] - spot.size[1] // 2
        
        # Plak vlek op overlay
        overlay.paste(spot, (spot_x, spot_y), spot)
    
    # Voeg enhanced gloed effect toe aan hele overlay (minder intensief dan bij verschijnende vlekken)
    if not all(pixel[3] == 0 for pixel in overlay.getdata()):  # Als er vlekken zijn
        overlay = add_glow_effect(overlay, enhanced=True, intensity=0.5)
    
    return overlay


def create_spots_disappearing_animation(output_filename=None, seed=None, color_scheme=None):
    """
    Creëert een animatie van vlekken die geleidelijk verdwijnen.
    
    Args:
        output_filename (str): Naam van output GIF (standaard uit constants)
        seed (int): Random seed voor reproduceerbare resultaten
        color_scheme (str): Naam van het kleurschema
        
    Returns:
        str: Pad naar gegenereerde GIF
    """
    if output_filename is None:
        output_filename = OUTPUT_FILENAMES['spots_disappearing']
    if color_scheme is None:
        color_scheme = DEFAULT_COLOR_SCHEME
    
    print(f"Genereren vlekken verdwijnend animatie...")
    print(f"- {TOTAL_FRAMES} frames @ {FRAMES_PER_SECOND} FPS")
    print(f"- {SPOT_COUNT} vlekken")
    print(f"- Grootte: {SPOT_MIN_SIZE}-{SPOT_MAX_SIZE} pixels")
    print(f"- Kleurschema: {color_scheme}")
    print(f"- Kleurvariatie: {SPOT_COLOR_VARIATION}")
    print(f"- Intensiteit mapping: {SPOT_INTENSITY_MAPPING}")
    
    # Laad achtergrond en maak masker
    background = load_background_image()
    oval_mask = create_oval_mask(background.size)
    
    # Genereer vlek posities
    spot_positions = generate_spot_positions(seed=seed)
    
    frames = []
    
    # Genereer alle frames
    for frame_num in range(TOTAL_FRAMES):
        print(f"Renderen frame {frame_num + 1}/{TOTAL_FRAMES}...", end='\r')
        
        # Render vlekken voor dit frame
        spots_overlay = render_disappearing_spots_frame(frame_num, spot_positions, background.size, color_scheme)
        
        # Pas ovaal masker toe op vlekken
        masked_spots = apply_oval_mask(spots_overlay, oval_mask)
        
        # Combineer met achtergrond
        frame = background.copy()
        frame.paste(masked_spots, (0, 0), masked_spots)
        
        frames.append(frame)
    
    print(f"\nOpslaan als {output_filename}...")
    
    # Maak GIF
    gif_path = create_gif_from_frames(frames, output_filename)
    
    print(f"✅ Vlekken verdwijnend animatie voltooid: {gif_path}")
    return gif_path


def create_demo_animation():
    """
    Creëert een demo versie van de vlekken verdwijnend animatie voor testen.
    
    Returns:
        str: Pad naar demo GIF
    """
    return create_spots_disappearing_animation(
        output_filename="demo_vlekken_verdwijnend.gif",
        seed=42  # Vaste seed voor consistente demo
    )


def create_color_scheme_demos():
    """
    Creëert demo animaties voor alle beschikbare kleurschema's.
    
    Returns:
        list: Lijst van paden naar gegenereerde GIFs
    """
    schemes = ['hot', 'cool', 'jet', 'viridis']
    demo_paths = []
    
    print("\n=== Genereren vlekken verdwijnend kleurschema demo's ===")
    
    for scheme in schemes:
        print(f"\n--- Kleurschema: {scheme} ---")
        output_filename = f"vlekken_verdwijnend_{scheme}.gif"
        
        path = create_spots_disappearing_animation(
            output_filename=output_filename,
            seed=42,  # Consistente seed voor vergelijking
            color_scheme=scheme
        )
        demo_paths.append(path)
    
    print(f"\n✅ {len(demo_paths)} kleurschema demo's voltooid")
    return demo_paths


if __name__ == "__main__":
    # Test de enhanced fMRI kleuren functionaliteit
    try:
        print("=== Test Enhanced fMRI Kleuren voor Vlekken Verdwijnend ===")
        
        # Test standaard animatie
        demo_path = create_demo_animation()
        print(f"Demo animatie: {demo_path}")
        
        # Test verschillende kleurschema's
        scheme_demos = create_color_scheme_demos()
        print(f"Kleurschema demo's: {scheme_demos}")
        
    except Exception as e:
        print(f"Test gefaald: {str(e)}")