"""
Bewegend mannetje animatie module voor fMRI-stijl visualisaties.

Deze module implementeert zowel een bewegend figuur dat een ovaalroute volgt als
een realistisch lopend mannetje dat vrij beweegt binnen het hersengebied,
met fMRI kleurenschema en gloed effecten.
"""

import math
import random
import numpy as np
from PIL import Image, ImageDraw
from utils.image_utils import (
    load_background_image, create_oval_mask, apply_oval_mask, composite_images,
    create_stick_figure, generate_random_position_in_oval, calculate_next_position,
    ensure_within_oval, is_position_in_oval
)
from utils.color_utils import get_fmri_color, add_glow_effect, create_colored_circle, create_pulsing_color
from utils.gif_utils import create_gif_from_frames
from config.constants import (
    OVAL_CENTER, OVAL_WIDTH, OVAL_HEIGHT, TOTAL_FRAMES, 
    MOVING_FIGURE_SIZE, MOVING_FIGURE_SPEED, OUTPUT_FILENAMES,
    WALKING_FIGURE_SIZE, WALKING_SPEED, WALKING_DIRECTION_CHANGE_CHANCE,
    WALKING_RANDOM_VARIATION, WALKING_POSE_CHANGE_SPEED, WALKING_BOUNDARY_MARGIN,
    RANDOM_WALK_STEP_SIZE, RANDOM_WALK_MOMENTUM, RANDOM_WALK_DIRECTION_NOISE
)


def calculate_oval_position(progress, clockwise=True, route_variation=0):
    """
    Berekent positie op ovaalroute op basis van voortgang.
    
    Args:
        progress (float): Voortgang van 0.0 tot 1.0
        clockwise (bool): True voor rechtsom, False voor linksom
        route_variation (int): Variatie in route (0=standaard, 1=elliptisch, 2=spiraal)
        
    Returns:
        tuple: (x, y) positie op ovaalroute
    """
    # Basis hoek berekening
    angle = progress * 2 * math.pi
    if not clockwise:
        angle = -angle
    
    # Route variaties
    if route_variation == 1:  # Meer elliptisch
        width_factor = 0.8
        height_factor = 1.2
    elif route_variation == 2:  # Spiraal effect
        width_factor = 1.0 + 0.1 * math.sin(progress * 4 * math.pi)
        height_factor = 1.0 + 0.1 * math.cos(progress * 4 * math.pi)
    else:  # Standaard ovaal
        width_factor = 1.0
        height_factor = 1.0
    
    # Bereken positie binnen ovaal
    x = OVAL_CENTER[0] + (OVAL_WIDTH / 2 * width_factor) * math.cos(angle)
    y = OVAL_CENTER[1] + (OVAL_HEIGHT / 2 * height_factor) * math.sin(angle)
    
    return (int(x), int(y))


def create_moving_figure(size=None, color=None, pulse_position=0.0):
    """
    Creëert een bewegend figuur met fMRI styling (originele cirkel versie).
    
    Args:
        size (int): Grootte van het figuur (standaard uit constants)
        color (tuple): RGB kleur (standaard fMRI primary)
        pulse_position (float): Positie voor pulsing effect (0.0-1.0)
        
    Returns:
        PIL.Image: Figuur afbeelding met transparante achtergrond
    """
    if size is None:
        size = MOVING_FIGURE_SIZE
    if color is None:
        color = get_fmri_color('primary')
    
    # Creëer pulserende kleur
    pulsing_color = create_pulsing_color(color, pulse_position)
    
    # Maak basis cirkel
    figure = create_colored_circle(size, pulsing_color, alpha=220)
    
    # Voeg gloed effect toe
    figure_with_glow = add_glow_effect(figure, pulsing_color, radius=8, intensity=0.6)
    
    return figure_with_glow


def create_walking_figure(pose_frame=0, size=None, color=None, pulse_position=0.0):
    """
    Creëert een realistisch lopend mannetje figuur met fMRI styling.
    
    Args:
        pose_frame (int): Frame nummer voor loop animatie
        size (int): Grootte van het mannetje (standaard uit constants)
        color (tuple): RGB kleur (standaard fMRI primary)
        pulse_position (float): Positie voor pulsing effect (0.0-1.0)
        
    Returns:
        PIL.Image: Stick figure afbeelding met transparante achtergrond en gloed
    """
    if size is None:
        size = WALKING_FIGURE_SIZE
    if color is None:
        color = get_fmri_color('primary')
    
    # Creëer pulserende kleur
    pulsing_color = create_pulsing_color(color, pulse_position)
    
    # Maak stick figure
    figure = create_stick_figure(pose_frame, size, pulsing_color)
    
    # Voeg gloed effect toe
    figure_with_glow = add_glow_effect(figure, pulsing_color, radius=6, intensity=0.5)
    
    return figure_with_glow


def generate_random_walk_path(total_frames, start_position=None):
    """
    Genereert een natuurlijke random walk route binnen het ovaal.
    
    Args:
        total_frames (int): Totaal aantal frames voor de route
        start_position (tuple): Start positie (willekeurig als None)
        
    Returns:
        list: Lijst van (x, y) posities voor elk frame
    """
    if start_position is None:
        start_position = generate_random_position_in_oval(margin=WALKING_BOUNDARY_MARGIN)
    
    positions = [start_position]
    current_direction = random.uniform(0, 2 * math.pi)
    current_speed = WALKING_SPEED
    
    for frame in range(1, total_frames):
        current_pos = positions[-1]
        
        # Voeg willekeurige variatie toe aan richting
        direction_change = random.uniform(-RANDOM_WALK_DIRECTION_NOISE, RANDOM_WALK_DIRECTION_NOISE)
        current_direction += direction_change
        
        # Kans op grotere richtingsverandering
        if random.random() < WALKING_DIRECTION_CHANGE_CHANCE:
            current_direction += random.uniform(-math.pi/3, math.pi/3)
        
        # Voeg momentum toe (behoud van richting)
        if frame > 1:
            prev_direction = math.atan2(
                positions[-1][1] - positions[-2][1],
                positions[-1][0] - positions[-2][0]
            )
            current_direction = (
                RANDOM_WALK_MOMENTUM * prev_direction + 
                (1 - RANDOM_WALK_MOMENTUM) * current_direction
            )
        
        # Normaliseer richting
        current_direction = current_direction % (2 * math.pi)
        
        # Varieer snelheid lichtjes
        speed_variation = 1.0 + random.uniform(-WALKING_RANDOM_VARIATION, WALKING_RANDOM_VARIATION)
        frame_speed = current_speed * speed_variation
        
        # Bereken volgende positie
        next_pos, new_direction = calculate_next_position(
            current_pos, current_direction, frame_speed,
            OVAL_CENTER, OVAL_WIDTH - WALKING_BOUNDARY_MARGIN, OVAL_HEIGHT - WALKING_BOUNDARY_MARGIN
        )
        
        # Update richting als er een bounce was
        current_direction = new_direction
        
        # Zorg ervoor dat positie binnen ovaal blijft
        next_pos = ensure_within_oval(
            next_pos, OVAL_CENTER, 
            OVAL_WIDTH - WALKING_BOUNDARY_MARGIN, 
            OVAL_HEIGHT - WALKING_BOUNDARY_MARGIN
        )
        
        positions.append(next_pos)
    
    return positions


def generate_animation_frames(clockwise=True, route_variation=0, speed_multiplier=None):
    """
    Genereert alle frames voor de bewegend mannetje animatie (originele ovaal route).
    
    Args:
        clockwise (bool): Bewegingsrichting
        route_variation (int): Route variatie (0-2)
        speed_multiplier (float): Snelheid multiplier (standaard uit constants)
        
    Returns:
        list: Lijst van PIL.Image frames
    """
    if speed_multiplier is None:
        speed_multiplier = MOVING_FIGURE_SPEED
    
    # Laad achtergrond en maak masker
    background = load_background_image()
    oval_mask = create_oval_mask(background.size)
    
    frames = []
    
    for frame_num in range(TOTAL_FRAMES):
        # Bereken voortgang met snelheid multiplier
        base_progress = frame_num / TOTAL_FRAMES
        progress = (base_progress * speed_multiplier) % 1.0
        
        # Bereken positie op ovaalroute
        x, y = calculate_oval_position(progress, clockwise, route_variation)
        
        # Creëer bewegend figuur met pulsing effect
        pulse_position = (frame_num / TOTAL_FRAMES) * 4  # 4 pulses per cyclus
        figure = create_moving_figure(pulse_position=pulse_position)
        
        # Maak frame met achtergrond
        frame = background.copy()
        
        # Bereken figuur positie (centreer figuur op berekende positie)
        figure_x = x - figure.width // 2
        figure_y = y - figure.height // 2
        
        # Plak figuur op frame
        frame.paste(figure, (figure_x, figure_y), figure)
        
        # Pas ovaal masker toe (alleen binnen hersenregio)
        masked_overlay = apply_oval_mask(frame, oval_mask)
        
        # Combineer met originele achtergrond
        final_frame = background.copy()
        final_frame = composite_images(final_frame, masked_overlay)
        
        frames.append(final_frame)
    
    return frames


def generate_walking_animation_frames(start_position=None):
    """
    Genereert alle frames voor de realistische lopende mannetje animatie.
    
    Args:
        start_position (tuple): Start positie (willekeurig als None)
        
    Returns:
        list: Lijst van PIL.Image frames
    """
    # Laad achtergrond en maak masker
    background = load_background_image()
    oval_mask = create_oval_mask(background.size)
    
    # Genereer random walk pad
    positions = generate_random_walk_path(TOTAL_FRAMES, start_position)
    
    frames = []
    
    for frame_num in range(TOTAL_FRAMES):
        x, y = positions[frame_num]
        
        # Bereken pose frame voor loop animatie
        pose_frame = (frame_num // WALKING_POSE_CHANGE_SPEED) % 6
        
        # Creëer lopend mannetje met pulsing effect
        pulse_position = (frame_num / TOTAL_FRAMES) * 3  # 3 pulses per cyclus
        figure = create_walking_figure(pose_frame=pose_frame, pulse_position=pulse_position)
        
        # Maak frame met achtergrond
        frame = background.copy()
        
        # Bereken figuur positie (centreer figuur op berekende positie)
        figure_x = x - figure.width // 2
        figure_y = y - figure.height // 2
        
        # Plak figuur op frame
        frame.paste(figure, (figure_x, figure_y), figure)
        
        # Pas ovaal masker toe (alleen binnen hersenregio)
        masked_overlay = apply_oval_mask(frame, oval_mask)
        
        # Combineer met originele achtergrond
        final_frame = background.copy()
        final_frame = composite_images(final_frame, masked_overlay)
        
        frames.append(final_frame)
    
    return frames


def create_smooth_interpolated_frames(frames, interpolation_factor=2):
    """
    Creëert extra frames tussen bestaande frames voor smoothere animatie.
    
    Args:
        frames (list): Originele frames
        interpolation_factor (int): Aantal extra frames tussen elk paar
        
    Returns:
        list: Uitgebreide lijst met geïnterpoleerde frames
    """
    if interpolation_factor <= 1:
        return frames
    
    smooth_frames = []
    
    for i in range(len(frames)):
        smooth_frames.append(frames[i])
        
        # Voeg geïnterpoleerde frames toe (behalve na laatste frame)
        if i < len(frames) - 1:
            current_frame = frames[i]
            next_frame = frames[(i + 1) % len(frames)]
            
            for j in range(1, interpolation_factor):
                # Simpele alpha blending voor interpolatie
                alpha = j / interpolation_factor
                
                # Converteer naar numpy arrays voor berekening
                current_array = np.array(current_frame.convert('RGBA'))
                next_array = np.array(next_frame.convert('RGBA'))
                
                # Lineaire interpolatie
                interpolated_array = (
                    current_array * (1 - alpha) + next_array * alpha
                ).astype(np.uint8)
                
                interpolated_frame = Image.fromarray(interpolated_array, 'RGBA')
                smooth_frames.append(interpolated_frame)
    
    return smooth_frames


def genereer_bewegend_mannetje_animatie(
    clockwise=True, 
    route_variation=0, 
    speed_multiplier=None,
    smooth_interpolation=False,
    output_filename=None
):
    """
    Hoofdfunctie voor het genereren van bewegend mannetje animatie (originele ovaal route).
    
    Args:
        clockwise (bool): Bewegingsrichting (True=rechtsom, False=linksom)
        route_variation (int): Route variatie (0=standaard, 1=elliptisch, 2=spiraal)
        speed_multiplier (float): Snelheid multiplier (standaard uit constants)
        smooth_interpolation (bool): Extra frames voor smoothere animatie
        output_filename (str): Naam van output bestand (standaard uit constants)
        
    Returns:
        str: Pad naar gegenereerde GIF
        
    Raises:
        Exception: Als er een fout optreedt tijdens generatie
    """
    try:
        print("Genereren van bewegend mannetje animatie (ovaal route)...")
        print(f"- Richting: {'Rechtsom' if clockwise else 'Linksom'}")
        print(f"- Route variatie: {route_variation}")
        print(f"- Snelheid multiplier: {speed_multiplier or MOVING_FIGURE_SPEED}")
        print(f"- Smooth interpolatie: {smooth_interpolation}")
        
        # Genereer basis frames
        frames = generate_animation_frames(clockwise, route_variation, speed_multiplier)
        print(f"- {len(frames)} basis frames gegenereerd")
        
        # Voeg smooth interpolatie toe indien gewenst
        if smooth_interpolation:
            frames = create_smooth_interpolated_frames(frames, interpolation_factor=2)
            print(f"- {len(frames)} frames na interpolatie")
        
        # Bepaal output bestandsnaam
        if output_filename is None:
            output_filename = OUTPUT_FILENAMES['moving_figure']
        
        # Creëer GIF
        output_path = create_gif_from_frames(frames, output_filename)
        print(f"✅ Animatie succesvol gegenereerd: {output_path}")
        
        return output_path
        
    except Exception as e:
        error_msg = f"Fout bij genereren bewegend mannetje animatie: {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)


def genereer_lopend_mannetje_animatie(
    start_position=None,
    smooth_interpolation=False,
    output_filename=None
):
    """
    Hoofdfunctie voor het genereren van realistisch lopend mannetje animatie.
    
    Args:
        start_position (tuple): Start positie (willekeurig als None)
        smooth_interpolation (bool): Extra frames voor smoothere animatie
        output_filename (str): Naam van output bestand (standaard uit constants)
        
    Returns:
        str: Pad naar gegenereerde GIF
        
    Raises:
        Exception: Als er een fout optreedt tijdens generatie
    """
    try:
        print("Genereren van realistisch lopend mannetje animatie...")
        print(f"- Start positie: {start_position or 'Willekeurig'}")
        print(f"- Mannetje grootte: {WALKING_FIGURE_SIZE}")
        print(f"- Bewegingssnelheid: {WALKING_SPEED} pixels/frame")
        print(f"- Smooth interpolatie: {smooth_interpolation}")
        
        # Genereer basis frames
        frames = generate_walking_animation_frames(start_position)
        print(f"- {len(frames)} basis frames gegenereerd")
        
        # Voeg smooth interpolatie toe indien gewenst
        if smooth_interpolation:
            frames = create_smooth_interpolated_frames(frames, interpolation_factor=2)
            print(f"- {len(frames)} frames na interpolatie")
        
        # Bepaal output bestandsnaam
        if output_filename is None:
            output_filename = OUTPUT_FILENAMES['walking_figure']
        
        # Creëer GIF
        output_path = create_gif_from_frames(frames, output_filename)
        print(f"✅ Realistisch lopend mannetje animatie succesvol gegenereerd: {output_path}")
        
        return output_path
        
    except Exception as e:
        error_msg = f"Fout bij genereren lopend mannetje animatie: {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)


def create_demo_variations():
    """
    Creëert demo variaties van de bewegend mannetje animaties.
    
    Returns:
        list: Lijst van paden naar gegenereerde GIFs
    """
    variations = []
    
    try:
        # Originele ovaal route variaties
        print("\n=== Genereren originele ovaal route variaties ===")
        
        # Standaard rechtsom
        path1 = genereer_bewegend_mannetje_animatie(
            clockwise=True, 
            route_variation=0,
            output_filename="bewegend_mannetje_rechtsom.gif"
        )
        variations.append(path1)
        
        # Linksom
        path2 = genereer_bewegend_mannetje_animatie(
            clockwise=False, 
            route_variation=0,
            output_filename="bewegend_mannetje_linksom.gif"
        )
        variations.append(path2)
        
        # Nieuwe realistisch lopend mannetje variaties
        print("\n=== Genereren realistisch lopend mannetje variaties ===")
        
        # Standaard vrije beweging
        path3 = genereer_lopend_mannetje_animatie(
            output_filename="bewegend_mannetje_vrij_lopend.gif"
        )
        variations.append(path3)
        
        # Vrije beweging met smooth interpolatie
        path4 = genereer_lopend_mannetje_animatie(
            smooth_interpolation=True,
            output_filename="bewegend_mannetje_vrij_lopend_smooth.gif"
        )
        variations.append(path4)
        
        # Vrije beweging vanaf centrum
        center_start = OVAL_CENTER
        path5 = genereer_lopend_mannetje_animatie(
            start_position=center_start,
            output_filename="bewegend_mannetje_vanaf_centrum.gif"
        )
        variations.append(path5)
        
        print(f"\n✅ {len(variations)} demo variaties succesvol gegenereerd")
        return variations
        
    except Exception as e:
        print(f"❌ Fout bij genereren demo variaties: {str(e)}")
        return variations


if __name__ == "__main__":
    # Test de nieuwe functionaliteit
    try:
        print("=== Test Realistisch Lopend Mannetje ===")
        output_path = genereer_lopend_mannetje_animatie()
        print(f"Test animatie gegenereerd: {output_path}")
        
        # Genereer ook demo variaties
        print("\n=== Genereren Demo Variaties ===")
        demo_paths = create_demo_variations()
        print(f"Demo variaties: {demo_paths}")
        
    except Exception as e:
        print(f"Test gefaald: {str(e)}")