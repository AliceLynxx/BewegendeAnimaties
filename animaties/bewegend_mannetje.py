"""
Bewegend mannetje animatie module voor fMRI-stijl visualisaties.

Deze module implementeert zowel een bewegend figuur dat een ovaalroute volgt als
een realistisch lopend mannetje dat vrij beweegt binnen het hersengebied,
met uitgebreide fMRI kleurenschema's, dynamische kleureffecten en fMRI-realisme.
"""

import math
import random
import numpy as np
from PIL import Image, ImageDraw
from utils.image_utils import (
    load_background_image, create_oval_mask, apply_oval_mask, composite_images,
    create_stick_figure, generate_random_position_in_oval, calculate_next_position,
    ensure_within_oval, is_position_in_oval, render_voxel_texture,
    apply_spatial_smoothing_image, enhance_edges_fmri_style, create_gradient_boundaries,
    add_anatomical_variation
)
from utils.color_utils import (
    get_fmri_color, add_glow_effect, create_colored_circle, create_pulsing_color,
    create_movement_based_color, create_activity_based_color, map_value_to_color,
    create_gradient_animation_color, get_color_scheme, apply_temporal_color_variation,
    enhance_fmri_realism, generate_fmri_noise, apply_statistical_thresholding,
    detect_activation_clusters, simulate_zscore_mapping, apply_temporal_correlation
)
from utils.gif_utils import create_gif_from_frames
from config.constants import (
    OVAL_CENTER, OVAL_WIDTH, OVAL_HEIGHT, TOTAL_FRAMES, 
    MOVING_FIGURE_SIZE, MOVING_FIGURE_SPEED, OUTPUT_FILENAMES,
    WALKING_FIGURE_SIZE, WALKING_SPEED, WALKING_DIRECTION_CHANGE_CHANCE,
    WALKING_RANDOM_VARIATION, WALKING_POSE_CHANGE_SPEED, WALKING_BOUNDARY_MARGIN,
    RANDOM_WALK_STEP_SIZE, RANDOM_WALK_MOMENTUM, RANDOM_WALK_DIRECTION_NOISE,
    DEFAULT_COLOR_SCHEME, DYNAMIC_COLORS, FMRI_REALISM
)


def calculate_movement_speed(current_pos, previous_pos):
    """
    Berekent bewegingssnelheid tussen twee posities.
    
    Args:
        current_pos (tuple): Huidige positie (x, y)
        previous_pos (tuple): Vorige positie (x, y)
        
    Returns:
        float: Bewegingssnelheid in pixels per frame
    """
    if previous_pos is None:
        return 0.0
    
    dx = current_pos[0] - previous_pos[0]
    dy = current_pos[1] - previous_pos[1]
    return math.sqrt(dx * dx + dy * dy)


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


def create_moving_figure(size=None, color=None, pulse_position=0.0, color_scheme=None, 
                        activity_level=0.5, time_factor=0.0, enable_fmri_realism=True):
    """
    Creëert een bewegend figuur met enhanced fMRI styling en realisme (originele cirkel versie).
    
    Args:
        size (int): Grootte van het figuur (standaard uit constants)
        color (tuple): RGB kleur (standaard uit kleurschema)
        pulse_position (float): Positie voor pulsing effect (0.0-1.0)
        color_scheme (str): Naam van het kleurschema
        activity_level (float): Activiteitsniveau voor kleurmapping (0.0-1.0)
        time_factor (float): Tijd factor voor dynamische effecten
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
        
    Returns:
        PIL.Image: Figuur afbeelding met transparante achtergrond
    """
    if size is None:
        size = MOVING_FIGURE_SIZE
    
    # Gebruik activity-based kleur als geen specifieke kleur gegeven
    if color is None:
        color = create_activity_based_color(activity_level, color_scheme, time_factor)
    
    # Creëer pulserende kleur
    pulsing_color = create_pulsing_color(color, pulse_position, color_scheme)
    
    # Maak basis cirkel
    figure = create_colored_circle(size, pulsing_color, alpha=220)
    
    # Voeg enhanced gloed effect toe
    figure_with_glow = add_glow_effect(figure, pulsing_color, enhanced=True)
    
    # Pas fMRI realisme toe indien ingeschakeld
    if enable_fmri_realism:
        figure_with_glow = apply_fmri_realism_to_figure(
            figure_with_glow, activity_level, time_factor
        )
    
    return figure_with_glow


def create_walking_figure(pose_frame=0, size=None, color=None, pulse_position=0.0,
                         color_scheme=None, movement_speed=0.0, max_speed=10.0, 
                         time_factor=0.0, enable_fmri_realism=True):
    """
    Creëert een realistisch lopend mannetje figuur met enhanced fMRI styling en realisme.
    
    Args:
        pose_frame (int): Frame nummer voor loop animatie
        size (int): Grootte van het mannetje (standaard uit constants)
        color (tuple): RGB kleur (standaard bewegingsgebaseerd)
        pulse_position (float): Positie voor pulsing effect (0.0-1.0)
        color_scheme (str): Naam van het kleurschema
        movement_speed (float): Huidige bewegingssnelheid
        max_speed (float): Maximale bewegingssnelheid
        time_factor (float): Tijd factor voor dynamische effecten
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
        
    Returns:
        PIL.Image: Stick figure afbeelding met transparante achtergrond en gloed
    """
    if size is None:
        size = WALKING_FIGURE_SIZE
    
    # Gebruik bewegingsgebaseerde kleur als geen specifieke kleur gegeven
    if color is None:
        color = create_movement_based_color(movement_speed, max_speed, color_scheme, time_factor)
    
    # Creëer pulserende kleur
    pulsing_color = create_pulsing_color(color, pulse_position, color_scheme)
    
    # Maak stick figure
    figure = create_stick_figure(pose_frame, size, pulsing_color)
    
    # Voeg enhanced gloed effect toe
    figure_with_glow = add_glow_effect(figure, pulsing_color, enhanced=True)
    
    # Pas fMRI realisme toe indien ingeschakeld
    if enable_fmri_realism:
        # Bereken activiteitsniveau gebaseerd op bewegingssnelheid
        activity_level = min(1.0, movement_speed / max_speed) if max_speed > 0 else 0.5
        figure_with_glow = apply_fmri_realism_to_figure(
            figure_with_glow, activity_level, time_factor
        )
    
    return figure_with_glow


def apply_fmri_realism_to_figure(figure, activity_level=0.5, time_factor=0.0):
    """
    Past fMRI realisme effecten toe op een figuur.
    
    Args:
        figure (PIL.Image): Figuur om realisme aan toe te voegen
        activity_level (float): Activiteitsniveau (0.0-1.0)
        time_factor (float): Tijd factor voor temporele effecten
        
    Returns:
        PIL.Image: Figuur met fMRI realisme effecten
    """
    # Pas voxel textuur toe
    if FMRI_REALISM.get('voxel_enabled', True):
        figure = render_voxel_texture(
            figure,
            FMRI_REALISM.get('voxel_size', 2),
            FMRI_REALISM.get('voxel_opacity', 0.3),
            'subtle'  # Gebruik subtiele voxel stijl voor figuren
        )
    
    # Pas spatial smoothing toe
    if FMRI_REALISM.get('smoothing_enabled', True):
        figure = apply_spatial_smoothing_image(
            figure,
            FMRI_REALISM.get('smoothing_kernel', 3),
            FMRI_REALISM.get('smoothing_preserve_edges', True)
        )
    
    # Verbeter randen met fMRI-stijl
    if FMRI_REALISM.get('edge_enhancement', True):
        figure = enhance_edges_fmri_style(
            figure,
            FMRI_REALISM.get('edge_glow_radius', 2),
            FMRI_REALISM.get('edge_glow_intensity', 0.4),
            FMRI_REALISM.get('edge_detection_threshold', 0.2)
        )
    
    # Voeg gradiënt grenzen toe
    if FMRI_REALISM.get('gradient_boundaries', True):
        figure = create_gradient_boundaries(
            figure,
            FMRI_REALISM.get('gradient_width', 5),
            FMRI_REALISM.get('gradient_falloff', 'gaussian'),
            FMRI_REALISM.get('gradient_strength', 0.7)
        )
    
    # Voeg anatomische variatie toe
    if FMRI_REALISM.get('anatomical_variation', True):
        figure = add_anatomical_variation(
            figure,
            FMRI_REALISM.get('anatomical_asymmetry', 0.1),
            FMRI_REALISM.get('anatomical_hotspots', True),
            FMRI_REALISM.get('anatomical_gradients', True)
        )
    
    return figure


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


def generate_animation_frames(clockwise=True, route_variation=0, speed_multiplier=None, 
                            color_scheme=None, enable_fmri_realism=True):
    """
    Genereert alle frames voor de bewegend mannetje animatie (originele ovaal route).
    
    Args:
        clockwise (bool): Bewegingsrichting
        route_variation (int): Route variatie (0-2)
        speed_multiplier (float): Snelheid multiplier (standaard uit constants)
        color_scheme (str): Naam van het kleurschema
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
        
    Returns:
        list: Lijst van PIL.Image frames
    """
    if speed_multiplier is None:
        speed_multiplier = MOVING_FIGURE_SPEED
    if color_scheme is None:
        color_scheme = DEFAULT_COLOR_SCHEME
    
    # Laad achtergrond en maak masker
    background = load_background_image()
    oval_mask = create_oval_mask(background.size)
    
    frames = []
    previous_pos = None
    previous_frame_data = None
    
    for frame_num in range(TOTAL_FRAMES):
        # Bereken voortgang met snelheid multiplier
        base_progress = frame_num / TOTAL_FRAMES
        progress = (base_progress * speed_multiplier) % 1.0
        
        # Bereken positie op ovaalroute
        x, y = calculate_oval_position(progress, clockwise, route_variation)
        current_pos = (x, y)
        
        # Bereken bewegingssnelheid voor kleurmapping
        movement_speed = calculate_movement_speed(current_pos, previous_pos)
        max_speed = WALKING_SPEED * speed_multiplier
        
        # Bereken activiteitsniveau gebaseerd op beweging en positie
        activity_level = min(1.0, movement_speed / max_speed) if max_speed > 0 else 0.5
        
        # Tijd factor voor dynamische effecten
        time_factor = frame_num / TOTAL_FRAMES
        
        # Creëer bewegend figuur met enhanced kleuren en realisme
        pulse_position = (frame_num / TOTAL_FRAMES) * 4  # 4 pulses per cyclus
        figure = create_moving_figure(
            pulse_position=pulse_position,
            color_scheme=color_scheme,
            activity_level=activity_level,
            time_factor=time_factor,
            enable_fmri_realism=enable_fmri_realism
        )
        
        # Maak frame met achtergrond
        frame = background.copy()
        
        # Bereken figuur positie (centreer figuur op berekende positie)
        figure_x = x - figure.width // 2
        figure_y = y - figure.height // 2
        
        # Plak figuur op frame
        frame.paste(figure, (figure_x, figure_y), figure)
        
        # Pas ovaal masker toe (alleen binnen hersenregio)
        masked_overlay = apply_oval_mask(frame, oval_mask)
        
        # Pas fMRI realisme toe op het hele frame indien ingeschakeld
        if enable_fmri_realism:
            enhanced_overlay, frame_data = enhance_fmri_realism(
                masked_overlay, 
                frame_number=frame_num,
                previous_frame=previous_frame_data
            )
            previous_frame_data = frame_data
            masked_overlay = enhanced_overlay
        
        # Combineer met originele achtergrond
        final_frame = background.copy()
        final_frame = composite_images(final_frame, masked_overlay)
        
        frames.append(final_frame)
        previous_pos = current_pos
    
    return frames


def generate_walking_animation_frames(start_position=None, color_scheme=None, enable_fmri_realism=True):
    """
    Genereert alle frames voor de realistische lopende mannetje animatie.
    
    Args:
        start_position (tuple): Start positie (willekeurig als None)
        color_scheme (str): Naam van het kleurschema
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
        
    Returns:
        list: Lijst van PIL.Image frames
    """
    if color_scheme is None:
        color_scheme = DEFAULT_COLOR_SCHEME
    
    # Laad achtergrond en maak masker
    background = load_background_image()
    oval_mask = create_oval_mask(background.size)
    
    # Genereer random walk pad
    positions = generate_random_walk_path(TOTAL_FRAMES, start_position)
    
    frames = []
    previous_pos = None
    previous_frame_data = None
    
    for frame_num in range(TOTAL_FRAMES):
        x, y = positions[frame_num]
        current_pos = (x, y)
        
        # Bereken bewegingssnelheid voor kleurmapping
        movement_speed = calculate_movement_speed(current_pos, previous_pos)
        max_speed = WALKING_SPEED * 2.0  # Verhoogde max voor betere kleurvariatie
        
        # Bereken pose frame voor loop animatie
        pose_frame = (frame_num // WALKING_POSE_CHANGE_SPEED) % 6
        
        # Tijd factor voor dynamische effecten
        time_factor = frame_num / TOTAL_FRAMES
        
        # Creëer lopend mannetje met enhanced kleuren en realisme
        pulse_position = (frame_num / TOTAL_FRAMES) * 3  # 3 pulses per cyclus
        figure = create_walking_figure(
            pose_frame=pose_frame,
            pulse_position=pulse_position,
            color_scheme=color_scheme,
            movement_speed=movement_speed,
            max_speed=max_speed,
            time_factor=time_factor,
            enable_fmri_realism=enable_fmri_realism
        )
        
        # Maak frame met achtergrond
        frame = background.copy()
        
        # Bereken figuur positie (centreer figuur op berekende positie)
        figure_x = x - figure.width // 2
        figure_y = y - figure.height // 2
        
        # Plak figuur op frame
        frame.paste(figure, (figure_x, figure_y), figure)
        
        # Pas ovaal masker toe (alleen binnen hersenregio)
        masked_overlay = apply_oval_mask(frame, oval_mask)
        
        # Pas fMRI realisme toe op het hele frame indien ingeschakeld
        if enable_fmri_realism:
            enhanced_overlay, frame_data = enhance_fmri_realism(
                masked_overlay, 
                frame_number=frame_num,
                previous_frame=previous_frame_data
            )
            previous_frame_data = frame_data
            masked_overlay = enhanced_overlay
        
        # Combineer met originele achtergrond
        final_frame = background.copy()
        final_frame = composite_images(final_frame, masked_overlay)
        
        frames.append(final_frame)
        previous_pos = current_pos
    
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
    output_filename=None,
    color_scheme=None,
    enable_fmri_realism=True
):
    """
    Hoofdfunctie voor het genereren van bewegend mannetje animatie (originele ovaal route).
    
    Args:
        clockwise (bool): Bewegingsrichting (True=rechtsom, False=linksom)
        route_variation (int): Route variatie (0=standaard, 1=elliptisch, 2=spiraal)
        speed_multiplier (float): Snelheid multiplier (standaard uit constants)
        smooth_interpolation (bool): Extra frames voor smoothere animatie
        output_filename (str): Naam van output bestand (standaard uit constants)
        color_scheme (str): Naam van het kleurschema
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
        
    Returns:
        str: Pad naar gegenereerde GIF
        
    Raises:
        Exception: Als er een fout optreedt tijdens generatie
    """
    try:
        if color_scheme is None:
            color_scheme = DEFAULT_COLOR_SCHEME
            
        print("Genereren van bewegend mannetje animatie (ovaal route)...")
        print(f"- Richting: {'Rechtsom' if clockwise else 'Linksom'}")
        print(f"- Route variatie: {route_variation}")
        print(f"- Snelheid multiplier: {speed_multiplier or MOVING_FIGURE_SPEED}")
        print(f"- Kleurschema: {color_scheme}")
        print(f"- Smooth interpolatie: {smooth_interpolation}")
        print(f"- fMRI realisme: {enable_fmri_realism}")
        
        # Genereer basis frames
        frames = generate_animation_frames(
            clockwise, route_variation, speed_multiplier, 
            color_scheme, enable_fmri_realism
        )
        print(f"- {len(frames)} basis frames gegenereerd")
        
        # Voeg smooth interpolatie toe indien gewenst
        if smooth_interpolation:
            frames = create_smooth_interpolated_frames(frames, interpolation_factor=2)
            print(f"- {len(frames)} frames na interpolatie")
        
        # Bepaal output bestandsnaam
        if output_filename is None:
            suffix = "_fmri_realism" if enable_fmri_realism else ""
            output_filename = OUTPUT_FILENAMES['moving_figure'].replace('.gif', f'{suffix}.gif')
        
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
    output_filename=None,
    color_scheme=None,
    enable_fmri_realism=True
):
    """
    Hoofdfunctie voor het genereren van realistisch lopend mannetje animatie.
    
    Args:
        start_position (tuple): Start positie (willekeurig als None)
        smooth_interpolation (bool): Extra frames voor smoothere animatie
        output_filename (str): Naam van output bestand (standaard uit constants)
        color_scheme (str): Naam van het kleurschema
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
        
    Returns:
        str: Pad naar gegenereerde GIF
        
    Raises:
        Exception: Als er een fout optreedt tijdens generatie
    """
    try:
        if color_scheme is None:
            color_scheme = DEFAULT_COLOR_SCHEME
            
        print("Genereren van realistisch lopend mannetje animatie...")
        print(f"- Start positie: {start_position or 'Willekeurig'}")
        print(f"- Mannetje grootte: {WALKING_FIGURE_SIZE}")
        print(f"- Bewegingssnelheid: {WALKING_SPEED} pixels/frame")
        print(f"- Kleurschema: {color_scheme}")
        print(f"- Smooth interpolatie: {smooth_interpolation}")
        print(f"- fMRI realisme: {enable_fmri_realism}")
        
        # Genereer basis frames
        frames = generate_walking_animation_frames(start_position, color_scheme, enable_fmri_realism)
        print(f"- {len(frames)} basis frames gegenereerd")
        
        # Voeg smooth interpolatie toe indien gewenst
        if smooth_interpolation:
            frames = create_smooth_interpolated_frames(frames, interpolation_factor=2)
            print(f"- {len(frames)} frames na interpolatie")
        
        # Bepaal output bestandsnaam
        if output_filename is None:
            suffix = "_fmri_realism" if enable_fmri_realism else ""
            output_filename = OUTPUT_FILENAMES['walking_figure'].replace('.gif', f'{suffix}.gif')
        
        # Creëer GIF
        output_path = create_gif_from_frames(frames, output_filename)
        print(f"✅ Realistisch lopend mannetje animatie succesvol gegenereerd: {output_path}")
        
        return output_path
        
    except Exception as e:
        error_msg = f"Fout bij genereren lopend mannetje animatie: {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)


def create_fmri_realism_demo():
    """
    Creëert demo animaties die het verschil tonen tussen normale en fMRI-realisme versies.
    
    Returns:
        list: Lijst van paden naar gegenereerde GIFs
    """
    demo_files = []
    
    try:
        print("\n=== fMRI Realisme Demo ===")
        
        # Test verschillende kleurschema's met en zonder realisme
        color_schemes = ['hot', 'viridis']
        
        for scheme in color_schemes:
            print(f"\n--- Kleurschema: {scheme} ---")
            
            # Normale versie
            normal_path = genereer_lopend_mannetje_animatie(
                color_scheme=scheme,
                enable_fmri_realism=False,
                output_filename=f"demo_{scheme}_normal.gif"
            )
            demo_files.append(normal_path)
            
            # fMRI realisme versie
            realism_path = genereer_lopend_mannetje_animatie(
                color_scheme=scheme,
                enable_fmri_realism=True,
                output_filename=f"demo_{scheme}_fmri_realism.gif"
            )
            demo_files.append(realism_path)
        
        print(f"\n✅ {len(demo_files)} demo bestanden gegenereerd")
        return demo_files
        
    except Exception as e:
        print(f"❌ Fout bij genereren fMRI realisme demo: {str(e)}")
        return demo_files


def create_demo_variations():
    """
    Creëert demo variaties van de bewegend mannetje animaties met verschillende kleurschema's.
    
    Returns:
        list: Lijst van paden naar gegenereerde GIFs
    """
    variations = []
    
    try:
        # Test verschillende kleurschema's
        color_schemes = ['hot', 'cool', 'jet', 'viridis']
        
        print("\n=== Genereren enhanced fMRI kleurschema variaties ===")
        
        for scheme in color_schemes:
            print(f"\n--- Kleurschema: {scheme} ---")
            
            # Originele ovaal route met kleurschema en fMRI realisme
            path1 = genereer_bewegend_mannetje_animatie(
                clockwise=True,
                route_variation=0,
                color_scheme=scheme,
                enable_fmri_realism=True,
                output_filename=f"bewegend_mannetje_{scheme}_ovaal_realism.gif"
            )
            variations.append(path1)
            
            # Realistisch lopend mannetje met kleurschema en fMRI realisme
            path2 = genereer_lopend_mannetje_animatie(
                color_scheme=scheme,
                enable_fmri_realism=True,
                output_filename=f"bewegend_mannetje_{scheme}_lopend_realism.gif"
            )
            variations.append(path2)
        
        print(f"\n✅ {len(variations)} enhanced kleurschema variaties succesvol gegenereerd")
        return variations
        
    except Exception as e:
        print(f"❌ Fout bij genereren demo variaties: {str(e)}")
        return variations


if __name__ == "__main__":
    # Test de nieuwe fMRI realisme functionaliteit
    try:
        print("=== Test Enhanced fMRI Realisme ===")
        
        # Test fMRI realisme demo
        print("\n--- fMRI Realisme Demo ---")
        demo_paths = create_fmri_realism_demo()
        print(f"Demo bestanden: {demo_paths}")
        
        # Test met verschillende kleurschema's en realisme
        schemes_to_test = ['hot', 'viridis']
        
        for scheme in schemes_to_test:
            print(f"\n--- Test {scheme} kleurschema met fMRI realisme ---")
            output_path = genereer_lopend_mannetje_animatie(
                color_scheme=scheme,
                enable_fmri_realism=True,
                output_filename=f"test_{scheme}_walking_realism.gif"
            )
            print(f"Test animatie gegenereerd: {output_path}")
        
        # Genereer ook demo variaties
        print("\n=== Genereren Demo Variaties ===")
        demo_variations = create_demo_variations()
        print(f"Demo variaties: {demo_variations}")
        
    except Exception as e:
        print(f"Test gefaald: {str(e)}")