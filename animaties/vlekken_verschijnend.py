"""
Vlekken Verschijnend Animatie voor BewegendeAnimaties.

Deze module implementeert een animatie waarbij vlekken zich geleidelijk uitspreiden
tot het hele ovaal gevuld is. De animatie simuleert toenemende hersenactiviteit
en gebruikt enhanced fMRI-stijl kleurschalen met dynamische kleureffecten en fMRI-realisme.
"""

import math
import random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from utils.image_utils import (
    load_background_image, create_oval_mask, apply_oval_mask,
    get_oval_bounds, point_in_oval, generate_random_position_in_oval,
    render_voxel_texture, apply_spatial_smoothing_image,
    enhance_edges_fmri_style, create_gradient_boundaries,
    add_anatomical_variation, render_activation_clusters
)
from utils.color_utils import (
    get_fmri_color, create_fmri_gradient, add_glow_effect,
    create_pulsing_color, map_value_to_color, create_activity_based_color,
    apply_temporal_color_variation, create_intensity_mapped_color,
    enhance_fmri_realism, simulate_zscore_mapping, detect_activation_clusters,
    apply_statistical_thresholding, apply_temporal_correlation
)
from utils.gif_utils import create_gif_from_frames
from config.constants import (
    SPOT_MIN_SIZE, SPOT_MAX_SIZE, SPOT_COUNT, ANIMATION_DURATION,
    FRAMES_PER_SECOND, TOTAL_FRAMES, OUTPUT_FILENAMES,
    OVAL_CENTER, OVAL_WIDTH, OVAL_HEIGHT, DEFAULT_COLOR_SCHEME,
    SPOT_COLOR_VARIATION, SPOT_INTENSITY_MAPPING, DYNAMIC_COLORS,
    FMRI_REALISM
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


def calculate_spot_growth(frame_number, total_frames, spot_index, total_spots):
    """
    Berekent de grootte van een vlek voor een specifiek frame.
    
    Args:
        frame_number (int): Huidige frame nummer (0-based)
        total_frames (int): Totaal aantal frames
        spot_index (int): Index van de vlek
        total_spots (int): Totaal aantal vlekken
        
    Returns:
        float: Grootte van de vlek (0.0 = niet zichtbaar, 1.0 = maximale grootte)
    """
    # Bereken voortgang van animatie (0.0 - 1.0)
    animation_progress = frame_number / (total_frames - 1) if total_frames > 1 else 0
    
    # Vlekken verschijnen geleidelijk - elke vlek start op een ander moment
    spot_start_time = (spot_index / total_spots) * 0.6  # Eerste 60% van animatie
    spot_growth_duration = 0.8  # Elke vlek groeit over 80% van de tijd
    
    # Bereken lokale voortgang voor deze vlek
    if animation_progress < spot_start_time:
        return 0.0  # Vlek nog niet gestart
    
    local_progress = (animation_progress - spot_start_time) / spot_growth_duration
    local_progress = max(0.0, min(1.0, local_progress))
    
    # Gebruik easing functie voor natuurlijke groei
    # Sigmoid curve voor smooth start en eind
    if local_progress <= 0:
        return 0.0
    elif local_progress >= 1:
        return 1.0
    else:
        # Sigmoid easing
        sigmoid = 1 / (1 + math.exp(-10 * (local_progress - 0.5)))
        return sigmoid


def calculate_spot_intensity(growth_factor, spot_index, total_spots, time_factor):
    """
    Berekent de kleurintensiteit van een vlek gebaseerd op grootte en tijd.
    
    Args:
        growth_factor (float): Grootte factor van de vlek (0.0-1.0)
        spot_index (int): Index van de vlek
        total_spots (int): Totaal aantal vlekken
        time_factor (float): Tijd factor voor dynamische effecten
        
    Returns:
        float: Intensiteitswaarde voor kleurmapping (0.0-1.0)
    """
    if not SPOT_INTENSITY_MAPPING:
        return 0.5  # Standaard intensiteit
    
    # Basis intensiteit gebaseerd op grootte
    base_intensity = growth_factor
    
    # Voeg variatie toe gebaseerd op vlek positie
    spot_variation = (spot_index / total_spots) * 0.3
    
    # Voeg tijdelijke variatie toe
    if DYNAMIC_COLORS.get('time_based', True):
        time_variation = math.sin(time_factor * 2 * math.pi + spot_index) * 0.2
        base_intensity += time_variation
    
    # Combineer alle factoren
    final_intensity = base_intensity + spot_variation
    
    # Clamp tussen 0 en 1
    return max(0.0, min(1.0, final_intensity))


def create_organic_spot(size, color=None, irregularity=0.3, enable_fmri_realism=True):
    """
    Creëert een natuurlijk uitziende vlek vorm met optionele fMRI realisme.
    
    Args:
        size (int): Basis grootte van de vlek
        color (tuple): RGB kleur (standaard fMRI primary)
        irregularity (float): Mate van onregelmatigheid (0.0-1.0)
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
        
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
    
    # Pas fMRI realisme toe indien ingeschakeld
    if enable_fmri_realism:
        spot_img = apply_fmri_realism_to_spot(spot_img, size)
    
    return spot_img


def apply_fmri_realism_to_spot(spot_image, spot_size):
    """
    Past fMRI realisme effecten toe op een vlek.
    
    Args:
        spot_image (PIL.Image): Vlek afbeelding om realisme aan toe te voegen
        spot_size (int): Grootte van de vlek voor schaling van effecten
        
    Returns:
        PIL.Image: Vlek afbeelding met fMRI realisme effecten
    """
    # Aangepaste instellingen voor vlekken (gebaseerd op grootte)
    spot_realism_settings = FMRI_REALISM.copy()
    
    # Schaal effecten gebaseerd op vlek grootte
    size_factor = min(1.0, spot_size / SPOT_MAX_SIZE)
    
    # Pas voxel textuur toe (subtiel voor kleine vlekken)
    if spot_realism_settings.get('voxel_enabled', True):
        voxel_opacity = spot_realism_settings.get('voxel_opacity', 0.3) * size_factor
        spot_image = render_voxel_texture(
            spot_image,
            max(1, spot_realism_settings.get('voxel_size', 2)),
            voxel_opacity,
            'subtle'
        )
    
    # Pas spatial smoothing toe
    if spot_realism_settings.get('smoothing_enabled', True):
        spot_image = apply_spatial_smoothing_image(
            spot_image,
            max(1, spot_realism_settings.get('smoothing_kernel', 3) - 1),  # Kleinere kernel voor vlekken
            spot_realism_settings.get('smoothing_preserve_edges', True)
        )
    
    # Verbeter randen met fMRI-stijl
    if spot_realism_settings.get('edge_enhancement', True):
        spot_image = enhance_edges_fmri_style(
            spot_image,
            max(1, spot_realism_settings.get('edge_glow_radius', 2)),
            spot_realism_settings.get('edge_glow_intensity', 0.4) * size_factor,
            spot_realism_settings.get('edge_detection_threshold', 0.2)
        )
    
    # Voeg gradiënt grenzen toe
    if spot_realism_settings.get('gradient_boundaries', True):
        spot_image = create_gradient_boundaries(
            spot_image,
            max(2, spot_realism_settings.get('gradient_width', 5)),
            spot_realism_settings.get('gradient_falloff', 'gaussian'),
            spot_realism_settings.get('gradient_strength', 0.7) * size_factor
        )
    
    return spot_image


def get_spot_color(spot_index, total_spots, growth_factor, time_factor, color_scheme=None):
    """
    Bepaalt de kleur van een vlek gebaseerd op verschillende factoren.
    
    Args:
        spot_index (int): Index van de vlek
        total_spots (int): Totaal aantal vlekken
        growth_factor (float): Grootte factor van de vlek (0.0-1.0)
        time_factor (float): Tijd factor voor dynamische effecten
        color_scheme (str): Naam van het kleurschema
        
    Returns:
        tuple: RGB kleur tuple
    """
    if color_scheme is None:
        color_scheme = DEFAULT_COLOR_SCHEME
    
    if SPOT_COLOR_VARIATION:
        # Bereken intensiteit gebaseerd op grootte en tijd
        intensity = calculate_spot_intensity(growth_factor, spot_index, total_spots, time_factor)
        
        # Simuleer z-score mapping voor wetenschappelijke authenticiteit
        if FMRI_REALISM.get('zscore_mapping', True):
            zscore, is_significant, zscore_intensity = simulate_zscore_mapping(
                intensity,
                FMRI_REALISM.get('zscore_range', (-3.0, 6.0)),
                FMRI_REALISM.get('zscore_threshold', 2.3)
            )
            
            # Gebruik z-score intensiteit voor kleurmapping
            if is_significant:
                intensity = zscore_intensity
            else:
                intensity *= 0.5  # Verminder intensiteit voor niet-significante activatie
        
        # Map intensiteit naar kleur uit kleurschema
        base_color = create_intensity_mapped_color(intensity, color_scheme)
        
        # Voeg tijdelijke variatie toe
        if DYNAMIC_COLORS.get('time_based', True):
            base_color = apply_temporal_color_variation(base_color, time_factor)
        
        return base_color
    else:
        # Gebruik standaard fMRI gradient
        color_gradient = create_fmri_gradient(steps=max(3, total_spots // 2), color_scheme=color_scheme)
        color_index = spot_index % len(color_gradient)
        return color_gradient[color_index]


def render_spots_frame(frame_number, spot_positions, background_size, color_scheme=None, 
                      enable_fmri_realism=True, previous_frame_data=None):
    """
    Rendert één frame van de vlekken animatie met optionele fMRI realisme.
    
    Args:
        frame_number (int): Frame nummer om te renderen
        spot_positions (list): Lijst van vlek posities
        background_size (tuple): (width, height) van achtergrond
        color_scheme (str): Naam van het kleurschema
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
        previous_frame_data (numpy.ndarray): Vorige frame data voor temporele correlatie
        
    Returns:
        tuple: (PIL.Image overlay, numpy.ndarray frame_data)
    """
    if color_scheme is None:
        color_scheme = DEFAULT_COLOR_SCHEME
    
    # Maak transparante overlay voor vlekken
    overlay = Image.new('RGBA', background_size, (0, 0, 0, 0))
    
    total_spots = len(spot_positions)
    time_factor = frame_number / TOTAL_FRAMES
    
    # Houd activatie data bij voor fMRI realisme
    activation_data = np.zeros(background_size[::-1])  # (height, width)
    
    for spot_index, position in enumerate(spot_positions):
        # Bereken grootte voor deze vlek in dit frame
        growth_factor = calculate_spot_growth(
            frame_number, TOTAL_FRAMES, spot_index, total_spots
        )
        
        if growth_factor <= 0:
            continue  # Vlek nog niet zichtbaar
        
        # Bereken werkelijke grootte
        min_size = SPOT_MIN_SIZE
        max_size = SPOT_MAX_SIZE
        current_size = int(min_size + (max_size - min_size) * growth_factor)
        
        # Bepaal kleur gebaseerd op verschillende factoren
        base_color = get_spot_color(
            spot_index, total_spots, growth_factor, time_factor, color_scheme
        )
        
        # Voeg pulsing effect toe voor meer dynamiek
        if DYNAMIC_COLORS.get('pulsing_enabled', True):
            pulse_position = (time_factor + spot_index * 0.1) % 1.0
            spot_color = create_pulsing_color(base_color, pulse_position, color_scheme)
        else:
            spot_color = base_color
        
        # Creëer organische vlek
        spot = create_organic_spot(current_size, spot_color, enable_fmri_realism=enable_fmri_realism)
        
        # Bereken positie om vlek te centreren
        spot_x = position[0] - spot.size[0] // 2
        spot_y = position[1] - spot.size[1] // 2
        
        # Plak vlek op overlay
        overlay.paste(spot, (spot_x, spot_y), spot)
        
        # Update activatie data voor fMRI realisme
        if enable_fmri_realism:
            # Voeg activatie toe aan data array
            spot_radius = current_size // 2
            y_min = max(0, position[1] - spot_radius)
            y_max = min(background_size[1], position[1] + spot_radius)
            x_min = max(0, position[0] - spot_radius)
            x_max = min(background_size[0], position[0] + spot_radius)
            
            # Voeg circulaire activatie toe
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    distance = math.sqrt((x - position[0])**2 + (y - position[1])**2)
                    if distance <= spot_radius:
                        # Gaussian falloff
                        intensity = growth_factor * math.exp(-(distance**2) / (2 * (spot_radius/3)**2))
                        activation_data[y, x] = max(activation_data[y, x], intensity)
    
    # Pas globale fMRI realisme effecten toe
    if enable_fmri_realism and not all(pixel[3] == 0 for pixel in overlay.getdata()):
        # Pas statistical thresholding toe op activatie data
        if FMRI_REALISM.get('threshold_enabled', True):
            activation_data = apply_statistical_thresholding(
                activation_data,
                FMRI_REALISM.get('threshold_level', 0.3),
                FMRI_REALISM.get('threshold_type', 'soft'),
                FMRI_REALISM.get('threshold_fade', 0.1)
            )
        
        # Detecteer activatie clusters
        if FMRI_REALISM.get('cluster_enabled', True):
            clusters = detect_activation_clusters(
                activation_data,
                FMRI_REALISM.get('cluster_min_size', 5),
                FMRI_REALISM.get('cluster_connectivity', 8)
            )
            
            # Render clusters (optioneel - kan overlay vervangen)
            if np.any(clusters > 0):
                # Gebruik cluster informatie om overlay te moduleren
                cluster_mask = clusters > 0
                overlay_array = np.array(overlay)
                if overlay_array.shape[:2] == cluster_mask.shape:
                    # Moduleer alpha channel gebaseerd op clusters
                    overlay_array[:, :, 3] = overlay_array[:, :, 3] * cluster_mask
                    overlay = Image.fromarray(overlay_array)
        
        # Pas temporele correlatie toe
        if previous_frame_data is not None:
            activation_data = apply_temporal_correlation(
                activation_data,
                previous_frame_data,
                FMRI_REALISM.get('temporal_correlation', 0.8)
            )
        
        # Voeg anatomische variatie toe
        if FMRI_REALISM.get('anatomical_variation', True):
            overlay = add_anatomical_variation(
                overlay,
                FMRI_REALISM.get('anatomical_asymmetry', 0.1),
                FMRI_REALISM.get('anatomical_hotspots', True),
                FMRI_REALISM.get('anatomical_gradients', True)
            )
    
    # Voeg enhanced gloed effect toe aan hele overlay
    if not all(pixel[3] == 0 for pixel in overlay.getdata()):  # Als er vlekken zijn
        overlay = add_glow_effect(overlay, enhanced=True, intensity=0.7)
    
    return overlay, activation_data


def create_spots_appearing_animation(output_filename=None, seed=None, color_scheme=None, enable_fmri_realism=True):
    """
    Creëert een animatie van vlekken die uitspreiden tot het ovaal gevuld is.
    
    Args:
        output_filename (str): Naam van output GIF (standaard uit constants)
        seed (int): Random seed voor reproduceerbare resultaten
        color_scheme (str): Naam van het kleurschema
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
        
    Returns:
        str: Pad naar gegenereerde GIF
    """
    if output_filename is None:
        suffix = "_fmri_realism" if enable_fmri_realism else ""
        output_filename = OUTPUT_FILENAMES['spots_appearing'].replace('.gif', f'{suffix}.gif')
    if color_scheme is None:
        color_scheme = DEFAULT_COLOR_SCHEME
    
    print(f"Genereren vlekken verschijnend animatie...")
    print(f"- {TOTAL_FRAMES} frames @ {FRAMES_PER_SECOND} FPS")
    print(f"- {SPOT_COUNT} vlekken")
    print(f"- Grootte: {SPOT_MIN_SIZE}-{SPOT_MAX_SIZE} pixels")
    print(f"- Kleurschema: {color_scheme}")
    print(f"- Kleurvariatie: {SPOT_COLOR_VARIATION}")
    print(f"- Intensiteit mapping: {SPOT_INTENSITY_MAPPING}")
    print(f"- fMRI realisme: {enable_fmri_realism}")
    
    # Laad achtergrond en maak masker
    background = load_background_image()
    oval_mask = create_oval_mask(background.size)
    
    # Genereer vlek posities
    spot_positions = generate_spot_positions(seed=seed)
    
    frames = []
    previous_frame_data = None
    
    # Genereer alle frames
    for frame_num in range(TOTAL_FRAMES):
        print(f"Renderen frame {frame_num + 1}/{TOTAL_FRAMES}...", end='\r')
        
        # Render vlekken voor dit frame
        spots_overlay, frame_data = render_spots_frame(
            frame_num, spot_positions, background.size, 
            color_scheme, enable_fmri_realism, previous_frame_data
        )
        previous_frame_data = frame_data
        
        # Pas ovaal masker toe op vlekken
        masked_spots = apply_oval_mask(spots_overlay, oval_mask)
        
        # Pas globale fMRI realisme toe op het hele frame indien ingeschakeld
        if enable_fmri_realism:
            enhanced_spots, _ = enhance_fmri_realism(
                masked_spots,
                frame_number=frame_num,
                previous_frame=previous_frame_data
            )
            masked_spots = enhanced_spots
        
        # Combineer met achtergrond
        frame = background.copy()
        frame.paste(masked_spots, (0, 0), masked_spots)
        
        frames.append(frame)
    
    print(f"\nOpslaan als {output_filename}...")
    
    # Maak GIF
    gif_path = create_gif_from_frames(frames, output_filename)
    
    print(f"✅ Vlekken verschijnend animatie voltooid: {gif_path}")
    return gif_path


def create_demo_animation(enable_fmri_realism=True):
    """
    Creëert een demo versie van de vlekken animatie voor testen.
    
    Args:
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
    
    Returns:
        str: Pad naar demo GIF
    """
    suffix = "_realism" if enable_fmri_realism else ""
    return create_spots_appearing_animation(
        output_filename=f"demo_vlekken_verschijnend{suffix}.gif",
        seed=42,  # Vaste seed voor consistente demo
        enable_fmri_realism=enable_fmri_realism
    )


def create_fmri_realism_demo():
    """
    Creëert demo animaties die het verschil tonen tussen normale en fMRI-realisme versies.
    
    Returns:
        list: Lijst van paden naar gegenereerde GIFs
    """
    demo_files = []
    
    try:
        print("\n=== fMRI Realisme Demo voor Vlekken ===")
        
        # Test verschillende kleurschema's met en zonder realisme
        color_schemes = ['hot', 'viridis']
        
        for scheme in color_schemes:
            print(f"\n--- Kleurschema: {scheme} ---")
            
            # Normale versie
            normal_path = create_spots_appearing_animation(
                output_filename=f"vlekken_{scheme}_normal.gif",
                seed=42,
                color_scheme=scheme,
                enable_fmri_realism=False
            )
            demo_files.append(normal_path)
            
            # fMRI realisme versie
            realism_path = create_spots_appearing_animation(
                output_filename=f"vlekken_{scheme}_fmri_realism.gif",
                seed=42,
                color_scheme=scheme,
                enable_fmri_realism=True
            )
            demo_files.append(realism_path)
        
        print(f"\n✅ {len(demo_files)} vlekken demo bestanden gegenereerd")
        return demo_files
        
    except Exception as e:
        print(f"❌ Fout bij genereren fMRI realisme demo: {str(e)}")
        return demo_files


def create_color_scheme_demos(enable_fmri_realism=True):
    """
    Creëert demo animaties voor alle beschikbare kleurschema's.
    
    Args:
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
    
    Returns:
        list: Lijst van paden naar gegenereerde GIFs
    """
    schemes = ['hot', 'cool', 'jet', 'viridis']
    demo_paths = []
    
    print("\n=== Genereren vlekken verschijnend kleurschema demo's ===")
    
    for scheme in schemes:
        print(f"\n--- Kleurschema: {scheme} ---")
        suffix = "_realism" if enable_fmri_realism else ""
        output_filename = f"vlekken_verschijnend_{scheme}{suffix}.gif"
        
        path = create_spots_appearing_animation(
            output_filename=output_filename,
            seed=42,  # Consistente seed voor vergelijking
            color_scheme=scheme,
            enable_fmri_realism=enable_fmri_realism
        )
        demo_paths.append(path)
    
    print(f"\n✅ {len(demo_paths)} kleurschema demo's voltooid")
    return demo_paths


if __name__ == "__main__":
    # Test de enhanced fMRI realisme functionaliteit
    try:
        print("=== Test Enhanced fMRI Realisme voor Vlekken Verschijnend ===")
        
        # Test fMRI realisme demo
        print("\n--- fMRI Realisme Demo ---")
        demo_paths = create_fmri_realism_demo()
        print(f"Demo bestanden: {demo_paths}")
        
        # Test standaard animatie met realisme
        demo_path = create_demo_animation(enable_fmri_realism=True)
        print(f"Demo animatie met realisme: {demo_path}")
        
        # Test verschillende kleurschema's met realisme
        scheme_demos = create_color_scheme_demos(enable_fmri_realism=True)
        print(f"Kleurschema demo's met realisme: {scheme_demos}")
        
        print("Test voltooid!")
        
    except Exception as e:
        print(f"Test gefaald: {str(e)}")