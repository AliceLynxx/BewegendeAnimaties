"""
Tekst verschijnend animatie module voor BewegendeAnimaties.

Deze module implementeert een animatie waarbij tekst letter voor letter zichtbaar wordt
binnen het ovaal, met enhanced fMRI-stijl kleurschalen, dynamische kleureffecten en fMRI-realisme.
"""

import math
import os
from PIL import Image, ImageDraw, ImageFont
from utils.image_utils import (
    load_background_image, create_oval_mask, apply_oval_mask, 
    composite_images, get_oval_bounds, point_in_oval,
    render_voxel_texture, apply_spatial_smoothing_image,
    enhance_edges_fmri_style, create_gradient_boundaries,
    add_anatomical_variation
)
from utils.color_utils import (
    get_fmri_color, add_glow_effect, create_pulsing_color,
    map_value_to_color, create_gradient_animation_color,
    apply_temporal_color_variation, get_color_scheme,
    enhance_fmri_realism, simulate_zscore_mapping,
    apply_temporal_correlation
)
from utils.gif_utils import create_gif_from_frames
from config.constants import (
    TEXT_CONTENT, TEXT_FONT_SIZE, TEXT_COLOR, TEXT_POSITION,
    TEXT_ANIMATION_SPEED, TEXT_GLOW_RADIUS, TEXT_GLOW_INTENSITY,
    TEXT_PULSE_SPEED, TEXT_FONT_PATH, OVAL_CENTER, OVAL_WIDTH, OVAL_HEIGHT,
    TOTAL_FRAMES, FRAMES_PER_SECOND, OUTPUT_FILENAMES, DEFAULT_COLOR_SCHEME,
    TEXT_COLOR_SCHEME_BASED, DYNAMIC_COLORS, FMRI_REALISM
)


def load_font(size=None, font_path=None):
    """
    Laadt een font voor tekst rendering.
    
    Args:
        size (int): Font grootte (standaard uit constants)
        font_path (str): Pad naar font bestand (standaard uit constants)
        
    Returns:
        PIL.ImageFont: Font object
    """
    if size is None:
        size = TEXT_FONT_SIZE
    if font_path is None:
        font_path = TEXT_FONT_PATH
    
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)
        else:
            # Probeer standaard fonts
            try:
                return ImageFont.truetype("arial.ttf", size)
            except OSError:
                try:
                    return ImageFont.truetype("DejaVuSans.ttf", size)
                except OSError:
                    # Fallback naar default font
                    return ImageFont.load_default()
    except Exception as e:
        print(f"Fout bij laden font: {e}")
        return ImageFont.load_default()


def calculate_text_position(text, font, position_type="center"):
    """
    Berekent de optimale tekstpositie binnen het ovaal.
    
    Args:
        text (str): Tekst om te positioneren
        font (PIL.ImageFont): Font object
        position_type (str): Type positionering ("center", "left", "right", "top", "bottom")
        
    Returns:
        tuple: (x, y) positie voor tekst
    """
    # Bereken tekst afmetingen
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Ovaal grenzen
    oval_left = OVAL_CENTER[0] - OVAL_WIDTH // 2
    oval_right = OVAL_CENTER[0] + OVAL_WIDTH // 2
    oval_top = OVAL_CENTER[1] - OVAL_HEIGHT // 2
    oval_bottom = OVAL_CENTER[1] + OVAL_HEIGHT // 2
    
    # Bereken positie gebaseerd op type
    if position_type == "center":
        x = OVAL_CENTER[0] - text_width // 2
        y = OVAL_CENTER[1] - text_height // 2
    elif position_type == "left":
        x = oval_left + 20  # Kleine marge
        y = OVAL_CENTER[1] - text_height // 2
    elif position_type == "right":
        x = oval_right - text_width - 20  # Kleine marge
        y = OVAL_CENTER[1] - text_height // 2
    elif position_type == "top":
        x = OVAL_CENTER[0] - text_width // 2
        y = oval_top + 20  # Kleine marge
    elif position_type == "bottom":
        x = OVAL_CENTER[0] - text_width // 2
        y = oval_bottom - text_height - 20  # Kleine marge
    else:
        # Fallback naar center
        x = OVAL_CENTER[0] - text_width // 2
        y = OVAL_CENTER[1] - text_height // 2
    
    return (int(x), int(y))


def ensure_text_within_oval(text, font, position):
    """
    Zorgt ervoor dat tekst binnen ovaal grenzen blijft.
    
    Args:
        text (str): Tekst om te controleren
        font (PIL.ImageFont): Font object
        position (tuple): (x, y) positie van tekst
        
    Returns:
        tuple: Aangepaste (x, y) positie binnen ovaal
    """
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x, y = position
    
    # Controleer alle hoeken van tekst
    corners = [
        (x, y),
        (x + text_width, y),
        (x, y + text_height),
        (x + text_width, y + text_height)
    ]
    
    # Als alle hoeken binnen ovaal liggen, return originele positie
    if all(point_in_oval(corner[0], corner[1]) for corner in corners):
        return position
    
    # Anders, centreer tekst
    return calculate_text_position(text, font, "center")


def get_text_color(char_index, total_chars, frame_number, color_scheme=None, base_color=None):
    """
    Bepaalt de kleur van een tekstkarakter gebaseerd op verschillende factoren.
    
    Args:
        char_index (int): Index van het karakter
        total_chars (int): Totaal aantal karakters
        frame_number (int): Frame nummer voor dynamische effecten
        color_scheme (str): Naam van het kleurschema
        base_color (tuple): Basis kleur (fallback)
        
    Returns:
        tuple: RGB kleur tuple
    """
    if color_scheme is None:
        color_scheme = DEFAULT_COLOR_SCHEME
    
    if TEXT_COLOR_SCHEME_BASED and color_scheme:
        # Gebruik kleurschema voor tekstkleuren
        char_progress = char_index / max(1, total_chars - 1)
        
        # Map karakter positie naar kleur uit kleurschema
        base_color = map_value_to_color(char_progress, 0.0, 1.0, color_scheme)
        
        # Voeg tijdelijke variatie toe
        if DYNAMIC_COLORS.get('time_based', True):
            time_factor = frame_number / TOTAL_FRAMES
            base_color = apply_temporal_color_variation(base_color, time_factor, variation_intensity=0.2)
        
        # Voeg gradient animatie toe
        if DYNAMIC_COLORS.get('gradient_animation', True):
            animation_progress = (frame_number + char_index * 5) / TOTAL_FRAMES
            gradient_color = create_gradient_animation_color(base_color, animation_progress, color_scheme)
            # Blend tussen base en gradient kleur
            blend_factor = 0.3
            r = int(base_color[0] * (1 - blend_factor) + gradient_color[0] * blend_factor)
            g = int(base_color[1] * (1 - blend_factor) + gradient_color[1] * blend_factor)
            b = int(base_color[2] * (1 - blend_factor) + gradient_color[2] * blend_factor)
            base_color = (r, g, b)
        
        return base_color
    else:
        # Gebruik standaard kleur
        return base_color or TEXT_COLOR


def apply_fmri_realism_to_text(text_image, activity_level=0.7, time_factor=0.0, preserve_readability=True):
    """
    Past fMRI realisme effecten toe op tekst terwijl leesbaarheid behouden blijft.
    
    Args:
        text_image (PIL.Image): Tekst afbeelding om realisme aan toe te voegen
        activity_level (float): Activiteitsniveau (0.0-1.0)
        time_factor (float): Tijd factor voor temporele effecten
        preserve_readability (bool): Behoud leesbaarheid van tekst
        
    Returns:
        PIL.Image: Tekst afbeelding met fMRI realisme effecten
    """
    # Aangepaste instellingen voor tekst (minder agressief dan voor andere elementen)
    text_realism_settings = FMRI_REALISM.copy()
    
    if preserve_readability:
        # Verminder effecten om leesbaarheid te behouden
        text_realism_settings['voxel_opacity'] = min(0.2, FMRI_REALISM.get('voxel_opacity', 0.3))
        text_realism_settings['noise_level'] = min(0.05, FMRI_REALISM.get('noise_level', 0.1))
        text_realism_settings['smoothing_kernel'] = max(1, FMRI_REALISM.get('smoothing_kernel', 3) - 1)
        text_realism_settings['edge_enhancement'] = True  # Belangrijk voor tekst leesbaarheid
        text_realism_settings['gradient_strength'] = min(0.5, FMRI_REALISM.get('gradient_strength', 0.7))
    
    # Pas voxel textuur toe (subtiel voor tekst)
    if text_realism_settings.get('voxel_enabled', True):
        text_image = render_voxel_texture(
            text_image,
            text_realism_settings.get('voxel_size', 2),
            text_realism_settings.get('voxel_opacity', 0.2),
            'subtle'  # Altijd subtiele voxel stijl voor tekst
        )
    
    # Pas lichte spatial smoothing toe
    if text_realism_settings.get('smoothing_enabled', True):
        text_image = apply_spatial_smoothing_image(
            text_image,
            text_realism_settings.get('smoothing_kernel', 2),  # Kleinere kernel voor tekst
            preserve_edges=True  # Altijd randen behouden voor tekst
        )
    
    # Verbeter randen (belangrijk voor tekst leesbaarheid)
    if text_realism_settings.get('edge_enhancement', True):
        text_image = enhance_edges_fmri_style(
            text_image,
            text_realism_settings.get('edge_glow_radius', 2),
            text_realism_settings.get('edge_glow_intensity', 0.6),  # Verhoogd voor tekst
            text_realism_settings.get('edge_detection_threshold', 0.2)
        )
    
    # Voeg subtiele gradiënt grenzen toe
    if text_realism_settings.get('gradient_boundaries', True):
        text_image = create_gradient_boundaries(
            text_image,
            text_realism_settings.get('gradient_width', 3),  # Kleinere width voor tekst
            text_realism_settings.get('gradient_falloff', 'gaussian'),
            text_realism_settings.get('gradient_strength', 0.5)  # Verminderde sterkte
        )
    
    # Voeg zeer subtiele anatomische variatie toe
    if text_realism_settings.get('anatomical_variation', True) and not preserve_readability:
        text_image = add_anatomical_variation(
            text_image,
            text_realism_settings.get('anatomical_asymmetry', 0.05),  # Zeer subtiel
            hotspots=False,  # Geen hotspots voor tekst
            gradients=True
        )
    
    return text_image


def render_partial_text(text, visible_chars, font, position, color, frame_number, 
                       background_size, color_scheme=None, enable_fmri_realism=True):
    """
    Rendert gedeeltelijke tekst voor animatie frames met enhanced kleuren en fMRI realisme.
    
    Args:
        text (str): Volledige tekst
        visible_chars (int): Aantal zichtbare karakters
        font (PIL.ImageFont): Font object
        position (tuple): (x, y) positie van tekst
        color (tuple): RGB basis kleur
        frame_number (int): Frame nummer voor pulsing effect
        background_size (tuple): (width, height) van achtergrond afbeelding
        color_scheme (str): Naam van het kleurschema
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
        
    Returns:
        PIL.Image: Afbeelding met gedeeltelijke tekst
    """
    # Maak transparante afbeelding met dezelfde afmetingen als achtergrond
    img = Image.new('RGBA', background_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Render elk zichtbaar karakter met eigen kleur
    visible_text = text[:visible_chars]
    if visible_text:
        current_x = position[0]
        
        for i, char in enumerate(visible_text):
            # Bepaal kleur voor dit karakter
            char_color = get_text_color(i, len(text), frame_number, color_scheme, color)
            
            # Bereken pulsing kleur
            pulse_position = (frame_number * TEXT_PULSE_SPEED + i * 0.2) / FRAMES_PER_SECOND
            pulsing_color = create_pulsing_color(char_color, pulse_position, color_scheme)
            
            # Simuleer z-score mapping voor wetenschappelijke authenticiteit
            if FMRI_REALISM.get('zscore_mapping', True):
                char_activity = (i + 1) / len(text)  # Activiteit neemt toe per karakter
                zscore, is_significant, intensity = simulate_zscore_mapping(
                    char_activity,
                    FMRI_REALISM.get('zscore_range', (-3.0, 6.0)),
                    FMRI_REALISM.get('zscore_threshold', 2.3)
                )
                
                # Moduleer kleur gebaseerd op significantie
                if is_significant:
                    # Verhoog intensiteit voor significante activatie
                    pulsing_color = tuple(min(255, int(c * (1.0 + intensity * 0.3))) for c in pulsing_color)
                else:
                    # Verminder intensiteit voor niet-significante activatie
                    pulsing_color = tuple(int(c * 0.7) for c in pulsing_color)
            
            # Teken karakter
            draw.text((current_x, position[1]), char, font=font, fill=pulsing_color)
            
            # Bereken positie voor volgend karakter
            char_bbox = font.getbbox(char)
            char_width = char_bbox[2] - char_bbox[0]
            current_x += char_width
    
    # Pas fMRI realisme toe indien ingeschakeld
    if enable_fmri_realism and visible_chars > 0:
        # Bereken activiteitsniveau gebaseerd op zichtbare karakters
        activity_level = min(1.0, visible_chars / len(text))
        time_factor = frame_number / TOTAL_FRAMES
        
        img = apply_fmri_realism_to_text(
            img, 
            activity_level=activity_level, 
            time_factor=time_factor,
            preserve_readability=True
        )
    
    return img


def create_text_appearing_animation(text=None, font_size=None, color=None, 
                                  position_type=None, animation_speed=None, 
                                  color_scheme=None, enable_fmri_realism=True):
    """
    Hoofdfunctie voor het creëren van tekst verschijnend animatie met fMRI realisme.
    
    Args:
        text (str): Tekst om te animeren (standaard uit constants)
        font_size (int): Font grootte (standaard uit constants)
        color (tuple): RGB kleur (standaard uit constants)
        position_type (str): Positionering type (standaard uit constants)
        animation_speed (float): Animatie snelheid (standaard uit constants)
        color_scheme (str): Naam van het kleurschema
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
        
    Returns:
        list: Lijst van PIL.Image frames
    """
    # Gebruik standaard waarden als niet opgegeven
    if text is None:
        text = TEXT_CONTENT
    if font_size is None:
        font_size = TEXT_FONT_SIZE
    if color is None:
        color = TEXT_COLOR
    if position_type is None:
        position_type = TEXT_POSITION
    if animation_speed is None:
        animation_speed = TEXT_ANIMATION_SPEED
    if color_scheme is None:
        color_scheme = DEFAULT_COLOR_SCHEME
    
    # Laad font en achtergrond
    font = load_font(font_size)
    background = load_background_image()
    mask = create_oval_mask(background.size)
    
    # Bereken tekst positie
    text_position = calculate_text_position(text, font, position_type)
    text_position = ensure_text_within_oval(text, font, text_position)
    
    # Bereken timing voor letter onthulling
    total_chars = len(text)
    chars_per_frame = animation_speed * FRAMES_PER_SECOND
    
    frames = []
    previous_frame_data = None
    
    for frame_num in range(TOTAL_FRAMES):
        # Bereken aantal zichtbare karakters
        progress = frame_num / TOTAL_FRAMES
        visible_chars = int(progress * total_chars / chars_per_frame)
        visible_chars = min(visible_chars, total_chars)
        
        # Als alle karakters zichtbaar zijn, houd ze zichtbaar
        if frame_num > total_chars / chars_per_frame:
            visible_chars = total_chars
        
        # Maak frame met achtergrond
        frame = background.copy()
        
        if visible_chars > 0:
            # Render gedeeltelijke tekst met enhanced kleuren en realisme
            text_img = render_partial_text(
                text, visible_chars, font, text_position, color, frame_num, 
                background.size, color_scheme, enable_fmri_realism
            )
            
            # Voeg enhanced gloed effect toe
            text_with_glow = add_glow_effect(
                text_img, color, TEXT_GLOW_RADIUS, TEXT_GLOW_INTENSITY, enhanced=True
            )
            
            # Pas ovaal masker toe
            masked_text = apply_oval_mask(text_with_glow, mask)
            
            # Pas globale fMRI realisme toe op het hele frame indien ingeschakeld
            if enable_fmri_realism:
                enhanced_text, frame_data = enhance_fmri_realism(
                    masked_text,
                    frame_number=frame_num,
                    previous_frame=previous_frame_data
                )
                previous_frame_data = frame_data
                masked_text = enhanced_text
            
            # Combineer met achtergrond
            frame = composite_images(frame, masked_text, (0, 0))
        
        frames.append(frame)
    
    return frames


def create_demo_animation(color_scheme=None, enable_fmri_realism=True):
    """
    Creëert een demo versie van de tekst verschijnend animatie.
    
    Args:
        color_scheme (str): Naam van het kleurschema
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
    
    Returns:
        list: Lijst van demo frames
    """
    if color_scheme is None:
        color_scheme = DEFAULT_COLOR_SCHEME
    
    # Demo met verschillende teksten en posities
    demo_configs = [
        {"text": "fMRI", "position_type": "center"},
        {"text": "BRAIN", "position_type": "top"},
        {"text": "SCAN", "position_type": "bottom"}
    ]
    
    all_frames = []
    
    for config in demo_configs:
        frames = create_text_appearing_animation(
            text=config["text"],
            position_type=config["position_type"],
            color_scheme=color_scheme,
            enable_fmri_realism=enable_fmri_realism
        )
        all_frames.extend(frames)
    
    return all_frames


def generate_text_appearing_gif(output_filename=None, demo=False, color_scheme=None, enable_fmri_realism=True):
    """
    Genereert een GIF van de tekst verschijnend animatie.
    
    Args:
        output_filename (str): Naam van output bestand (standaard uit constants)
        demo (bool): Of demo versie moet worden gegenereerd
        color_scheme (str): Naam van het kleurschema
        enable_fmri_realism (bool): Schakel fMRI realisme effecten in
        
    Returns:
        str: Pad naar gegenereerd GIF bestand
    """
    if output_filename is None:
        suffix = "_fmri_realism" if enable_fmri_realism else ""
        output_filename = OUTPUT_FILENAMES['text_appearing'].replace('.gif', f'{suffix}.gif')
    if color_scheme is None:
        color_scheme = DEFAULT_COLOR_SCHEME
    
    print("Genereren tekst verschijnend animatie...")
    print(f"- Kleurschema: {color_scheme}")
    print(f"- Kleurschema-gebaseerde kleuren: {TEXT_COLOR_SCHEME_BASED}")
    print(f"- fMRI realisme: {enable_fmri_realism}")
    
    if demo:
        frames = create_demo_animation(color_scheme, enable_fmri_realism)
        print(f"Demo animatie gegenereerd met {len(frames)} frames")
    else:
        frames = create_text_appearing_animation(color_scheme=color_scheme, enable_fmri_realism=enable_fmri_realism)
        print(f"Animatie gegenereerd met {len(frames)} frames")
    
    # Maak GIF
    gif_path = create_gif_from_frames(frames, output_filename)
    print(f"GIF opgeslagen als: {gif_path}")
    
    return gif_path


def create_fmri_realism_demo():
    """
    Creëert demo animaties die het verschil tonen tussen normale en fMRI-realisme versies.
    
    Returns:
        list: Lijst van paden naar gegenereerde GIFs
    """
    demo_files = []
    
    try:
        print("\n=== fMRI Realisme Demo voor Tekst ===")
        
        # Test verschillende kleurschema's met en zonder realisme
        color_schemes = ['hot', 'viridis']
        
        for scheme in color_schemes:
            print(f"\n--- Kleurschema: {scheme} ---")
            
            # Normale versie
            normal_path = generate_text_appearing_gif(
                output_filename=f"tekst_{scheme}_normal.gif",
                demo=False,
                color_scheme=scheme,
                enable_fmri_realism=False
            )
            demo_files.append(normal_path)
            
            # fMRI realisme versie
            realism_path = generate_text_appearing_gif(
                output_filename=f"tekst_{scheme}_fmri_realism.gif",
                demo=False,
                color_scheme=scheme,
                enable_fmri_realism=True
            )
            demo_files.append(realism_path)
        
        print(f"\n✅ {len(demo_files)} tekst demo bestanden gegenereerd")
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
    
    print("\n=== Genereren tekst verschijnend kleurschema demo's ===")
    
    for scheme in schemes:
        print(f"\n--- Kleurschema: {scheme} ---")
        suffix = "_realism" if enable_fmri_realism else ""
        output_filename = f"tekst_verschijnend_{scheme}{suffix}.gif"
        
        path = generate_text_appearing_gif(
            output_filename=output_filename,
            demo=False,
            color_scheme=scheme,
            enable_fmri_realism=enable_fmri_realism
        )
        demo_paths.append(path)
    
    print(f"\n✅ {len(demo_paths)} kleurschema demo's voltooid")
    return demo_paths


if __name__ == "__main__":
    # Test de enhanced fMRI realisme functionaliteit
    try:
        print("=== Test Enhanced fMRI Realisme voor Tekst Verschijnend ===")
        
        # Test fMRI realisme demo
        print("\n--- fMRI Realisme Demo ---")
        demo_paths = create_fmri_realism_demo()
        print(f"Demo bestanden: {demo_paths}")
        
        # Test standaard animatie met realisme
        gif_path = generate_text_appearing_gif(enable_fmri_realism=True)
        print(f"Standaard animatie met realisme: {gif_path}")
        
        # Test verschillende kleurschema's met realisme
        scheme_demos = create_color_scheme_demos(enable_fmri_realism=True)
        print(f"Kleurschema demo's met realisme: {scheme_demos}")
        
        # Test demo animatie met realisme
        demo_path = generate_text_appearing_gif(
            "tekst_verschijnend_demo_realism.gif", 
            demo=True, 
            enable_fmri_realism=True
        )
        print(f"Demo animatie met realisme: {demo_path}")
        
        print("Test voltooid!")
        
    except Exception as e:
        print(f"Test gefaald: {str(e)}")