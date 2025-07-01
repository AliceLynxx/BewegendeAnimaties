"""
Tekst verschijnend animatie module voor BewegendeAnimaties.

Deze module implementeert een animatie waarbij tekst letter voor letter zichtbaar wordt
binnen het ovaal, met enhanced fMRI-stijl kleurschalen en dynamische kleureffecten.
"""

import math
import os
from PIL import Image, ImageDraw, ImageFont
from utils.image_utils import (
    load_background_image, create_oval_mask, apply_oval_mask, 
    composite_images, get_oval_bounds, point_in_oval
)
from utils.color_utils import (
    get_fmri_color, add_glow_effect, create_pulsing_color,
    map_value_to_color, create_gradient_animation_color,
    apply_temporal_color_variation, get_color_scheme
)
from utils.gif_utils import create_gif_from_frames
from config.constants import (
    TEXT_CONTENT, TEXT_FONT_SIZE, TEXT_COLOR, TEXT_POSITION,
    TEXT_ANIMATION_SPEED, TEXT_GLOW_RADIUS, TEXT_GLOW_INTENSITY,
    TEXT_PULSE_SPEED, TEXT_FONT_PATH, OVAL_CENTER, OVAL_WIDTH, OVAL_HEIGHT,
    TOTAL_FRAMES, FRAMES_PER_SECOND, OUTPUT_FILENAMES, DEFAULT_COLOR_SCHEME,
    TEXT_COLOR_SCHEME_BASED, DYNAMIC_COLORS
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


def render_partial_text(text, visible_chars, font, position, color, frame_number, 
                       background_size, color_scheme=None):
    """
    Rendert gedeeltelijke tekst voor animatie frames met enhanced kleuren.
    
    Args:
        text (str): Volledige tekst
        visible_chars (int): Aantal zichtbare karakters
        font (PIL.ImageFont): Font object
        position (tuple): (x, y) positie van tekst
        color (tuple): RGB basis kleur
        frame_number (int): Frame nummer voor pulsing effect
        background_size (tuple): (width, height) van achtergrond afbeelding
        color_scheme (str): Naam van het kleurschema
        
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
            
            # Teken karakter
            draw.text((current_x, position[1]), char, font=font, fill=pulsing_color)
            
            # Bereken positie voor volgend karakter
            char_bbox = font.getbbox(char)
            char_width = char_bbox[2] - char_bbox[0]
            current_x += char_width
    
    return img


def create_text_appearing_animation(text=None, font_size=None, color=None, 
                                  position_type=None, animation_speed=None, color_scheme=None):
    """
    Hoofdfunctie voor het creëren van tekst verschijnend animatie.
    
    Args:
        text (str): Tekst om te animeren (standaard uit constants)
        font_size (int): Font grootte (standaard uit constants)
        color (tuple): RGB kleur (standaard uit constants)
        position_type (str): Positionering type (standaard uit constants)
        animation_speed (float): Animatie snelheid (standaard uit constants)
        color_scheme (str): Naam van het kleurschema
        
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
            # Render gedeeltelijke tekst met enhanced kleuren
            text_img = render_partial_text(
                text, visible_chars, font, text_position, color, frame_num, 
                background.size, color_scheme
            )
            
            # Voeg enhanced gloed effect toe
            text_with_glow = add_glow_effect(
                text_img, color, TEXT_GLOW_RADIUS, TEXT_GLOW_INTENSITY, enhanced=True
            )
            
            # Pas ovaal masker toe
            masked_text = apply_oval_mask(text_with_glow, mask)
            
            # Combineer met achtergrond
            frame = composite_images(frame, masked_text, (0, 0))
        
        frames.append(frame)
    
    return frames


def create_demo_animation(color_scheme=None):
    """
    Creëert een demo versie van de tekst verschijnend animatie.
    
    Args:
        color_scheme (str): Naam van het kleurschema
    
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
            color_scheme=color_scheme
        )
        all_frames.extend(frames)
    
    return all_frames


def generate_text_appearing_gif(output_filename=None, demo=False, color_scheme=None):
    """
    Genereert een GIF van de tekst verschijnend animatie.
    
    Args:
        output_filename (str): Naam van output bestand (standaard uit constants)
        demo (bool): Of demo versie moet worden gegenereerd
        color_scheme (str): Naam van het kleurschema
        
    Returns:
        str: Pad naar gegenereerd GIF bestand
    """
    if output_filename is None:
        output_filename = OUTPUT_FILENAMES['text_appearing']
    if color_scheme is None:
        color_scheme = DEFAULT_COLOR_SCHEME
    
    print("Genereren tekst verschijnend animatie...")
    print(f"- Kleurschema: {color_scheme}")
    print(f"- Kleurschema-gebaseerde kleuren: {TEXT_COLOR_SCHEME_BASED}")
    
    if demo:
        frames = create_demo_animation(color_scheme)
        print(f"Demo animatie gegenereerd met {len(frames)} frames")
    else:
        frames = create_text_appearing_animation(color_scheme=color_scheme)
        print(f"Animatie gegenereerd met {len(frames)} frames")
    
    # Maak GIF
    gif_path = create_gif_from_frames(frames, output_filename)
    print(f"GIF opgeslagen als: {gif_path}")
    
    return gif_path


def create_color_scheme_demos():
    """
    Creëert demo animaties voor alle beschikbare kleurschema's.
    
    Returns:
        list: Lijst van paden naar gegenereerde GIFs
    """
    schemes = ['hot', 'cool', 'jet', 'viridis']
    demo_paths = []
    
    print("\n=== Genereren tekst verschijnend kleurschema demo's ===")
    
    for scheme in schemes:
        print(f"\n--- Kleurschema: {scheme} ---")
        output_filename = f"tekst_verschijnend_{scheme}.gif"
        
        path = generate_text_appearing_gif(
            output_filename=output_filename,
            demo=False,
            color_scheme=scheme
        )
        demo_paths.append(path)
    
    print(f"\n✅ {len(demo_paths)} kleurschema demo's voltooid")
    return demo_paths


if __name__ == "__main__":
    # Test de enhanced fMRI kleuren functionaliteit
    try:
        print("=== Test Enhanced fMRI Kleuren voor Tekst Verschijnend ===")
        
        # Test standaard animatie
        gif_path = generate_text_appearing_gif()
        print(f"Standaard animatie: {gif_path}")
        
        # Test verschillende kleurschema's
        scheme_demos = create_color_scheme_demos()
        print(f"Kleurschema demo's: {scheme_demos}")
        
        # Test demo animatie
        demo_path = generate_text_appearing_gif("tekst_verschijnend_demo.gif", demo=True)
        print(f"Demo animatie: {demo_path}")
        
        print("Test voltooid!")
        
    except Exception as e:
        print(f"Test gefaald: {str(e)}")