"""
Tekst verschijnend animatie module voor BewegendeAnimaties.

Deze module implementeert een animatie waarbij tekst letter voor letter zichtbaar wordt
binnen het ovaal, met fMRI-stijl kleuren en gloed effecten.
"""

import math
import os
from PIL import Image, ImageDraw, ImageFont
from utils.image_utils import (
    load_background_image, create_oval_mask, apply_oval_mask, 
    composite_images, get_oval_bounds, point_in_oval
)
from utils.color_utils import (
    get_fmri_color, add_glow_effect, create_pulsing_color
)
from utils.gif_utils import create_gif_from_frames
from config.constants import (
    TEXT_CONTENT, TEXT_FONT_SIZE, TEXT_COLOR, TEXT_POSITION,
    TEXT_ANIMATION_SPEED, TEXT_GLOW_RADIUS, TEXT_GLOW_INTENSITY,
    TEXT_PULSE_SPEED, TEXT_FONT_PATH, OVAL_CENTER, OVAL_WIDTH, OVAL_HEIGHT,
    TOTAL_FRAMES, FRAMES_PER_SECOND, OUTPUT_FILENAMES
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


def render_partial_text(text, visible_chars, font, position, color, frame_number, background_size):
    """
    Rendert gedeeltelijke tekst voor animatie frames.
    
    Args:
        text (str): Volledige tekst
        visible_chars (int): Aantal zichtbare karakters
        font (PIL.ImageFont): Font object
        position (tuple): (x, y) positie van tekst
        color (tuple): RGB kleur
        frame_number (int): Frame nummer voor pulsing effect
        background_size (tuple): (width, height) van achtergrond afbeelding
        
    Returns:
        PIL.Image: Afbeelding met gedeeltelijke tekst
    """
    # Maak transparante afbeelding met dezelfde afmetingen als achtergrond
    img = Image.new('RGBA', background_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Bereken pulsing kleur
    pulse_position = (frame_number * TEXT_PULSE_SPEED) / FRAMES_PER_SECOND
    pulsing_color = create_pulsing_color(color, pulse_position)
    
    # Teken zichtbare deel van tekst
    visible_text = text[:visible_chars]
    if visible_text:
        # Gebruik directe positie (geen offset meer nodig)
        draw.text(position, visible_text, font=font, fill=pulsing_color)
    
    return img


def create_text_appearing_animation(text=None, font_size=None, color=None, 
                                  position_type=None, animation_speed=None):
    """
    Hoofdfunctie voor het creëren van tekst verschijnend animatie.
    
    Args:
        text (str): Tekst om te animeren (standaard uit constants)
        font_size (int): Font grootte (standaard uit constants)
        color (tuple): RGB kleur (standaard uit constants)
        position_type (str): Positionering type (standaard uit constants)
        animation_speed (float): Animatie snelheid (standaard uit constants)
        
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
            # Render gedeeltelijke tekst met achtergrond afmetingen
            text_img = render_partial_text(
                text, visible_chars, font, text_position, color, frame_num, background.size
            )
            
            # Voeg gloed effect toe
            text_with_glow = add_glow_effect(
                text_img, color, TEXT_GLOW_RADIUS, TEXT_GLOW_INTENSITY
            )
            
            # Pas ovaal masker toe
            masked_text = apply_oval_mask(text_with_glow, mask)
            
            # Combineer met achtergrond (positie (0,0) omdat afbeeldingen nu dezelfde grootte hebben)
            frame = composite_images(frame, masked_text, (0, 0))
        
        frames.append(frame)
    
    return frames


def create_demo_animation():
    """
    Creëert een demo versie van de tekst verschijnend animatie.
    
    Returns:
        list: Lijst van demo frames
    """
    # Demo met verschillende teksten en posities
    demo_configs = [
        {"text": "fMRI", "position_type": "center", "color": get_fmri_color('primary')},
        {"text": "BRAIN", "position_type": "top", "color": get_fmri_color('secondary')},
        {"text": "SCAN", "position_type": "bottom", "color": get_fmri_color('accent')}
    ]
    
    all_frames = []
    
    for config in demo_configs:
        frames = create_text_appearing_animation(
            text=config["text"],
            position_type=config["position_type"],
            color=config["color"]
        )
        all_frames.extend(frames)
    
    return all_frames


def generate_text_appearing_gif(output_filename=None, demo=False):
    """
    Genereert een GIF van de tekst verschijnend animatie.
    
    Args:
        output_filename (str): Naam van output bestand (standaard uit constants)
        demo (bool): Of demo versie moet worden gegenereerd
        
    Returns:
        str: Pad naar gegenereerd GIF bestand
    """
    if output_filename is None:
        output_filename = OUTPUT_FILENAMES['text_appearing']
    
    print("Genereren tekst verschijnend animatie...")
    
    if demo:
        frames = create_demo_animation()
        print(f"Demo animatie gegenereerd met {len(frames)} frames")
    else:
        frames = create_text_appearing_animation()
        print(f"Animatie gegenereerd met {len(frames)} frames")
    
    # Maak GIF
    gif_path = create_gif_from_frames(frames, output_filename)
    print(f"GIF opgeslagen als: {gif_path}")
    
    return gif_path


if __name__ == "__main__":
    # Test de module
    print("Test tekst verschijnend animatie...")
    
    # Genereer standaard animatie
    gif_path = generate_text_appearing_gif()
    print(f"Standaard animatie: {gif_path}")
    
    # Genereer demo animatie
    demo_path = generate_text_appearing_gif("tekst_verschijnend_demo.gif", demo=True)
    print(f"Demo animatie: {demo_path}")
    
    print("Test voltooid!")