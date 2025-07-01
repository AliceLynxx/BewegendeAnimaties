"""
Afbeelding utilities voor BewegendeAnimaties.
Bevat functies voor achtergrond laden, ovaal masker generatie en compositing.
"""

import os
import math
import random
from PIL import Image, ImageDraw
import numpy as np
from config.constants import (
    BACKGROUND_IMAGE_PATH, BACKGROUND_DEFAULT_SIZE, 
    OVAL_CENTER, OVAL_WIDTH, OVAL_HEIGHT
)


def load_background_image():
    """
    Laadt de achtergrond afbeelding (hersenafbeelding).
    
    Returns:
        PIL.Image: Achtergrond afbeelding, of een standaard grijze achtergrond
    """
    try:
        if os.path.exists(BACKGROUND_IMAGE_PATH):
            background = Image.open(BACKGROUND_IMAGE_PATH)
            return background.convert('RGBA')
        else:
            # Maak standaard grijze achtergrond als bestand niet bestaat
            background = Image.new('RGBA', BACKGROUND_DEFAULT_SIZE, (128, 128, 128, 255))
            return background
    except Exception as e:
        print(f"Fout bij laden achtergrond: {e}")
        # Fallback naar standaard achtergrond
        background = Image.new('RGBA', BACKGROUND_DEFAULT_SIZE, (128, 128, 128, 255))
        return background


def create_oval_mask(image_size):
    """
    Creëert een ovaal masker voor de hersenregio.
    
    Args:
        image_size (tuple): (width, height) van de afbeelding
        
    Returns:
        PIL.Image: Zwart-wit masker met wit ovaal
    """
    mask = Image.new('L', image_size, 0)  # Zwarte achtergrond
    draw = ImageDraw.Draw(mask)
    
    # Bereken ovaal coördinaten
    left = OVAL_CENTER[0] - OVAL_WIDTH // 2
    top = OVAL_CENTER[1] - OVAL_HEIGHT // 2
    right = OVAL_CENTER[0] + OVAL_WIDTH // 2
    bottom = OVAL_CENTER[1] + OVAL_HEIGHT // 2
    
    # Teken wit ovaal
    draw.ellipse([left, top, right, bottom], fill=255)
    
    return mask


def apply_oval_mask(image, mask):
    """
    Past ovaal masker toe op een afbeelding.
    
    Args:
        image (PIL.Image): Afbeelding om te maskeren
        mask (PIL.Image): Masker om toe te passen
        
    Returns:
        PIL.Image: Gemaskeerde afbeelding
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Maak transparante versie met masker
    masked = Image.new('RGBA', image.size, (0, 0, 0, 0))
    masked.paste(image, mask=mask)
    
    return masked


def composite_images(background, overlay, position=(0, 0)):
    """
    Combineert twee afbeeldingen door overlay op achtergrond te plaatsen.
    
    Args:
        background (PIL.Image): Achtergrond afbeelding
        overlay (PIL.Image): Overlay afbeelding
        position (tuple): (x, y) positie voor overlay
        
    Returns:
        PIL.Image: Gecombineerde afbeelding
    """
    if background.mode != 'RGBA':
        background = background.convert('RGBA')
    if overlay.mode != 'RGBA':
        overlay = overlay.convert('RGBA')
    
    # Maak kopie van achtergrond
    result = background.copy()
    
    # Plak overlay op achtergrond
    result.paste(overlay, position, overlay)
    
    return result


def get_oval_bounds():
    """
    Geeft de grenzen van het ovaal terug.
    
    Returns:
        tuple: (left, top, right, bottom) coördinaten van ovaal
    """
    left = OVAL_CENTER[0] - OVAL_WIDTH // 2
    top = OVAL_CENTER[1] - OVAL_HEIGHT // 2
    right = OVAL_CENTER[0] + OVAL_WIDTH // 2
    bottom = OVAL_CENTER[1] + OVAL_HEIGHT // 2
    
    return (left, top, right, bottom)


def point_in_oval(x, y):
    """
    Controleert of een punt binnen het ovaal ligt.
    
    Args:
        x (int): X coördinaat
        y (int): Y coördinaat
        
    Returns:
        bool: True als punt binnen ovaal ligt
    """
    # Normaliseer coördinaten naar ovaal centrum
    dx = x - OVAL_CENTER[0]
    dy = y - OVAL_CENTER[1]
    
    # Ellips vergelijking: (x/a)² + (y/b)² <= 1
    a = OVAL_WIDTH / 2
    b = OVAL_HEIGHT / 2
    
    return (dx / a) ** 2 + (dy / b) ** 2 <= 1


def is_position_in_oval(position, center=None, width=None, height=None):
    """
    Controleert of een positie binnen het ovaal valt.
    
    Args:
        position (tuple): (x, y) positie om te controleren
        center (tuple): Centrum van ovaal (standaard uit constants)
        width (int): Breedte van ovaal (standaard uit constants)
        height (int): Hoogte van ovaal (standaard uit constants)
        
    Returns:
        bool: True als positie binnen ovaal ligt
    """
    if center is None:
        center = OVAL_CENTER
    if width is None:
        width = OVAL_WIDTH
    if height is None:
        height = OVAL_HEIGHT
    
    x, y = position
    dx = x - center[0]
    dy = y - center[1]
    
    # Ellips vergelijking: (x/a)² + (y/b)² <= 1
    a = width / 2
    b = height / 2
    
    return (dx / a) ** 2 + (dy / b) ** 2 <= 1


def draw_stick_figure(draw_context, position, pose, color, size=15):
    """
    Tekent een stick figure mannetje op de gegeven draw context.
    
    Args:
        draw_context (ImageDraw.Draw): PIL ImageDraw context
        position (tuple): (x, y) centrum positie van het mannetje
        pose (int): Pose frame nummer (0-5 voor loop cyclus)
        color (tuple): RGB kleur voor het mannetje
        size (int): Grootte van het mannetje
        
    Returns:
        None (tekent direct op draw_context)
    """
    x, y = position
    
    # Basis afmetingen gebaseerd op grootte
    head_radius = size // 4
    body_length = size // 2
    arm_length = size // 3
    leg_length = size // 2
    
    # Hoofd
    head_x, head_y = x, y - body_length // 2 - head_radius
    draw_context.ellipse([
        head_x - head_radius, head_y - head_radius,
        head_x + head_radius, head_y + head_radius
    ], fill=color, outline=color)
    
    # Lichaam (verticale lijn)
    body_top = head_y + head_radius
    body_bottom = body_top + body_length
    draw_context.line([x, body_top, x, body_bottom], fill=color, width=2)
    
    # Armen - variëren per pose voor loop animatie
    arm_y = body_top + body_length // 3
    if pose % 6 in [0, 3]:  # Neutrale arm positie
        left_arm_end = (x - arm_length, arm_y)
        right_arm_end = (x + arm_length, arm_y)
    elif pose % 6 in [1, 4]:  # Armen naar voren/achteren
        left_arm_end = (x - arm_length // 2, arm_y - arm_length // 3)
        right_arm_end = (x + arm_length // 2, arm_y + arm_length // 3)
    else:  # Armen naar achteren/voren
        left_arm_end = (x - arm_length // 2, arm_y + arm_length // 3)
        right_arm_end = (x + arm_length // 2, arm_y - arm_length // 3)
    
    draw_context.line([x, arm_y, left_arm_end[0], left_arm_end[1]], fill=color, width=2)
    draw_context.line([x, arm_y, right_arm_end[0], right_arm_end[1]], fill=color, width=2)
    
    # Benen - variëren per pose voor loop animatie
    if pose % 6 in [0, 3]:  # Neutrale been positie
        left_leg_end = (x - leg_length // 3, body_bottom + leg_length)
        right_leg_end = (x + leg_length // 3, body_bottom + leg_length)
    elif pose % 6 in [1, 4]:  # Linker been naar voren
        left_leg_end = (x - leg_length // 2, body_bottom + leg_length - leg_length // 4)
        right_leg_end = (x + leg_length // 4, body_bottom + leg_length)
    else:  # Rechter been naar voren
        left_leg_end = (x - leg_length // 4, body_bottom + leg_length)
        right_leg_end = (x + leg_length // 2, body_bottom + leg_length - leg_length // 4)
    
    draw_context.line([x, body_bottom, left_leg_end[0], left_leg_end[1]], fill=color, width=2)
    draw_context.line([x, body_bottom, right_leg_end[0], right_leg_end[1]], fill=color, width=2)


def create_stick_figure(pose_frame=0, size=15, color=(255, 140, 0)):
    """
    Creëert een stick figure afbeelding met transparante achtergrond.
    
    Args:
        pose_frame (int): Frame nummer voor loop animatie (0-5)
        size (int): Grootte van het mannetje
        color (tuple): RGB kleur
        
    Returns:
        PIL.Image: Afbeelding van stick figure met transparante achtergrond
    """
    # Bereken afbeelding grootte (met wat extra ruimte)
    img_size = size * 2
    
    # Maak transparante afbeelding
    img = Image.new('RGBA', (img_size, img_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Teken stick figure in centrum
    center = (img_size // 2, img_size // 2)
    draw_stick_figure(draw, center, pose_frame, color, size)
    
    return img


def generate_random_position_in_oval(center=None, width=None, height=None, margin=10):
    """
    Genereert een willekeurige positie binnen het ovaal.
    
    Args:
        center (tuple): Centrum van ovaal (standaard uit constants)
        width (int): Breedte van ovaal (standaard uit constants)
        height (int): Hoogte van ovaal (standaard uit constants)
        margin (int): Marge vanaf rand van ovaal
        
    Returns:
        tuple: (x, y) willekeurige positie binnen ovaal
    """
    if center is None:
        center = OVAL_CENTER
    if width is None:
        width = OVAL_WIDTH
    if height is None:
        height = OVAL_HEIGHT
    
    # Reduceer afmetingen met marge
    effective_width = width - 2 * margin
    effective_height = height - 2 * margin
    
    # Genereer willekeurige positie binnen ellips
    while True:
        # Genereer punt binnen rechthoek
        x = center[0] + random.uniform(-effective_width/2, effective_width/2)
        y = center[1] + random.uniform(-effective_height/2, effective_height/2)
        
        # Controleer of punt binnen ellips ligt
        if is_position_in_oval((x, y), center, effective_width, effective_height):
            return (int(x), int(y))


def calculate_next_position(current_pos, direction, speed, center=None, width=None, height=None):
    """
    Berekent de volgende positie gebaseerd op huidige positie, richting en snelheid.
    
    Args:
        current_pos (tuple): Huidige (x, y) positie
        direction (float): Richting in radialen
        speed (float): Snelheid in pixels per frame
        center (tuple): Centrum van ovaal (standaard uit constants)
        width (int): Breedte van ovaal (standaard uit constants)
        height (int): Hoogte van ovaal (standaard uit constants)
        
    Returns:
        tuple: (nieuwe_positie, nieuwe_richting) - richting kan veranderen bij bounce
    """
    if center is None:
        center = OVAL_CENTER
    if width is None:
        width = OVAL_WIDTH
    if height is None:
        height = OVAL_HEIGHT
    
    x, y = current_pos
    
    # Bereken nieuwe positie
    new_x = x + speed * math.cos(direction)
    new_y = y + speed * math.sin(direction)
    
    # Controleer of nieuwe positie binnen ovaal ligt
    if is_position_in_oval((new_x, new_y), center, width, height):
        return ((int(new_x), int(new_y)), direction)
    
    # Als buiten ovaal, bounce van de rand
    # Simpele bounce: keer richting om en voeg wat randomness toe
    new_direction = direction + math.pi + random.uniform(-0.5, 0.5)
    
    # Normaliseer richting
    new_direction = new_direction % (2 * math.pi)
    
    # Probeer nieuwe positie met nieuwe richting
    bounce_x = x + speed * math.cos(new_direction)
    bounce_y = y + speed * math.sin(new_direction)
    
    # Als nog steeds buiten ovaal, blijf op huidige positie
    if not is_position_in_oval((bounce_x, bounce_y), center, width, height):
        return (current_pos, new_direction)
    
    return ((int(bounce_x), int(bounce_y)), new_direction)


def ensure_within_oval(position, center=None, width=None, height=None):
    """
    Zorgt ervoor dat een positie binnen het ovaal blijft door deze naar de rand te verplaatsen.
    
    Args:
        position (tuple): (x, y) positie om te controleren
        center (tuple): Centrum van ovaal (standaard uit constants)
        width (int): Breedte van ovaal (standaard uit constants)
        height (int): Hoogte van ovaal (standaard uit constants)
        
    Returns:
        tuple: (x, y) positie binnen ovaal
    """
    if center is None:
        center = OVAL_CENTER
    if width is None:
        width = OVAL_WIDTH
    if height is None:
        height = OVAL_HEIGHT
    
    x, y = position
    
    # Als al binnen ovaal, return originele positie
    if is_position_in_oval((x, y), center, width, height):
        return position
    
    # Bereken afstand tot centrum
    dx = x - center[0]
    dy = y - center[1]
    
    # Schaal naar ovaal rand
    a = width / 2
    b = height / 2
    
    # Bereken schaalfactor om op rand te komen
    scale = math.sqrt((dx / a) ** 2 + (dy / b) ** 2)
    
    if scale > 0:
        # Schaal terug naar net binnen de rand
        scale_factor = 0.95 / scale  # 0.95 om net binnen rand te blijven
        new_x = center[0] + dx * scale_factor
        new_y = center[1] + dy * scale_factor
        return (int(new_x), int(new_y))
    
    # Fallback naar centrum
    return center


def create_test_image():
    """
    Creëert een test afbeelding om de functies te testen.
    
    Returns:
        PIL.Image: Test afbeelding met achtergrond en ovaal outline
    """
    background = load_background_image()
    mask = create_oval_mask(background.size)
    
    # Teken ovaal outline op achtergrond voor visualisatie
    draw = ImageDraw.Draw(background)
    left, top, right, bottom = get_oval_bounds()
    draw.ellipse([left, top, right, bottom], outline=(255, 0, 0, 255), width=2)
    
    return background