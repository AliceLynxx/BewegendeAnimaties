"""
Afbeelding utilities voor BewegendeAnimaties.
Bevat functies voor achtergrond laden, ovaal masker generatie en compositing.
"""

import os
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