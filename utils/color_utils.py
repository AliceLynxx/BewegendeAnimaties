"""
Kleur utilities voor fMRI-stijl kleuren en gloed effecten.
"""

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from config.constants import FMRI_COLORS, GLOW_RADIUS, GLOW_INTENSITY


def get_fmri_color(color_name='primary'):
    """
    Geeft een fMRI kleur terug op basis van naam.
    
    Args:
        color_name (str): Naam van de kleur ('primary', 'secondary', 'accent', 'highlight')
        
    Returns:
        tuple: RGB kleur tuple
    """
    return FMRI_COLORS.get(color_name, FMRI_COLORS['primary'])


def create_gradient_color(start_color, end_color, position):
    """
    Creëert een gradient kleur tussen twee kleuren.
    
    Args:
        start_color (tuple): Start RGB kleur
        end_color (tuple): Eind RGB kleur  
        position (float): Positie in gradient (0.0 - 1.0)
        
    Returns:
        tuple: RGB kleur op gegeven positie
    """
    position = max(0.0, min(1.0, position))  # Clamp tussen 0 en 1
    
    r = int(start_color[0] + (end_color[0] - start_color[0]) * position)
    g = int(start_color[1] + (end_color[1] - start_color[1]) * position)
    b = int(start_color[2] + (end_color[2] - start_color[2]) * position)
    
    return (r, g, b)


def create_fmri_gradient(steps=10):
    """
    Creëert een fMRI kleur gradient van oranje naar geel.
    
    Args:
        steps (int): Aantal stappen in de gradient
        
    Returns:
        list: Lijst van RGB kleur tuples
    """
    start_color = get_fmri_color('primary')  # Oranje
    end_color = get_fmri_color('highlight')  # Geel
    
    gradient = []
    for i in range(steps):
        position = i / (steps - 1) if steps > 1 else 0
        color = create_gradient_color(start_color, end_color, position)
        gradient.append(color)
    
    return gradient


def add_glow_effect(image, color=None, radius=None, intensity=None):
    """
    Voegt gloed effect toe aan een afbeelding.
    
    Args:
        image (PIL.Image): Afbeelding om gloed aan toe te voegen
        color (tuple): RGB kleur voor gloed (standaard fMRI primary)
        radius (int): Gloed radius (standaard uit constants)
        intensity (float): Gloed intensiteit 0.0-1.0 (standaard uit constants)
        
    Returns:
        PIL.Image: Afbeelding met gloed effect
    """
    if color is None:
        color = get_fmri_color('primary')
    if radius is None:
        radius = GLOW_RADIUS
    if intensity is None:
        intensity = GLOW_INTENSITY
    
    # Converteer naar RGBA als nodig
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Maak gloed laag
    glow = Image.new('RGBA', image.size, (0, 0, 0, 0))
    
    # Kopieer alpha channel van origineel voor gloed vorm
    alpha = image.split()[-1]  # Alpha channel
    
    # Maak gekleurde versie van alpha voor gloed
    glow_colored = Image.new('RGBA', image.size, color + (0,))
    glow_colored.putalpha(alpha)
    
    # Pas blur toe voor gloed effect
    glow_blurred = glow_colored.filter(ImageFilter.GaussianBlur(radius=radius))
    
    # Verminder intensiteit
    if intensity < 1.0:
        # Maak alpha zwakker
        alpha_array = np.array(glow_blurred.split()[-1])
        alpha_array = (alpha_array * intensity).astype(np.uint8)
        glow_blurred.putalpha(Image.fromarray(alpha_array))
    
    # Combineer gloed met origineel
    result = Image.new('RGBA', image.size, (0, 0, 0, 0))
    result.paste(glow_blurred, (0, 0), glow_blurred)
    result.paste(image, (0, 0), image)
    
    return result


def create_colored_circle(size, color=None, alpha=255):
    """
    Creëert een gekleurde cirkel met fMRI kleuren.
    
    Args:
        size (int): Diameter van de cirkel
        color (tuple): RGB kleur (standaard fMRI primary)
        alpha (int): Transparantie 0-255
        
    Returns:
        PIL.Image: Cirkel afbeelding
    """
    if color is None:
        color = get_fmri_color('primary')
    
    # Maak transparante afbeelding
    circle = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(circle)
    
    # Teken gevulde cirkel
    draw.ellipse([0, 0, size-1, size-1], fill=color + (alpha,))
    
    return circle


def create_pulsing_color(base_color, pulse_position):
    """
    Creëert een pulserende kleur effect.
    
    Args:
        base_color (tuple): Basis RGB kleur
        pulse_position (float): Puls positie 0.0-1.0
        
    Returns:
        tuple: RGB kleur met puls effect
    """
    # Gebruik sinus voor smooth pulsing
    pulse_factor = (np.sin(pulse_position * 2 * np.pi) + 1) / 2  # 0.0 - 1.0
    
    # Mix met highlight kleur voor puls effect
    highlight = get_fmri_color('highlight')
    
    r = int(base_color[0] + (highlight[0] - base_color[0]) * pulse_factor * 0.3)
    g = int(base_color[1] + (highlight[1] - base_color[1]) * pulse_factor * 0.3)
    b = int(base_color[2] + (highlight[2] - base_color[2]) * pulse_factor * 0.3)
    
    return (r, g, b)


def get_color_palette():
    """
    Geeft het volledige fMRI kleurenpalet terug.
    
    Returns:
        dict: Dictionary met alle fMRI kleuren
    """
    return FMRI_COLORS.copy()