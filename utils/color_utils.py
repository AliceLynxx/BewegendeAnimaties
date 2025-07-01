"""
Kleur utilities voor fMRI-stijl kleuren en gloed effecten.
Uitgebreid met neuroimaging standaard kleurschalen en vloeiende overgangen.
"""

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import math
from config.constants import (
    FMRI_COLORS, FMRI_COLOR_SCHEMES, DEFAULT_COLOR_SCHEME, COLOR_MAPPING,
    GLOW_RADIUS, GLOW_INTENSITY, ENHANCED_GLOW_RADIUS, ENHANCED_GLOW_INTENSITY,
    DYNAMIC_COLORS
)


def get_fmri_color(color_name='primary'):
    """
    Geeft een fMRI kleur terug op basis van naam.
    
    Args:
        color_name (str): Naam van de kleur ('primary', 'secondary', 'accent', 'highlight')
        
    Returns:
        tuple: RGB kleur tuple
    """
    return FMRI_COLORS.get(color_name, FMRI_COLORS['primary'])


def get_color_scheme(scheme_name=None):
    """
    Geeft een kleurschema terug op basis van naam.
    
    Args:
        scheme_name (str): Naam van het kleurschema ('hot', 'cool', 'jet', 'viridis')
        
    Returns:
        dict: Kleurschema dictionary met kleuren en metadata
    """
    if scheme_name is None:
        scheme_name = DEFAULT_COLOR_SCHEME
    
    return FMRI_COLOR_SCHEMES.get(scheme_name, FMRI_COLOR_SCHEMES[DEFAULT_COLOR_SCHEME])


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


def create_smooth_gradient(colors, steps):
    """
    Creëert een vloeiende gradient met meerdere kleuren.
    
    Args:
        colors (list): Lijst van RGB kleur tuples
        steps (int): Aantal stappen in de gradient
        
    Returns:
        list: Lijst van RGB kleur tuples voor gradient
    """
    if len(colors) < 2:
        return colors * steps
    
    gradient = []
    segment_length = steps / (len(colors) - 1)
    
    for i in range(steps):
        # Bepaal welk kleurensegment we gebruiken
        segment_index = i / segment_length
        color_index = int(segment_index)
        
        # Zorg ervoor dat we binnen de grenzen blijven
        if color_index >= len(colors) - 1:
            gradient.append(colors[-1])
        else:
            # Bereken positie binnen het segment
            segment_position = segment_index - color_index
            
            # Interpoleer tussen de twee kleuren
            start_color = colors[color_index]
            end_color = colors[color_index + 1]
            interpolated_color = create_gradient_color(start_color, end_color, segment_position)
            gradient.append(interpolated_color)
    
    return gradient


def map_value_to_color(value, min_val, max_val, color_scheme=None, use_negative=False):
    """
    Mapt een waarde naar een kleur uit een kleurschema.
    
    Args:
        value (float): Waarde om te mappen
        min_val (float): Minimale waarde
        max_val (float): Maximale waarde
        color_scheme (str): Naam van het kleurschema
        use_negative (bool): Gebruik negatieve kleuren voor waarden onder 0
        
    Returns:
        tuple: RGB kleur tuple
    """
    scheme = get_color_scheme(color_scheme)
    
    # Normaliseer waarde
    if max_val == min_val:
        normalized_value = 0.5
    else:
        normalized_value = (value - min_val) / (max_val - min_val)
    
    # Gebruik negatieve kleuren voor negatieve waarden
    if use_negative and value < 0 and 'negative_colors' in scheme:
        colors = scheme['negative_colors']
        # Gebruik absolute waarde voor negatieve mapping
        normalized_value = abs(normalized_value)
    else:
        colors = scheme['colors']
    
    # Clamp tussen 0 en 1
    normalized_value = max(0.0, min(1.0, normalized_value))
    
    # Map naar kleurindex
    if len(colors) == 1:
        return colors[0]
    
    color_index = normalized_value * (len(colors) - 1)
    lower_index = int(color_index)
    upper_index = min(lower_index + 1, len(colors) - 1)
    
    # Interpoleer tussen kleuren voor vloeiende overgang
    if lower_index == upper_index:
        return colors[lower_index]
    
    interpolation_factor = color_index - lower_index
    return create_gradient_color(colors[lower_index], colors[upper_index], interpolation_factor)


def create_fmri_gradient(steps=10, color_scheme=None):
    """
    Creëert een fMRI kleur gradient.
    
    Args:
        steps (int): Aantal stappen in de gradient
        color_scheme (str): Naam van het kleurschema
        
    Returns:
        list: Lijst van RGB kleur tuples
    """
    scheme = get_color_scheme(color_scheme)
    return create_smooth_gradient(scheme['colors'], steps)


def apply_temporal_color_variation(base_color, time_factor, variation_intensity=None):
    """
    Past tijdelijke kleurvariatie toe op een basiskleur.
    
    Args:
        base_color (tuple): Basis RGB kleur
        time_factor (float): Tijd factor voor variatie (0.0-1.0)
        variation_intensity (float): Intensiteit van variatie (0.0-1.0)
        
    Returns:
        tuple: RGB kleur met tijdelijke variatie
    """
    if variation_intensity is None:
        variation_intensity = DYNAMIC_COLORS.get('intensity_variation', 0.3)
    
    # Gebruik sinus voor cyclische variatie
    variation = math.sin(time_factor * 2 * math.pi) * variation_intensity
    
    # Pas variatie toe op elke kleurcomponent
    r = int(base_color[0] * (1.0 + variation))
    g = int(base_color[1] * (1.0 + variation))
    b = int(base_color[2] * (1.0 + variation))
    
    # Clamp tussen 0 en 255
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    return (r, g, b)


def create_activity_based_color(activity_level, color_scheme=None, time_factor=0.0):
    """
    Creëert een kleur gebaseerd op activiteitsniveau.
    
    Args:
        activity_level (float): Activiteitsniveau (0.0-1.0)
        color_scheme (str): Naam van het kleurschema
        time_factor (float): Tijd factor voor dynamische effecten
        
    Returns:
        tuple: RGB kleur tuple
    """
    # Map activiteitsniveau naar kleur
    base_color = map_value_to_color(activity_level, 0.0, 1.0, color_scheme)
    
    # Voeg tijdelijke variatie toe indien ingeschakeld
    if DYNAMIC_COLORS.get('time_based', True):
        base_color = apply_temporal_color_variation(base_color, time_factor)
    
    return base_color


def create_movement_based_color(speed, max_speed, color_scheme=None, time_factor=0.0):
    """
    Creëert een kleur gebaseerd op bewegingssnelheid.
    
    Args:
        speed (float): Huidige snelheid
        max_speed (float): Maximale snelheid
        color_scheme (str): Naam van het kleurschema
        time_factor (float): Tijd factor voor dynamische effecten
        
    Returns:
        tuple: RGB kleur tuple
    """
    if not DYNAMIC_COLORS.get('movement_based', True):
        return get_fmri_color('primary')
    
    # Normaliseer snelheid
    normalized_speed = min(speed / max_speed, 1.0) if max_speed > 0 else 0.0
    
    return create_activity_based_color(normalized_speed, color_scheme, time_factor)


def add_glow_effect(image, color=None, radius=None, intensity=None, enhanced=False):
    """
    Voegt gloed effect toe aan een afbeelding.
    
    Args:
        image (PIL.Image): Afbeelding om gloed aan toe te voegen
        color (tuple): RGB kleur voor gloed (standaard fMRI primary)
        radius (int): Gloed radius (standaard uit constants)
        intensity (float): Gloed intensiteit 0.0-1.0 (standaard uit constants)
        enhanced (bool): Gebruik enhanced gloed instellingen
        
    Returns:
        PIL.Image: Afbeelding met gloed effect
    """
    if color is None:
        color = get_fmri_color('primary')
    
    if enhanced:
        if radius is None:
            radius = ENHANCED_GLOW_RADIUS
        if intensity is None:
            intensity = ENHANCED_GLOW_INTENSITY
    else:
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


def create_pulsing_color(base_color, pulse_position, color_scheme=None):
    """
    Creëert een pulserende kleur effect.
    
    Args:
        base_color (tuple): Basis RGB kleur
        pulse_position (float): Puls positie 0.0-1.0
        color_scheme (str): Kleurschema voor pulsing effect
        
    Returns:
        tuple: RGB kleur met puls effect
    """
    if not DYNAMIC_COLORS.get('pulsing_enabled', True):
        return base_color
    
    # Gebruik sinus voor smooth pulsing
    pulse_factor = (math.sin(pulse_position * 2 * math.pi) + 1) / 2  # 0.0 - 1.0
    
    # Gebruik kleurschema voor pulsing als beschikbaar
    if color_scheme:
        scheme = get_color_scheme(color_scheme)
        if len(scheme['colors']) > 1:
            # Gebruik laatste kleur van scheme als highlight
            highlight = scheme['colors'][-1]
        else:
            highlight = get_fmri_color('highlight')
    else:
        highlight = get_fmri_color('highlight')
    
    # Mix met highlight kleur voor puls effect
    r = int(base_color[0] + (highlight[0] - base_color[0]) * pulse_factor * 0.4)
    g = int(base_color[1] + (highlight[1] - base_color[1]) * pulse_factor * 0.4)
    b = int(base_color[2] + (highlight[2] - base_color[2]) * pulse_factor * 0.4)
    
    return (r, g, b)


def create_gradient_animation_color(base_color, animation_progress, color_scheme=None):
    """
    Creëert een kleur voor geanimeerde gradiënten.
    
    Args:
        base_color (tuple): Basis RGB kleur
        animation_progress (float): Animatie voortgang (0.0-1.0)
        color_scheme (str): Kleurschema voor gradient animatie
        
    Returns:
        tuple: RGB kleur voor gradient animatie
    """
    if not DYNAMIC_COLORS.get('gradient_animation', True):
        return base_color
    
    scheme = get_color_scheme(color_scheme)
    colors = scheme['colors']
    
    if len(colors) < 2:
        return base_color
    
    # Cyclische beweging door het kleurschema
    cycle_position = (animation_progress * 2) % 1.0  # 2 cycli per animatie
    
    # Map naar kleurschema
    return map_value_to_color(cycle_position, 0.0, 1.0, color_scheme)


def get_color_palette(color_scheme=None):
    """
    Geeft het volledige kleurenpalet terug.
    
    Args:
        color_scheme (str): Naam van het kleurschema
        
    Returns:
        dict: Dictionary met alle kleuren en metadata
    """
    if color_scheme:
        return get_color_scheme(color_scheme)
    else:
        return {
            'original_fmri': FMRI_COLORS.copy(),
            'schemes': FMRI_COLOR_SCHEMES.copy()
        }


def create_intensity_mapped_color(intensity, color_scheme=None, use_negative=False):
    """
    Creëert een kleur gebaseerd op intensiteitswaarde.
    
    Args:
        intensity (float): Intensiteitswaarde (-1.0 tot 1.0)
        color_scheme (str): Naam van het kleurschema
        use_negative (bool): Gebruik negatieve kleuren voor negatieve intensiteit
        
    Returns:
        tuple: RGB kleur tuple
    """
    # Bepaal of we negatieve kleuren moeten gebruiken
    if use_negative and intensity < 0:
        return map_value_to_color(abs(intensity), 0.0, 1.0, color_scheme, use_negative=True)
    else:
        # Gebruik positieve intensiteit
        positive_intensity = max(0.0, intensity)
        return map_value_to_color(positive_intensity, 0.0, 1.0, color_scheme, use_negative=False)


def blend_colors(color1, color2, blend_factor):
    """
    Mengt twee kleuren met een blend factor.
    
    Args:
        color1 (tuple): Eerste RGB kleur
        color2 (tuple): Tweede RGB kleur
        blend_factor (float): Blend factor (0.0 = color1, 1.0 = color2)
        
    Returns:
        tuple: Gemengde RGB kleur
    """
    blend_factor = max(0.0, min(1.0, blend_factor))
    
    r = int(color1[0] * (1 - blend_factor) + color2[0] * blend_factor)
    g = int(color1[1] * (1 - blend_factor) + color2[1] * blend_factor)
    b = int(color1[2] * (1 - blend_factor) + color2[2] * blend_factor)
    
    return (r, g, b)


def get_available_color_schemes():
    """
    Geeft een lijst van beschikbare kleurschema's terug.
    
    Returns:
        list: Lijst van kleurschema namen met beschrijvingen
    """
    schemes = []
    for name, scheme in FMRI_COLOR_SCHEMES.items():
        schemes.append({
            'name': name,
            'display_name': scheme.get('name', name),
            'description': scheme.get('description', ''),
            'color_count': len(scheme['colors'])
        })
    
    return schemes


def create_color_preview(color_scheme=None, width=400, height=50):
    """
    Creëert een preview afbeelding van een kleurschema.
    
    Args:
        color_scheme (str): Naam van het kleurschema
        width (int): Breedte van preview
        height (int): Hoogte van preview
        
    Returns:
        PIL.Image: Preview afbeelding van kleurschema
    """
    scheme = get_color_scheme(color_scheme)
    colors = scheme['colors']
    
    # Creëer gradient met alle kleuren
    gradient_colors = create_smooth_gradient(colors, width)
    
    # Maak preview afbeelding
    preview = Image.new('RGB', (width, height))
    
    for x in range(width):
        color = gradient_colors[x] if x < len(gradient_colors) else gradient_colors[-1]
        for y in range(height):
            preview.putpixel((x, y), color)
    
    return preview