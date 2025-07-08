"""
Kleur utilities voor fMRI-stijl kleuren en gloed effecten.
Uitgebreid met neuroimaging standaard kleurschalen, vloeiende overgangen, fMRI-realisme en heatmap enhancement.
"""

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import math
import random
from config.constants import (
    FMRI_COLORS, FMRI_COLOR_SCHEMES, DEFAULT_COLOR_SCHEME, COLOR_MAPPING,
    GLOW_RADIUS, GLOW_INTENSITY, ENHANCED_GLOW_RADIUS, ENHANCED_GLOW_INTENSITY,
    DYNAMIC_COLORS, FMRI_REALISM, HEATMAP_ENHANCEMENT
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


# ===== NIEUWE HEATMAP ENHANCEMENT FUNCTIES =====

def create_heatmap_gradient(intensity_levels=None, color_scheme=None, logarithmic=None):
    """
    Creëert een echte heatmap-stijl gradiënt met realistische intensiteit mapping.
    
    Args:
        intensity_levels (int): Aantal intensiteitsniveaus
        color_scheme (str): Naam van het kleurschema
        logarithmic (bool): Gebruik logaritmische schaling voor realistische intensiteit
        
    Returns:
        list: Lijst van RGB kleur tuples voor heatmap gradiënt
    """
    if intensity_levels is None:
        intensity_levels = HEATMAP_ENHANCEMENT.get('intensity_levels', 12)
    if logarithmic is None:
        logarithmic = HEATMAP_ENHANCEMENT.get('intensity_mapping', 'logarithmic') == 'logarithmic'
    
    scheme = get_color_scheme(color_scheme)
    base_colors = scheme['colors']
    
    if logarithmic:
        # Logaritmische schaling voor realistische fMRI intensiteit
        levels = []
        for i in range(intensity_levels):
            # Logaritmische verdeling (meer detail in lage intensiteiten)
            log_pos = math.log(i + 1) / math.log(intensity_levels)
            levels.append(log_pos)
    else:
        # Lineaire verdeling
        levels = [i / (intensity_levels - 1) for i in range(intensity_levels)]
    
    # Map levels naar kleuren
    heatmap_colors = []
    for level in levels:
        color = map_value_to_color(level, 0.0, 1.0, color_scheme)
        heatmap_colors.append(color)
    
    return heatmap_colors


def apply_heatmap_blending(base_image, heatmap_overlay, blend_mode=None, opacity=None):
    """
    Past authentieke heatmap blending toe voor realistische compositing.
    
    Args:
        base_image (PIL.Image): Basis afbeelding (achtergrond)
        heatmap_overlay (PIL.Image): Heatmap overlay
        blend_mode (str): Blending mode ('screen', 'overlay', 'multiply', 'normal')
        opacity (float): Opacity van heatmap overlay (0.0-1.0)
        
    Returns:
        PIL.Image: Afbeelding met heatmap blending
    """
    if blend_mode is None:
        blend_mode = HEATMAP_ENHANCEMENT.get('blending_mode', 'screen')
    if opacity is None:
        opacity = HEATMAP_ENHANCEMENT.get('blending_opacity', 0.8)
    
    if base_image.mode != 'RGBA':
        base_image = base_image.convert('RGBA')
    if heatmap_overlay.mode != 'RGBA':
        heatmap_overlay = heatmap_overlay.convert('RGBA')
    
    # Zorg dat afbeeldingen dezelfde grootte hebben
    if base_image.size != heatmap_overlay.size:
        heatmap_overlay = heatmap_overlay.resize(base_image.size, Image.LANCZOS)
    
    # Converteer naar numpy arrays voor blending
    base_array = np.array(base_image).astype(np.float32) / 255.0
    overlay_array = np.array(heatmap_overlay).astype(np.float32) / 255.0
    
    # Pas opacity toe op overlay
    overlay_array[:, :, 3] *= opacity
    
    # Verschillende blending modes
    if blend_mode == 'screen':
        # Screen blending: 1 - (1-base) * (1-overlay)
        result_rgb = 1.0 - (1.0 - base_array[:, :, :3]) * (1.0 - overlay_array[:, :, :3])
    elif blend_mode == 'overlay':
        # Overlay blending: complexere formule
        result_rgb = _apply_overlay_blend(base_array[:, :, :3], overlay_array[:, :, :3])
    elif blend_mode == 'multiply':
        # Multiply blending: base * overlay
        result_rgb = base_array[:, :, :3] * overlay_array[:, :, :3]
    else:  # normal
        # Normal alpha blending
        alpha = overlay_array[:, :, 3:4]
        result_rgb = base_array[:, :, :3] * (1.0 - alpha) + overlay_array[:, :, :3] * alpha
    
    # Combineer met alpha channel
    result_alpha = np.maximum(base_array[:, :, 3], overlay_array[:, :, 3])
    result = np.concatenate([result_rgb, result_alpha[:, :, np.newaxis]], axis=2)
    
    # Converteer terug naar PIL Image
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result, 'RGBA')


def _apply_overlay_blend(base, overlay):
    """Hulpfunctie voor overlay blending."""
    result = np.zeros_like(base)
    
    # Voor elke kleurcomponent
    for i in range(3):
        base_channel = base[:, :, i]
        overlay_channel = overlay[:, :, i]
        
        # Overlay formule: if base < 0.5: 2*base*overlay, else: 1-2*(1-base)*(1-overlay)
        mask = base_channel < 0.5
        result[:, :, i] = np.where(
            mask,
            2 * base_channel * overlay_channel,
            1 - 2 * (1 - base_channel) * (1 - overlay_channel)
        )
    
    return result


def enhance_heatmap_contrast(image, contrast_factor=None, adaptive=None, preserve_range=None):
    """
    Verbetert contrast voor authentieke heatmap-look zonder oversaturatie.
    
    Args:
        image (PIL.Image): Afbeelding om contrast te verbeteren
        contrast_factor (float): Contrast versterking factor
        adaptive (bool): Gebruik adaptieve contrast verbetering
        preserve_range (bool): Behoud kleur bereik
        
    Returns:
        PIL.Image: Afbeelding met verbeterd contrast
    """
    if not HEATMAP_ENHANCEMENT.get('enabled', True):
        return image
    
    if contrast_factor is None:
        contrast_factor = HEATMAP_ENHANCEMENT.get('contrast_enhancement', 1.2)
    if adaptive is None:
        adaptive = HEATMAP_ENHANCEMENT.get('adaptive_contrast', True)
    if preserve_range is None:
        preserve_range = HEATMAP_ENHANCEMENT.get('contrast_preserve_range', True)
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    if adaptive:
        # Adaptieve contrast verbetering per kleurkanaal
        return _apply_adaptive_contrast(image, contrast_factor, preserve_range)
    else:
        # Standaard contrast verbetering
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(contrast_factor)
        
        if preserve_range:
            # Behoud originele kleur bereik
            enhanced = _preserve_color_range(image, enhanced)
        
        return enhanced


def _apply_adaptive_contrast(image, contrast_factor, preserve_range):
    """Past adaptieve contrast verbetering toe."""
    try:
        from scipy import ndimage
    except ImportError:
        # Fallback naar standaard contrast als scipy niet beschikbaar is
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(contrast_factor)
    
    # Converteer naar numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Pas contrast toe per kleurkanaal
    for i in range(3):  # RGB kanalen
        channel = img_array[:, :, i]
        
        # Bereken lokaal gemiddelde voor adaptieve contrast
        local_mean = ndimage.uniform_filter(channel, size=15)
        
        # Pas contrast toe relatief tot lokaal gemiddelde
        enhanced_channel = local_mean + (channel - local_mean) * contrast_factor
        
        # Clamp waarden
        img_array[:, :, i] = np.clip(enhanced_channel, 0.0, 1.0)
    
    # Converteer terug naar PIL Image
    result = (img_array * 255).astype(np.uint8)
    enhanced = Image.fromarray(result, 'RGBA')
    
    if preserve_range:
        enhanced = _preserve_color_range(image, enhanced)
    
    return enhanced


def _preserve_color_range(original, enhanced):
    """Behoudt originele kleur bereik na enhancement."""
    # Simpele implementatie - in praktijk complexer
    return Image.blend(original, enhanced, 0.7)


def apply_heatmap_transparency(image, transparency_levels=None, intensity_based=None):
    """
    Past realistische transparantie effecten toe voor heatmap-stijl.
    
    Args:
        image (PIL.Image): Afbeelding om transparantie toe te passen
        transparency_levels (list): Lijst van transparantie niveaus (0.0-1.0)
        intensity_based (bool): Transparantie gebaseerd op intensiteit
        
    Returns:
        PIL.Image: Afbeelding met heatmap transparantie
    """
    if transparency_levels is None:
        transparency_levels = HEATMAP_ENHANCEMENT.get('transparency_levels', [0.7, 0.8, 0.9, 1.0])
    if intensity_based is None:
        intensity_based = HEATMAP_ENHANCEMENT.get('intensity_based_transparency', True)
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Converteer naar numpy array
    img_array = np.array(image).astype(np.float32)
    
    if intensity_based:
        # Bereken intensiteit gebaseerd op kleur helderheid
        rgb = img_array[:, :, :3]
        intensity = np.mean(rgb, axis=2) / 255.0
        
        # Map intensiteit naar transparantie niveaus
        alpha_channel = img_array[:, :, 3]
        
        for i, level in enumerate(transparency_levels):
            threshold = i / len(transparency_levels)
            mask = (intensity >= threshold) & (intensity < threshold + 1/len(transparency_levels))
            alpha_channel[mask] = alpha_channel[mask] * level
    else:
        # Uniforme transparantie
        alpha_channel = img_array[:, :, 3] * transparency_levels[-1]
    
    # Update alpha channel
    img_array[:, :, 3] = np.clip(alpha_channel, 0, 255)
    
    # Converteer terug naar PIL Image
    result = img_array.astype(np.uint8)
    return Image.fromarray(result, 'RGBA')


def create_intensity_heatmap(intensity_data, color_scheme=None, zscore_mapping=None):
    """
    Creëert intensiteit-gebaseerde heatmap kleuren zoals in echte fMRI software.
    
    Args:
        intensity_data (numpy.ndarray): 2D array met intensiteitswaarden
        color_scheme (str): Naam van het kleurschema
        zscore_mapping (bool): Gebruik z-score gebaseerde mapping
        
    Returns:
        PIL.Image: Heatmap afbeelding met intensiteit kleuren
    """
    if zscore_mapping is None:
        zscore_mapping = HEATMAP_ENHANCEMENT.get('zscore_based_mapping', True)
    
    height, width = intensity_data.shape
    
    # Normaliseer intensiteit data
    if zscore_mapping:
        # Simuleer z-score mapping
        normalized_data = np.zeros_like(intensity_data)
        for i in range(height):
            for j in range(width):
                zscore, is_significant, intensity = simulate_zscore_mapping(
                    intensity_data[i, j],
                    FMRI_REALISM.get('zscore_range', (-3.0, 6.0)),
                    FMRI_REALISM.get('zscore_threshold', 2.3)
                )
                normalized_data[i, j] = intensity if is_significant else 0.0
    else:
        # Standaard normalisatie
        min_val, max_val = np.min(intensity_data), np.max(intensity_data)
        if max_val > min_val:
            normalized_data = (intensity_data - min_val) / (max_val - min_val)
        else:
            normalized_data = intensity_data
    
    # Maak heatmap afbeelding
    heatmap_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Vul pixels met heatmap kleuren
    for i in range(height):
        for j in range(width):
            intensity = normalized_data[i, j]
            if intensity > 0:
                color = map_value_to_color(intensity, 0.0, 1.0, color_scheme)
                alpha = int(255 * min(1.0, intensity * 2))  # Verhoog alpha voor zichtbaarheid
                heatmap_image.putpixel((j, i), color + (alpha,))
    
    return heatmap_image


def apply_smooth_color_transitions(image, transition_width=None, method=None):
    """
    Past vloeiende kleurovergangen toe voor natuurlijke heatmap-look.
    
    Args:
        image (PIL.Image): Afbeelding om overgangen toe te passen
        transition_width (int): Breedte van overgangszone
        method (str): Methode voor overgangen ('gaussian', 'linear', 'cubic')
        
    Returns:
        PIL.Image: Afbeelding met vloeiende overgangen
    """
    if transition_width is None:
        transition_width = HEATMAP_ENHANCEMENT.get('gradient_transition_width', 3)
    if method is None:
        method = HEATMAP_ENHANCEMENT.get('gradient_method', 'gaussian')
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    if method == 'gaussian':
        # Gaussische blur voor vloeiende overgangen
        blurred = image.filter(ImageFilter.GaussianBlur(radius=transition_width))
        
        # Blend met origineel voor behoud van detail
        result = Image.blend(image, blurred, 0.6)
    elif method == 'linear':
        # Lineaire interpolatie tussen pixels
        result = _apply_linear_transitions(image, transition_width)
    else:  # cubic of fallback
        # Cubic interpolatie (vereenvoudigd)
        result = image.filter(ImageFilter.SMOOTH_MORE)
    
    return result


def _apply_linear_transitions(image, width):
    """Past lineaire overgangen toe tussen pixels."""
    # Vereenvoudigde implementatie - in praktijk complexer
    return image.filter(ImageFilter.BoxBlur(radius=width))


def enhance_color_saturation(image, saturation_factor=None, intensity_based=True):
    """
    Verhoogt kleur saturatie voor levendige heatmap-effecten.
    
    Args:
        image (PIL.Image): Afbeelding om saturatie te verhogen
        saturation_factor (float): Saturatie versterking factor
        intensity_based (bool): Saturatie gebaseerd op intensiteit
        
    Returns:
        PIL.Image: Afbeelding met verhoogde saturatie
    """
    if not HEATMAP_ENHANCEMENT.get('enabled', True):
        return image
    
    if saturation_factor is None:
        saturation_factor = HEATMAP_ENHANCEMENT.get('color_saturation', 1.3)
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    if intensity_based:
        # Variabele saturatie gebaseerd op intensiteit
        return _apply_intensity_based_saturation(image, saturation_factor)
    else:
        # Uniforme saturatie verhoging
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(saturation_factor)


def _apply_intensity_based_saturation(image, base_factor):
    """Past intensiteit-gebaseerde saturatie toe."""
    # Converteer naar HSV voor saturatie manipulatie
    hsv_image = image.convert('HSV')
    hsv_array = np.array(hsv_image).astype(np.float32)
    
    # Bereken intensiteit van originele RGB
    rgb_array = np.array(image.convert('RGB')).astype(np.float32)
    intensity = np.mean(rgb_array, axis=2) / 255.0
    
    # Pas saturatie toe gebaseerd op intensiteit
    saturation_channel = hsv_array[:, :, 1]
    
    # Hogere intensiteit = hogere saturatie
    intensity_factor = 1.0 + intensity * (base_factor - 1.0)
    enhanced_saturation = saturation_channel * intensity_factor[:, :, np.newaxis]
    enhanced_saturation = np.clip(enhanced_saturation, 0, 255)
    
    hsv_array[:, :, 1] = enhanced_saturation.squeeze()
    
    # Converteer terug naar RGBA
    enhanced_hsv = Image.fromarray(hsv_array.astype(np.uint8), 'HSV')
    enhanced_rgb = enhanced_hsv.convert('RGBA')
    
    # Behoud originele alpha channel
    enhanced_rgb.putalpha(image.split()[-1])
    
    return enhanced_rgb


def apply_adaptive_brightness(image, brightness_levels=5, spatial_adaptation=None):
    """
    Past adaptieve helderheid toe per regio voor realistische heatmap-look.
    
    Args:
        image (PIL.Image): Afbeelding om helderheid aan te passen
        brightness_levels (int): Aantal helderheid niveaus
        spatial_adaptation (bool): Ruimtelijke adaptatie van helderheid
        
    Returns:
        PIL.Image: Afbeelding met adaptieve helderheid
    """
    if spatial_adaptation is None:
        spatial_adaptation = HEATMAP_ENHANCEMENT.get('adaptive_brightness', True)
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    if spatial_adaptation:
        return _apply_spatial_brightness_adaptation(image, brightness_levels)
    else:
        # Uniforme helderheid aanpassing
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        brightness_factor = HEATMAP_ENHANCEMENT.get('brightness_enhancement', 1.1)
        return enhancer.enhance(brightness_factor)


def _apply_spatial_brightness_adaptation(image, levels):
    """Past ruimtelijke helderheid adaptatie toe."""
    # Converteer naar numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    height, width = img_array.shape[:2]
    
    # Creëer helderheid masker gebaseerd op positie
    y_coords, x_coords = np.ogrid[:height, :width]
    
    # Radiaal helderheid patroon (centrum helderder)
    center_y, center_x = height // 2, width // 2
    distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    # Normaliseer afstand en creëer helderheid factor
    normalized_distance = distance / max_distance
    brightness_factor = 1.0 + 0.3 * (1.0 - normalized_distance)  # Centrum 30% helderder
    
    # Pas helderheid toe op RGB kanalen
    for i in range(3):
        img_array[:, :, i] *= brightness_factor
    
    # Clamp waarden
    img_array = np.clip(img_array, 0.0, 1.0)
    
    # Converteer terug naar PIL Image
    result = (img_array * 255).astype(np.uint8)
    return Image.fromarray(result, 'RGBA')


def apply_gaussian_heatmap_blur(image, sigma=None, preserve_edges=None):
    """
    Past specifieke Gaussische blur toe voor heatmap-effect met rand behoud.
    
    Args:
        image (PIL.Image): Afbeelding om te blurren
        sigma (float): Blur sterkte (sigma waarde)
        preserve_edges (bool): Behoud scherpe randen
        
    Returns:
        PIL.Image: Geblurde afbeelding met heatmap-effect
    """
    if sigma is None:
        sigma = HEATMAP_ENHANCEMENT.get('gaussian_blur_sigma', 1.5)
    if preserve_edges is None:
        preserve_edges = HEATMAP_ENHANCEMENT.get('blur_preserve_edges', True)
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    if preserve_edges:
        # Edge-preserving blur
        return _apply_edge_preserving_heatmap_blur(image, sigma)
    else:
        # Standaard Gaussische blur
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def _apply_edge_preserving_heatmap_blur(image, sigma):
    """Past edge-preserving blur toe specifiek voor heatmaps."""
    # Detecteer randen
    edges = image.filter(ImageFilter.FIND_EDGES)
    
    # Pas blur toe
    blurred = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    # Gebruik randen als masker om detail te behouden
    # Vereenvoudigde implementatie
    result = Image.blend(blurred, image, 0.3)
    
    return result


def create_activation_hotspots(image, num_hotspots=None, intensity_range=None):
    """
    Creëert realistische activatie hotspots voor authentieke fMRI-look.
    
    Args:
        image (PIL.Image): Basis afbeelding
        num_hotspots (int): Aantal hotspots om te creëren
        intensity_range (tuple): (min, max) intensiteit voor hotspots
        
    Returns:
        PIL.Image: Afbeelding met activatie hotspots
    """
    if num_hotspots is None:
        num_hotspots = HEATMAP_ENHANCEMENT.get('hotspot_count', 3)
    if intensity_range is None:
        intensity_range = HEATMAP_ENHANCEMENT.get('hotspot_intensity_range', (0.6, 0.9))
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    width, height = image.size
    hotspot_overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Genereer willekeurige hotspots
    size_range = HEATMAP_ENHANCEMENT.get('hotspot_size_range', (15, 40))
    
    for _ in range(num_hotspots):
        # Willekeurige positie
        center_x = np.random.randint(width // 4, 3 * width // 4)
        center_y = np.random.randint(height // 4, 3 * height // 4)
        
        # Willekeurige radius
        radius = np.random.randint(*size_range)
        
        # Willekeurige intensiteit
        intensity = np.random.uniform(*intensity_range)
        
        # Creëer hotspot
        hotspot = _create_single_hotspot(
            (width, height), (center_x, center_y), radius, intensity
        )
        
        # Voeg toe aan overlay
        hotspot_overlay = Image.alpha_composite(hotspot_overlay, hotspot)
    
    # Combineer met originele afbeelding
    result = Image.alpha_composite(image, hotspot_overlay)
    
    return result


def _create_single_hotspot(size, center, radius, intensity):
    """Creëert een enkele hotspot."""
    width, height = size
    center_x, center_y = center
    
    hotspot = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Creëer Gaussische hotspot
    for y in range(max(0, center_y - radius), min(height, center_y + radius)):
        for x in range(max(0, center_x - radius), min(width, center_x + radius)):
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            if distance <= radius:
                # Gaussische falloff
                gaussian_factor = math.exp(-(distance**2) / (2 * (radius/3)**2))
                alpha = int(255 * intensity * gaussian_factor)
                
                # Gebruik heatmap kleur
                color = map_value_to_color(intensity, 0.0, 1.0, DEFAULT_COLOR_SCHEME)
                hotspot.putpixel((x, y), color + (alpha,))
    
    return hotspot


def enhance_spatial_gradients(image, gradient_strength=None, direction=None):
    """
    Verbetert ruimtelijke gradiënten voor natuurlijke activatie patronen.
    
    Args:
        image (PIL.Image): Afbeelding om gradiënten te verbeteren
        gradient_strength (float): Sterkte van gradiënt effect (0.0-1.0)
        direction (str): Richting van gradiënt ('radial', 'horizontal', 'vertical')
        
    Returns:
        PIL.Image: Afbeelding met verbeterde gradiënten
    """
    if gradient_strength is None:
        gradient_strength = HEATMAP_ENHANCEMENT.get('gradient_strength', 0.7)
    if direction is None:
        direction = HEATMAP_ENHANCEMENT.get('gradient_direction', 'radial')
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    width, height = image.size
    
    # Creëer gradiënt masker
    if direction == 'radial':
        gradient_mask = _create_radial_gradient_mask((width, height))
    elif direction == 'horizontal':
        gradient_mask = _create_horizontal_gradient_mask((width, height))
    elif direction == 'vertical':
        gradient_mask = _create_vertical_gradient_mask((width, height))
    else:
        gradient_mask = _create_radial_gradient_mask((width, height))
    
    # Pas gradiënt toe op afbeelding
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Moduleer alpha channel met gradiënt
    alpha_channel = img_array[:, :, 3]
    modulated_alpha = alpha_channel * (1.0 - gradient_strength * (1.0 - gradient_mask))
    img_array[:, :, 3] = modulated_alpha
    
    # Converteer terug naar PIL Image
    result = (np.clip(img_array, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(result, 'RGBA')


def _create_radial_gradient_mask(size):
    """Creëert radiaal gradiënt masker."""
    width, height = size
    center_x, center_y = width // 2, height // 2
    
    y_coords, x_coords = np.ogrid[:height, :width]
    distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    # Normaliseer en inverteer (centrum = 1.0, randen = 0.0)
    gradient = 1.0 - (distance / max_distance)
    gradient = np.clip(gradient, 0.0, 1.0)
    
    return gradient


def _create_horizontal_gradient_mask(size):
    """Creëert horizontaal gradiënt masker."""
    width, height = size
    gradient = np.linspace(0.0, 1.0, width)
    return np.tile(gradient, (height, 1))


def _create_vertical_gradient_mask(size):
    """Creëert verticaal gradiënt masker."""
    width, height = size
    gradient = np.linspace(0.0, 1.0, height).reshape(-1, 1)
    return np.tile(gradient, (1, width))


def apply_edge_softening(image, softening_strength=None, method=None):
    """
    Past zachte randen toe zoals in echte fMRI-scans.
    
    Args:
        image (PIL.Image): Afbeelding om randen te verzachten
        softening_strength (float): Sterkte van rand verzachting (0.0-1.0)
        method (str): Methode voor verzachting ('gaussian', 'bilateral', 'morphological')
        
    Returns:
        PIL.Image: Afbeelding met zachte randen
    """
    if softening_strength is None:
        softening_strength = HEATMAP_ENHANCEMENT.get('edge_softening', 0.8)
    if method is None:
        method = HEATMAP_ENHANCEMENT.get('edge_softening_method', 'gaussian')
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    if method == 'gaussian':
        # Gaussische rand verzachting
        return _apply_gaussian_edge_softening(image, softening_strength)
    elif method == 'bilateral':
        # Bilateral filter voor rand behoud
        return _apply_bilateral_edge_softening(image, softening_strength)
    else:  # morphological
        # Morfologische operaties
        return _apply_morphological_edge_softening(image, softening_strength)


def _apply_gaussian_edge_softening(image, strength):
    """Past Gaussische rand verzachting toe."""
    # Detecteer randen
    alpha = image.split()[-1]
    edges = alpha.filter(ImageFilter.FIND_EDGES)
    
    # Blur randen
    softened_edges = edges.filter(ImageFilter.GaussianBlur(radius=2 * strength))
    
    # Combineer met origineel
    result = image.copy()
    
    # Moduleer alpha channel met verzachte randen
    alpha_array = np.array(alpha).astype(np.float32)
    edges_array = np.array(softened_edges).astype(np.float32) / 255.0
    
    # Verminder alpha waar randen zijn
    modulated_alpha = alpha_array * (1.0 - edges_array * strength)
    modulated_alpha = np.clip(modulated_alpha, 0, 255).astype(np.uint8)
    
    result.putalpha(Image.fromarray(modulated_alpha))
    
    return result


def _apply_bilateral_edge_softening(image, strength):
    """Past bilateral filter toe voor rand verzachting."""
    # Vereenvoudigde implementatie
    return image.filter(ImageFilter.SMOOTH)


def _apply_morphological_edge_softening(image, strength):
    """Past morfologische rand verzachting toe."""
    # Vereenvoudigde implementatie
    return image.filter(ImageFilter.MaxFilter(size=int(3 * strength)))


def integrate_heatmap_effects(image, settings=None, frame_number=0):
    """
    Hoofdfunctie die alle heatmap effecten integreert voor authentieke fMRI-look.
    
    Args:
        image (PIL.Image): Originele afbeelding
        settings (dict): Heatmap instellingen (gebruikt HEATMAP_ENHANCEMENT als standaard)
        frame_number (int): Frame nummer voor temporele effecten
        
    Returns:
        PIL.Image: Afbeelding met geïntegreerde heatmap effecten
    """
    if settings is None:
        settings = HEATMAP_ENHANCEMENT
    
    if not settings.get('enabled', True):
        return image
    
    result = image.copy()
    
    # 1. Verbeter kleur saturatie
    if settings.get('color_saturation', 1.0) != 1.0:
        result = enhance_color_saturation(
            result, 
            settings.get('color_saturation', 1.3),
            intensity_based=True
        )
    
    # 2. Pas adaptieve helderheid toe
    if settings.get('adaptive_brightness', True):
        result = apply_adaptive_brightness(
            result,
            brightness_levels=5,
            spatial_adaptation=True
        )
    
    # 3. Voeg vloeiende kleurovergangen toe
    if settings.get('smooth_gradients', True):
        result = apply_smooth_color_transitions(
            result,
            transition_width=settings.get('gradient_transition_width', 3),
            method=settings.get('gradient_method', 'gaussian')
        )
    
    # 4. Pas Gaussische blur toe voor heatmap-effect
    if settings.get('gaussian_blur_sigma', 0) > 0:
        result = apply_gaussian_heatmap_blur(
            result,
            sigma=settings.get('gaussian_blur_sigma', 1.5),
            preserve_edges=settings.get('blur_preserve_edges', True)
        )
    
    # 5. Voeg activatie hotspots toe
    if settings.get('hotspot_enhancement', True):
        result = create_activation_hotspots(
            result,
            num_hotspots=settings.get('hotspot_count', 3),
            intensity_range=settings.get('hotspot_intensity_range', (0.6, 0.9))
        )
    
    # 6. Verbeter ruimtelijke gradiënten
    if settings.get('spatial_gradients', True):
        result = enhance_spatial_gradients(
            result,
            gradient_strength=settings.get('gradient_strength', 0.7),
            direction=settings.get('gradient_direction', 'radial')
        )
    
    # 7. Pas rand verzachting toe
    if settings.get('edge_softening', 0) > 0:
        result = apply_edge_softening(
            result,
            softening_strength=settings.get('edge_softening', 0.8),
            method=settings.get('edge_softening_method', 'gaussian')
        )
    
    # 8. Verbeter contrast
    if settings.get('contrast_enhancement', 1.0) != 1.0:
        result = enhance_heatmap_contrast(
            result,
            contrast_factor=settings.get('contrast_enhancement', 1.2),
            adaptive=settings.get('adaptive_contrast', True),
            preserve_range=settings.get('contrast_preserve_range', True)
        )
    
    # 9. Pas transparantie effecten toe
    if settings.get('transparency_levels'):
        result = apply_heatmap_transparency(
            result,
            transparency_levels=settings.get('transparency_levels'),
            intensity_based=settings.get('intensity_based_transparency', True)
        )
    
    return result


def optimize_heatmap_visibility(image, enhancement_factor=1.2, preserve_detail=True):
    """
    Optimaliseert zichtbaarheid van heatmap voor maximale impact.
    
    Args:
        image (PIL.Image): Afbeelding om te optimaliseren
        enhancement_factor (float): Versterking factor voor zichtbaarheid
        preserve_detail (bool): Behoud detail tijdens optimalisatie
        
    Returns:
        PIL.Image: Geoptimaliseerde afbeelding
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Verhoog saturatie voor betere zichtbaarheid
    enhanced = enhance_color_saturation(image, enhancement_factor, intensity_based=True)
    
    # Verbeter contrast
    enhanced = enhance_heatmap_contrast(enhanced, enhancement_factor, adaptive=True)
    
    if preserve_detail:
        # Blend met origineel om detail te behouden
        enhanced = Image.blend(image, enhanced, 0.8)
    
    return enhanced


def apply_scientific_color_mapping(intensity_data, color_scheme=None, statistical_threshold=None):
    """
    Past wetenschappelijk accurate kleurmapping toe zoals in fMRI software.
    
    Args:
        intensity_data (numpy.ndarray): 2D array met intensiteitswaarden
        color_scheme (str): Naam van het kleurschema
        statistical_threshold (bool): Pas statistische thresholding toe
        
    Returns:
        PIL.Image: Wetenschappelijk accurate heatmap
    """
    if statistical_threshold is None:
        statistical_threshold = HEATMAP_ENHANCEMENT.get('statistical_thresholding', True)
    
    if statistical_threshold:
        # Pas statistische thresholding toe
        thresholded_data = apply_statistical_thresholding(
            intensity_data,
            threshold=FMRI_REALISM.get('threshold_level', 0.3),
            threshold_type=FMRI_REALISM.get('threshold_type', 'soft'),
            fade_zone=FMRI_REALISM.get('threshold_fade', 0.1)
        )
    else:
        thresholded_data = intensity_data
    
    # Creëer heatmap met wetenschappelijke kleuren
    heatmap = create_intensity_heatmap(
        thresholded_data,
        color_scheme=color_scheme,
        zscore_mapping=HEATMAP_ENHANCEMENT.get('zscore_based_mapping', True)
    )
    
    return heatmap


def enhance_depth_perception(image, depth_factor=None, method=None):
    """
    Verbetert diepte perceptie voor 3D-achtige heatmap effecten.
    
    Args:
        image (PIL.Image): Afbeelding om diepte aan toe te voegen
        depth_factor (float): Sterkte van diepte effect (0.0-1.0)
        method (str): Methode voor diepte ('gradient', 'shadow', 'highlight')
        
    Returns:
        PIL.Image: Afbeelding met verbeterde diepte perceptie
    """
    if depth_factor is None:
        depth_factor = HEATMAP_ENHANCEMENT.get('depth_factor', 0.3)
    if method is None:
        method = HEATMAP_ENHANCEMENT.get('depth_method', 'gradient')
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    if method == 'gradient':
        # Gebruik gradiënt voor diepte effect
        return enhance_spatial_gradients(image, depth_factor, 'radial')
    elif method == 'shadow':
        # Voeg schaduw effecten toe
        return _add_shadow_depth(image, depth_factor)
    else:  # highlight
        # Voeg highlight effecten toe
        return _add_highlight_depth(image, depth_factor)


def _add_shadow_depth(image, depth_factor):
    """Voegt schaduw effecten toe voor diepte."""
    # Creëer schaduw laag
    shadow = image.copy()
    
    # Maak schaduw donkerder
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(shadow)
    shadow = enhancer.enhance(0.7)
    
    # Offset schaduw
    shadow_offset = Image.new('RGBA', image.size, (0, 0, 0, 0))
    shadow_offset.paste(shadow, (2, 2), shadow)
    
    # Combineer met origineel
    result = Image.alpha_composite(shadow_offset, image)
    
    return result


def _add_highlight_depth(image, depth_factor):
    """Voegt highlight effecten toe voor diepte."""
    # Creëer highlight laag
    highlight = image.copy()
    
    # Maak highlight helderder
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(highlight)
    highlight = enhancer.enhance(1.3)
    
    # Verminder opacity van highlight
    alpha = highlight.split()[-1]
    alpha_array = np.array(alpha) * depth_factor
    highlight.putalpha(Image.fromarray(alpha_array.astype(np.uint8)))
    
    # Combineer met origineel
    result = Image.alpha_composite(image, highlight)
    
    return result


# ===== BESTAANDE fMRI REALISME FUNCTIES =====

def generate_fmri_noise(size, noise_type='gaussian', noise_level=0.1, temporal_factor=0.0):
    """
    Genereert realistische fMRI ruis patronen.
    
    Args:
        size (tuple): (width, height) van de ruis
        noise_type (str): Type ruis ('gaussian', 'uniform', 'salt_pepper')
        noise_level (float): Intensiteit van ruis (0.0-1.0)
        temporal_factor (float): Tijdelijke variatie factor (0.0-1.0)
        
    Returns:
        numpy.ndarray: 2D array met ruis waarden (-1.0 tot 1.0)
    """
    width, height = size
    
    if noise_type == 'gaussian':
        # Gaussische ruis (meest realistisch voor fMRI)
        noise = np.random.normal(0, noise_level, (height, width))
    elif noise_type == 'uniform':
        # Uniforme ruis
        noise = np.random.uniform(-noise_level, noise_level, (height, width))
    elif noise_type == 'salt_pepper':
        # Salt-and-pepper ruis
        noise = np.zeros((height, width))
        salt_pepper_mask = np.random.random((height, width)) < noise_level
        noise[salt_pepper_mask] = np.random.choice([-1, 1], size=np.sum(salt_pepper_mask)) * noise_level
    else:
        # Fallback naar gaussische ruis
        noise = np.random.normal(0, noise_level, (height, width))
    
    # Voeg tijdelijke variatie toe
    if temporal_factor > 0:
        temporal_modulation = 1.0 + 0.3 * math.sin(temporal_factor * 2 * math.pi)
        noise *= temporal_modulation
    
    # Clamp tussen -1 en 1
    noise = np.clip(noise, -1.0, 1.0)
    
    return noise


def apply_statistical_thresholding(image_array, threshold=0.3, threshold_type='soft', fade_zone=0.1):
    """
    Past statistische thresholding toe zoals in fMRI analyse.
    
    Args:
        image_array (numpy.ndarray): 2D array met activatiewaarden (0.0-1.0)
        threshold (float): Drempelwaarde (0.0-1.0)
        threshold_type (str): Type thresholding ('hard', 'soft')
        fade_zone (float): Fade zone voor soft thresholding (0.0-1.0)
        
    Returns:
        numpy.ndarray: Gethresholde array
    """
    if not FMRI_REALISM.get('threshold_enabled', True):
        return image_array
    
    thresholded = image_array.copy()
    
    if threshold_type == 'hard':
        # Harde thresholding: alles onder drempel wordt 0
        thresholded[thresholded < threshold] = 0.0
    elif threshold_type == 'soft':
        # Zachte thresholding: geleidelijke overgang
        fade_start = threshold - fade_zone / 2
        fade_end = threshold + fade_zone / 2
        
        # Onder fade zone: 0
        thresholded[thresholded < fade_start] = 0.0
        
        # In fade zone: geleidelijke overgang
        fade_mask = (thresholded >= fade_start) & (thresholded <= fade_end)
        fade_values = (thresholded[fade_mask] - fade_start) / fade_zone
        thresholded[fade_mask] = fade_values * thresholded[fade_mask]
    
    return thresholded


def simulate_zscore_mapping(activity_level, zscore_range=(-3.0, 6.0), threshold=2.3):
    """
    Simuleert z-score gebaseerde kleurmapping zoals in fMRI software.
    
    Args:
        activity_level (float): Activiteitsniveau (0.0-1.0)
        zscore_range (tuple): (min_z, max_z) z-score bereik
        threshold (float): Significantie drempel (z-score)
        
    Returns:
        tuple: (zscore, is_significant, intensity)
    """
    if not FMRI_REALISM.get('zscore_mapping', True):
        return activity_level, True, activity_level
    
    min_z, max_z = zscore_range
    
    # Map activiteit naar z-score
    zscore = min_z + activity_level * (max_z - min_z)
    
    # Bepaal significantie
    is_significant = abs(zscore) >= threshold
    
    # Bereken intensiteit gebaseerd op z-score
    if is_significant:
        # Normaliseer significante z-scores naar 0-1
        if zscore >= 0:
            intensity = min(1.0, (zscore - threshold) / (max_z - threshold))
        else:
            intensity = min(1.0, (abs(zscore) - threshold) / (abs(min_z) - threshold))
    else:
        intensity = 0.0
    
    return zscore, is_significant, intensity


def enhance_fmri_realism(image, settings=None, frame_number=0, previous_frame=None):
    """
    Hoofdfunctie die alle fMRI-realisme effecten toepast.
    
    Args:
        image (PIL.Image): Originele afbeelding
        settings (dict): Custom instellingen (gebruikt FMRI_REALISM als standaard)
        frame_number (int): Frame nummer voor temporele effecten
        previous_frame (numpy.ndarray): Vorige frame voor temporele correlatie
        
    Returns:
        tuple: (enhanced_image, frame_data) - enhanced PIL.Image en numpy array voor volgende frame
    """
    if settings is None:
        settings = FMRI_REALISM
    
    # Converteer naar numpy array voor verwerking
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Extraheer alpha channel voor activatie intensiteit
    alpha_array = np.array(image.split()[-1]) / 255.0
    
    # Temporele factor voor dynamische effecten
    temporal_factor = frame_number / 30.0  # Assumeer ~30 frames cyclus
    
    # 1. Voeg ruis toe
    if settings.get('noise_level', 0) > 0:
        noise = generate_fmri_noise(
            image.size, 
            settings.get('noise_type', 'gaussian'),
            settings.get('noise_level', 0.1),
            temporal_factor if settings.get('noise_temporal', True) else 0.0
        )
        alpha_array = np.clip(alpha_array + noise * 0.1, 0.0, 1.0)
    
    # 2. Pas statistical thresholding toe
    if settings.get('threshold_enabled', True):
        alpha_array = apply_statistical_thresholding(
            alpha_array,
            settings.get('threshold_level', 0.3),
            settings.get('threshold_type', 'soft'),
            settings.get('threshold_fade', 0.1)
        )
    
    # 3. Pas temporele correlatie toe
    if previous_frame is not None:
        alpha_array = apply_temporal_correlation(
            alpha_array,
            previous_frame,
            settings.get('temporal_correlation', 0.8)
        )
    
    # Converteer terug naar PIL Image
    enhanced_alpha = (alpha_array * 255).astype(np.uint8)
    
    # Maak nieuwe afbeelding met enhanced alpha
    enhanced_image = image.copy()
    enhanced_image.putalpha(Image.fromarray(enhanced_alpha))
    
    return enhanced_image, alpha_array


def apply_temporal_correlation(current_frame, previous_frame, correlation=0.8):
    """
    Past temporele correlatie toe tussen opeenvolgende frames.
    
    Args:
        current_frame (numpy.ndarray): Huidige frame activatie
        previous_frame (numpy.ndarray): Vorige frame activatie
        correlation (float): Correlatie factor (0.0-1.0)
        
    Returns:
        numpy.ndarray: Frame met temporele correlatie
    """
    if not FMRI_REALISM.get('temporal_smoothing', True) or previous_frame is None:
        return current_frame
    
    # Gewogen gemiddelde tussen huidige en vorige frame
    correlated_frame = correlation * previous_frame + (1.0 - correlation) * current_frame
    
    return np.clip(correlated_frame, 0.0, 1.0)


# ===== BESTAANDE FUNCTIES (ongewijzigd) =====

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