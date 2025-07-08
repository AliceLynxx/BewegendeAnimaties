"""
Heatmap utilities voor enhanced fMRI-stijl heatmap rendering.

Deze module bevat gespecialiseerde functies voor het creëren van authentieke
fMRI-heatmap visualisaties met realistische kleurmapping, blending en effecten.
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from scipy import ndimage
from config.constants import (
    FMRI_COLOR_SCHEMES, DEFAULT_COLOR_SCHEME, FMRI_REALISM,
    HEATMAP_ENHANCEMENT
)
from utils.color_utils import (
    get_color_scheme, map_value_to_color, create_smooth_gradient,
    simulate_zscore_mapping
)


def create_heatmap_gradient(intensity_levels=12, color_scheme=None, logarithmic=True):
    """
    Creëert een echte heatmap-stijl gradiënt met realistische intensiteit mapping.
    
    Args:
        intensity_levels (int): Aantal intensiteitsniveaus
        color_scheme (str): Naam van het kleurschema
        logarithmic (bool): Gebruik logaritmische schaling voor realistische intensiteit
        
    Returns:
        list: Lijst van RGB kleur tuples voor heatmap gradiënt
    """
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


def apply_heatmap_blending(base_image, heatmap_overlay, blend_mode='screen', opacity=0.8):
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


def enhance_heatmap_contrast(image, contrast_factor=1.2, adaptive=True, preserve_range=True):
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
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    if adaptive:
        # Adaptieve contrast verbetering per kleurkanaal
        return _apply_adaptive_contrast(image, contrast_factor, preserve_range)
    else:
        # Standaard contrast verbetering
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(contrast_factor)
        
        if preserve_range:
            # Behoud originele kleur bereik
            enhanced = _preserve_color_range(image, enhanced)
        
        return enhanced


def _apply_adaptive_contrast(image, contrast_factor, preserve_range):
    """Past adaptieve contrast verbetering toe."""
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


def apply_heatmap_transparency(image, transparency_levels=None, intensity_based=True):
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


def create_intensity_heatmap(intensity_data, color_scheme=None, zscore_mapping=True):
    """
    Creëert intensiteit-gebaseerde heatmap kleuren zoals in echte fMRI software.
    
    Args:
        intensity_data (numpy.ndarray): 2D array met intensiteitswaarden
        color_scheme (str): Naam van het kleurschema
        zscore_mapping (bool): Gebruik z-score gebaseerde mapping
        
    Returns:
        PIL.Image: Heatmap afbeelding met intensiteit kleuren
    """
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


def apply_smooth_color_transitions(image, transition_width=3, method='gaussian'):
    """
    Past vloeiende kleurovergangen toe voor natuurlijke heatmap-look.
    
    Args:
        image (PIL.Image): Afbeelding om overgangen toe te passen
        transition_width (int): Breedte van overgangszone
        method (str): Methode voor overgangen ('gaussian', 'linear', 'cubic')
        
    Returns:
        PIL.Image: Afbeelding met vloeiende overgangen
    """
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


def enhance_color_saturation(image, saturation_factor=1.3, intensity_based=True):
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
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    if intensity_based:
        # Variabele saturatie gebaseerd op intensiteit
        return _apply_intensity_based_saturation(image, saturation_factor)
    else:
        # Uniforme saturatie verhoging
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


def apply_adaptive_brightness(image, brightness_levels=5, spatial_adaptation=True):
    """
    Past adaptieve helderheid toe per regio voor realistische heatmap-look.
    
    Args:
        image (PIL.Image): Afbeelding om helderheid aan te passen
        brightness_levels (int): Aantal helderheid niveaus
        spatial_adaptation (bool): Ruimtelijke adaptatie van helderheid
        
    Returns:
        PIL.Image: Afbeelding met adaptieve helderheid
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    if spatial_adaptation:
        return _apply_spatial_brightness_adaptation(image, brightness_levels)
    else:
        # Uniforme helderheid aanpassing
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.1)


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


def apply_gaussian_heatmap_blur(image, sigma=1.5, preserve_edges=True):
    """
    Past specifieke Gaussische blur toe voor heatmap-effect met rand behoud.
    
    Args:
        image (PIL.Image): Afbeelding om te blurren
        sigma (float): Blur sterkte (sigma waarde)
        preserve_edges (bool): Behoud scherpe randen
        
    Returns:
        PIL.Image: Geblurde afbeelding met heatmap-effect
    """
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


def create_activation_hotspots(image, num_hotspots=3, intensity_range=(0.7, 1.0)):
    """
    Creëert realistische activatie hotspots voor authentieke fMRI-look.
    
    Args:
        image (PIL.Image): Basis afbeelding
        num_hotspots (int): Aantal hotspots om te creëren
        intensity_range (tuple): (min, max) intensiteit voor hotspots
        
    Returns:
        PIL.Image: Afbeelding met activatie hotspots
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    width, height = image.size
    hotspot_overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Genereer willekeurige hotspots
    for _ in range(num_hotspots):
        # Willekeurige positie
        center_x = np.random.randint(width // 4, 3 * width // 4)
        center_y = np.random.randint(height // 4, 3 * height // 4)
        
        # Willekeurige radius
        radius = np.random.randint(15, 40)
        
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


def enhance_spatial_gradients(image, gradient_strength=0.8, direction='radial'):
    """
    Verbetert ruimtelijke gradiënten voor natuurlijke activatie patronen.
    
    Args:
        image (PIL.Image): Afbeelding om gradiënten te verbeteren
        gradient_strength (float): Sterkte van gradiënt effect (0.0-1.0)
        direction (str): Richting van gradiënt ('radial', 'horizontal', 'vertical')
        
    Returns:
        PIL.Image: Afbeelding met verbeterde gradiënten
    """
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


def apply_edge_softening(image, softening_strength=0.8, method='gaussian'):
    """
    Past zachte randen toe zoals in echte fMRI-scans.
    
    Args:
        image (PIL.Image): Afbeelding om randen te verzachten
        softening_strength (float): Sterkte van rand verzachting (0.0-1.0)
        method (str): Methode voor verzachting ('gaussian', 'bilateral', 'morphological')
        
    Returns:
        PIL.Image: Afbeelding met zachte randen
    """
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
            transition_width=3,
            method='gaussian'
        )
    
    # 4. Pas Gaussische blur toe voor heatmap-effect
    if settings.get('gaussian_blur_sigma', 0) > 0:
        result = apply_gaussian_heatmap_blur(
            result,
            sigma=settings.get('gaussian_blur_sigma', 1.5),
            preserve_edges=True
        )
    
    # 5. Voeg activatie hotspots toe
    if settings.get('hotspot_enhancement', True):
        result = create_activation_hotspots(
            result,
            num_hotspots=2,
            intensity_range=(0.6, 0.9)
        )
    
    # 6. Verbeter ruimtelijke gradiënten
    if settings.get('depth_simulation', True):
        result = enhance_spatial_gradients(
            result,
            gradient_strength=0.7,
            direction='radial'
        )
    
    # 7. Pas rand verzachting toe
    if settings.get('edge_softening', 0) > 0:
        result = apply_edge_softening(
            result,
            softening_strength=settings.get('edge_softening', 0.8),
            method='gaussian'
        )
    
    # 8. Verbeter contrast
    if settings.get('contrast_enhancement', 1.0) != 1.0:
        result = enhance_heatmap_contrast(
            result,
            contrast_factor=settings.get('contrast_enhancement', 1.2),
            adaptive=True,
            preserve_range=True
        )
    
    # 9. Pas transparantie effecten toe
    if settings.get('transparency_levels'):
        result = apply_heatmap_transparency(
            result,
            transparency_levels=settings.get('transparency_levels'),
            intensity_based=True
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


def apply_scientific_color_mapping(intensity_data, color_scheme=None, statistical_threshold=True):
    """
    Past wetenschappelijk accurate kleurmapping toe zoals in fMRI software.
    
    Args:
        intensity_data (numpy.ndarray): 2D array met intensiteitswaarden
        color_scheme (str): Naam van het kleurschema
        statistical_threshold (bool): Pas statistische thresholding toe
        
    Returns:
        PIL.Image: Wetenschappelijk accurate heatmap
    """
    if statistical_threshold:
        # Pas statistische thresholding toe
        from utils.color_utils import apply_statistical_thresholding
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
        zscore_mapping=FMRI_REALISM.get('zscore_mapping', True)
    )
    
    return heatmap


def enhance_depth_perception(image, depth_factor=0.3, method='gradient'):
    """
    Verbetert diepte perceptie voor 3D-achtige heatmap effecten.
    
    Args:
        image (PIL.Image): Afbeelding om diepte aan toe te voegen
        depth_factor (float): Sterkte van diepte effect (0.0-1.0)
        method (str): Methode voor diepte ('gradient', 'shadow', 'highlight')
        
    Returns:
        PIL.Image: Afbeelding met verbeterde diepte perceptie
    """
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
    enhancer = ImageEnhance.Brightness(highlight)
    highlight = enhancer.enhance(1.3)
    
    # Verminder opacity van highlight
    alpha = highlight.split()[-1]
    alpha_array = np.array(alpha) * depth_factor
    highlight.putalpha(Image.fromarray(alpha_array.astype(np.uint8)))
    
    # Combineer met origineel
    result = Image.alpha_composite(image, highlight)
    
    return result