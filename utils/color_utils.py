"""
Kleur utilities voor fMRI-stijl kleuren en gloed effecten.
Uitgebreid met neuroimaging standaard kleurschalen, vloeiende overgangen en fMRI-realisme.
"""

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import math
import random
from config.constants import (
    FMRI_COLORS, FMRI_COLOR_SCHEMES, DEFAULT_COLOR_SCHEME, COLOR_MAPPING,
    GLOW_RADIUS, GLOW_INTENSITY, ENHANCED_GLOW_RADIUS, ENHANCED_GLOW_INTENSITY,
    DYNAMIC_COLORS, FMRI_REALISM
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


# ===== NIEUWE fMRI REALISME FUNCTIES =====

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


def apply_voxel_texture(image, voxel_size=2, opacity=0.3):
    """
    Past voxel-achtige textuur toe voor realistische fMRI look.
    
    Args:
        image (PIL.Image): Afbeelding om textuur aan toe te voegen
        voxel_size (int): Grootte van voxels in pixels
        opacity (float): Transparantie van voxel raster (0.0-1.0)
        
    Returns:
        PIL.Image: Afbeelding met voxel textuur
    """
    if not FMRI_REALISM.get('voxel_enabled', True) or voxel_size <= 1:
        return image
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    width, height = image.size
    
    # Maak voxel overlay
    voxel_overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(voxel_overlay)
    
    # Teken voxel raster
    grid_color = (128, 128, 128, int(255 * opacity))
    
    # Verticale lijnen
    for x in range(0, width, voxel_size):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    
    # Horizontale lijnen
    for y in range(0, height, voxel_size):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    
    # Combineer met originele afbeelding
    result = Image.alpha_composite(image, voxel_overlay)
    
    return result


def apply_spatial_smoothing(image_array, kernel_size=3, iterations=1, preserve_edges=True):
    """
    Past spatial smoothing toe zoals gebruikt in fMRI analyse.
    
    Args:
        image_array (numpy.ndarray): 2D array met intensiteitswaarden
        kernel_size (int): Grootte van smoothing kernel
        iterations (int): Aantal smoothing iteraties
        preserve_edges (bool): Behoud randen tijdens smoothing
        
    Returns:
        numpy.ndarray: Gesmoothde array
    """
    if not FMRI_REALISM.get('smoothing_enabled', True) or kernel_size <= 1:
        return image_array
    
    smoothed = image_array.copy()
    
    for _ in range(iterations):
        if preserve_edges:
            # Edge-preserving smoothing (bilateral filter simulatie)
            smoothed = _bilateral_filter_simple(smoothed, kernel_size)
        else:
            # Standaard Gaussische smoothing
            smoothed = _gaussian_smooth(smoothed, kernel_size)
    
    return smoothed


def _gaussian_smooth(array, kernel_size):
    """Eenvoudige Gaussische smoothing."""
    from scipy import ndimage
    sigma = kernel_size / 3.0  # Standaard relatie tussen kernel size en sigma
    return ndimage.gaussian_filter(array, sigma=sigma)


def _bilateral_filter_simple(array, kernel_size):
    """Eenvoudige bilateral filter implementatie."""
    # Vereenvoudigde versie - in praktijk zou je scipy.ndimage gebruiken
    # Voor nu gebruiken we gewone Gaussische filter
    return _gaussian_smooth(array, kernel_size)


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


def detect_activation_clusters(image_array, min_size=5, connectivity=8):
    """
    Detecteert activatie clusters zoals in fMRI analyse.
    
    Args:
        image_array (numpy.ndarray): 2D array met activatiewaarden
        min_size (int): Minimale cluster grootte in pixels
        connectivity (int): Connectiviteit voor clustering (4 of 8)
        
    Returns:
        numpy.ndarray: Array met cluster labels (0 = geen cluster)
    """
    if not FMRI_REALISM.get('cluster_enabled', True):
        return np.zeros_like(image_array, dtype=int)
    
    # Simpele cluster detectie implementatie
    # In praktijk zou je scipy.ndimage.label gebruiken
    binary_mask = image_array > FMRI_REALISM.get('threshold_level', 0.3)
    
    # Voor nu: eenvoudige connected components
    clusters = np.zeros_like(image_array, dtype=int)
    cluster_id = 1
    
    height, width = image_array.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    
    def flood_fill(start_y, start_x, cluster_id):
        """Eenvoudige flood fill voor cluster detectie."""
        stack = [(start_y, start_x)]
        cluster_pixels = []
        
        while stack:
            y, x = stack.pop()
            
            if (y < 0 or y >= height or x < 0 or x >= width or 
                visited[y, x] or not binary_mask[y, x]):
                continue
            
            visited[y, x] = True
            cluster_pixels.append((y, x))
            
            # Voeg buren toe (4 of 8 connectiviteit)
            neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
            if connectivity == 8:
                neighbors.extend([(y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)])
            
            for ny, nx in neighbors:
                stack.append((ny, nx))
        
        return cluster_pixels
    
    # Vind alle clusters
    for y in range(height):
        for x in range(width):
            if binary_mask[y, x] and not visited[y, x]:
                cluster_pixels = flood_fill(y, x, cluster_id)
                
                # Alleen clusters boven minimale grootte behouden
                if len(cluster_pixels) >= min_size:
                    for py, px in cluster_pixels:
                        clusters[py, px] = cluster_id
                    cluster_id += 1
    
    return clusters


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


def create_multi_level_intensity(base_intensity, level_count=3, level_spacing=0.3):
    """
    Creëert multi-level intensiteit binnen activatiegebieden.
    
    Args:
        base_intensity (float): Basis intensiteit (0.0-1.0)
        level_count (int): Aantal intensiteitsniveaus
        level_spacing (float): Afstand tussen niveaus (0.0-1.0)
        
    Returns:
        float: Aangepaste intensiteit met levels
    """
    if not FMRI_REALISM.get('multi_level_enabled', True) or level_count <= 1:
        return base_intensity
    
    # Bereken level
    level_size = 1.0 / level_count
    current_level = int(base_intensity / level_size)
    current_level = min(current_level, level_count - 1)
    
    # Bereken intensiteit binnen level
    level_start = current_level * level_size
    level_progress = (base_intensity - level_start) / level_size
    
    # Pas level spacing toe
    if FMRI_REALISM.get('level_blending', 'smooth') == 'smooth':
        # Smooth blending tussen levels
        adjusted_intensity = level_start + level_progress * level_size * (1.0 - level_spacing)
    else:
        # Sharp levels
        adjusted_intensity = level_start + level_spacing * level_size
    
    return adjusted_intensity


def apply_baseline_fluctuation(image_array, baseline_level=0.1, variation=0.05, temporal_factor=0.0):
    """
    Past baseline fluctuaties toe zoals in echte fMRI data.
    
    Args:
        image_array (numpy.ndarray): 2D array met activatiewaarden
        baseline_level (float): Basis activatie niveau (0.0-1.0)
        variation (float): Variatie in baseline (0.0-1.0)
        temporal_factor (float): Tijdelijke factor voor fluctuaties
        
    Returns:
        numpy.ndarray: Array met baseline fluctuaties
    """
    if not FMRI_REALISM.get('baseline_temporal', True):
        return image_array
    
    height, width = image_array.shape
    
    # Genereer baseline fluctuaties
    if FMRI_REALISM.get('baseline_spatial', True):
        # Ruimtelijke variatie in baseline
        spatial_variation = np.random.normal(0, variation * 0.5, (height, width))
    else:
        spatial_variation = np.zeros((height, width))
    
    # Tijdelijke variatie
    temporal_variation = variation * math.sin(temporal_factor * 2 * math.pi)
    
    # Combineer baseline met fluctuaties
    baseline = baseline_level + spatial_variation + temporal_variation
    baseline = np.clip(baseline, 0.0, 1.0)
    
    # Voeg toe aan activatie (maar niet boven 1.0)
    result = np.clip(image_array + baseline, 0.0, 1.0)
    
    return result


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
    
    # 2. Pas spatial smoothing toe
    if settings.get('smoothing_enabled', True):
        alpha_array = apply_spatial_smoothing(
            alpha_array,
            settings.get('smoothing_kernel', 3),
            settings.get('smoothing_iterations', 1),
            settings.get('smoothing_preserve_edges', True)
        )
    
    # 3. Pas statistical thresholding toe
    if settings.get('threshold_enabled', True):
        alpha_array = apply_statistical_thresholding(
            alpha_array,
            settings.get('threshold_level', 0.3),
            settings.get('threshold_type', 'soft'),
            settings.get('threshold_fade', 0.1)
        )
    
    # 4. Detecteer clusters
    if settings.get('cluster_enabled', True):
        clusters = detect_activation_clusters(
            alpha_array,
            settings.get('cluster_min_size', 5),
            settings.get('cluster_connectivity', 8)
        )
        # Gebruik clusters om activatie te moduleren
        cluster_mask = clusters > 0
        alpha_array = alpha_array * cluster_mask
    
    # 5. Pas multi-level intensiteit toe
    if settings.get('multi_level_enabled', True):
        vectorized_multi_level = np.vectorize(lambda x: create_multi_level_intensity(
            x, 
            settings.get('level_count', 3),
            settings.get('level_spacing', 0.3)
        ))
        alpha_array = vectorized_multi_level(alpha_array)
    
    # 6. Voeg baseline fluctuaties toe
    alpha_array = apply_baseline_fluctuation(
        alpha_array,
        settings.get('baseline_level', 0.1),
        settings.get('baseline_variation', 0.05),
        temporal_factor
    )
    
    # 7. Pas temporele correlatie toe
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
    
    # 8. Pas voxel textuur toe
    if settings.get('voxel_enabled', True):
        enhanced_image = apply_voxel_texture(
            enhanced_image,
            settings.get('voxel_size', 2),
            settings.get('voxel_opacity', 0.3)
        )
    
    return enhanced_image, alpha_array


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