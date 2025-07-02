"""
Afbeelding utilities voor BewegendeAnimaties.
Bevat functies voor achtergrond laden, ovaal masker generatie, compositing en fMRI-realisme.
"""

import os
import math
import random
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
from config.constants import (
    BACKGROUND_IMAGE_PATH, BACKGROUND_DEFAULT_SIZE, 
    OVAL_CENTER, OVAL_WIDTH, OVAL_HEIGHT, FMRI_REALISM
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
        
    Raises:
        ValueError: Als afbeelding en masker verschillende afmetingen hebben
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Controleer of afmetingen overeenkomen
    if image.size != mask.size:
        raise ValueError(
            f"Afbeelding en masker hebben verschillende afmetingen: "
            f"afbeelding {image.size}, masker {mask.size}. "
            f"Beide moeten dezelfde afmetingen hebben voor masking."
        )
    
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


# ===== NIEUWE fMRI REALISME FUNCTIES =====

def render_voxel_texture(image, voxel_size=2, opacity=0.3, style='grid'):
    """
    Rendert voxel-achtige textuur voor realistische fMRI look.
    
    Args:
        image (PIL.Image): Afbeelding om textuur aan toe te voegen
        voxel_size (int): Grootte van voxels in pixels
        opacity (float): Transparantie van voxel effect (0.0-1.0)
        style (str): Stijl van voxel effect ('grid', 'blocks', 'subtle')
        
    Returns:
        PIL.Image: Afbeelding met voxel textuur
    """
    if not FMRI_REALISM.get('voxel_enabled', True) or voxel_size <= 1:
        return image
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    width, height = image.size
    
    if style == 'grid':
        # Grid-stijl voxel textuur
        return _render_grid_voxels(image, voxel_size, opacity)
    elif style == 'blocks':
        # Block-stijl voxel textuur
        return _render_block_voxels(image, voxel_size, opacity)
    elif style == 'subtle':
        # Subtiele voxel textuur
        return _render_subtle_voxels(image, voxel_size, opacity)
    else:
        return _render_grid_voxels(image, voxel_size, opacity)


def _render_grid_voxels(image, voxel_size, opacity):
    """Rendert grid-stijl voxel textuur."""
    width, height = image.size
    voxel_overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(voxel_overlay)
    
    grid_color = (128, 128, 128, int(255 * opacity))
    
    # Verticale lijnen
    for x in range(0, width, voxel_size):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    
    # Horizontale lijnen
    for y in range(0, height, voxel_size):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    
    return Image.alpha_composite(image, voxel_overlay)


def _render_block_voxels(image, voxel_size, opacity):
    """Rendert block-stijl voxel textuur."""
    width, height = image.size
    
    # Maak pixelated versie
    small_image = image.resize(
        (width // voxel_size, height // voxel_size), 
        Image.NEAREST
    )
    pixelated = small_image.resize((width, height), Image.NEAREST)
    
    # Blend met origineel
    blend_factor = 1.0 - opacity
    result = Image.blend(pixelated.convert('RGB'), image.convert('RGB'), blend_factor)
    
    # Behoud alpha channel
    if image.mode == 'RGBA':
        result.putalpha(image.split()[-1])
    
    return result.convert('RGBA')


def _render_subtle_voxels(image, voxel_size, opacity):
    """Rendert subtiele voxel textuur."""
    # Combinatie van grid en lichte pixelation
    grid_result = _render_grid_voxels(image, voxel_size, opacity * 0.5)
    block_result = _render_block_voxels(grid_result, voxel_size, opacity * 0.3)
    
    return block_result


def apply_spatial_smoothing_image(image, kernel_size=3, preserve_edges=True):
    """
    Past spatial smoothing toe op een PIL Image.
    
    Args:
        image (PIL.Image): Afbeelding om te smoothen
        kernel_size (int): Grootte van smoothing kernel
        preserve_edges (bool): Behoud randen tijdens smoothing
        
    Returns:
        PIL.Image: Gesmoothde afbeelding
    """
    if not FMRI_REALISM.get('smoothing_enabled', True) or kernel_size <= 1:
        return image
    
    if preserve_edges:
        # Edge-preserving smoothing
        return _apply_edge_preserving_smooth(image, kernel_size)
    else:
        # Standaard Gaussische blur
        sigma = kernel_size / 3.0
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def _apply_edge_preserving_smooth(image, kernel_size):
    """Past edge-preserving smoothing toe."""
    # Vereenvoudigde edge-preserving filter
    # In praktijk zou je een bilateral filter gebruiken
    
    # Detecteer randen
    edges = image.filter(ImageFilter.FIND_EDGES)
    
    # Pas Gaussische blur toe
    sigma = kernel_size / 3.0
    smoothed = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    # Blend gebaseerd op rand sterkte
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    if smoothed.mode != 'RGBA':
        smoothed = smoothed.convert('RGBA')
    
    # Eenvoudige blending - in praktijk complexer
    result = Image.blend(smoothed.convert('RGB'), image.convert('RGB'), 0.3)
    
    if image.mode == 'RGBA':
        result.putalpha(image.split()[-1])
    
    return result.convert('RGBA')


def enhance_edges_fmri_style(image, glow_radius=2, glow_intensity=0.4, threshold=0.2):
    """
    Verbetert randen met fMRI-stijl gloed effect.
    
    Args:
        image (PIL.Image): Afbeelding om randen te verbeteren
        glow_radius (int): Radius voor rand gloed
        glow_intensity (float): Intensiteit van rand gloed (0.0-1.0)
        threshold (float): Drempel voor rand detectie
        
    Returns:
        PIL.Image: Afbeelding met verbeterde randen
    """
    if not FMRI_REALISM.get('edge_enhancement', True):
        return image
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Detecteer randen
    edges = image.filter(ImageFilter.FIND_EDGES)
    
    # Maak gloed effect voor randen
    edge_glow = edges.filter(ImageFilter.GaussianBlur(radius=glow_radius))
    
    # Verhoog intensiteit van gloed
    enhancer = ImageEnhance.Brightness(edge_glow)
    edge_glow = enhancer.enhance(1.0 + glow_intensity)
    
    # Combineer met origineel
    result = Image.alpha_composite(image, edge_glow)
    
    return result


def render_activation_clusters(image, cluster_data, colors, min_size=5):
    """
    Rendert activatie clusters met realistische fMRI kleuren.
    
    Args:
        image (PIL.Image): Basis afbeelding
        cluster_data (numpy.ndarray): 2D array met cluster labels
        colors (list): Lijst van kleuren voor clusters
        min_size (int): Minimale cluster grootte om te renderen
        
    Returns:
        PIL.Image: Afbeelding met gerenderde clusters
    """
    if not FMRI_REALISM.get('cluster_enabled', True):
        return image
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    width, height = image.size
    cluster_overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Render elke cluster
    unique_clusters = np.unique(cluster_data)
    unique_clusters = unique_clusters[unique_clusters > 0]  # Skip background (0)
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = cluster_data == cluster_id
        cluster_size = np.sum(cluster_mask)
        
        if cluster_size >= min_size:
            # Kies kleur voor cluster
            color = colors[i % len(colors)] if colors else (255, 140, 0)
            
            # Render cluster
            cluster_overlay = _render_single_cluster(
                cluster_overlay, cluster_mask, color
            )
    
    # Combineer met origineel
    result = Image.alpha_composite(image, cluster_overlay)
    
    return result


def _render_single_cluster(overlay, cluster_mask, color):
    """Rendert een enkele cluster op de overlay."""
    height, width = cluster_mask.shape
    
    # Converteer mask naar PIL Image
    mask_image = Image.fromarray((cluster_mask * 255).astype(np.uint8), mode='L')
    
    # Maak gekleurde versie
    colored_cluster = Image.new('RGBA', (width, height), color + (128,))
    
    # Pas mask toe
    colored_cluster.putalpha(mask_image)
    
    # Combineer met overlay
    result = Image.alpha_composite(overlay, colored_cluster)
    
    return result


def create_gradient_boundaries(image, gradient_width=5, falloff='gaussian', strength=0.7):
    """
    Creëert zachte gradiënt grenzen voor natuurlijke overgangen.
    
    Args:
        image (PIL.Image): Afbeelding om gradiënten aan toe te voegen
        gradient_width (int): Breedte van gradiënt zone (pixels)
        falloff (str): Type falloff ('linear', 'gaussian', 'exponential')
        strength (float): Sterkte van gradiënt effect (0.0-1.0)
        
    Returns:
        PIL.Image: Afbeelding met gradiënt grenzen
    """
    if not FMRI_REALISM.get('gradient_boundaries', True):
        return image
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Detecteer randen voor gradiënt plaatsing
    alpha = image.split()[-1]
    edges = alpha.filter(ImageFilter.FIND_EDGES)
    
    # Maak gradiënt masker
    if falloff == 'gaussian':
        gradient_mask = edges.filter(ImageFilter.GaussianBlur(radius=gradient_width))
    elif falloff == 'linear':
        gradient_mask = edges.filter(ImageFilter.BoxBlur(radius=gradient_width))
    else:  # exponential of fallback
        gradient_mask = edges.filter(ImageFilter.GaussianBlur(radius=gradient_width * 0.7))
    
    # Pas gradiënt toe op alpha channel
    gradient_array = np.array(gradient_mask) / 255.0
    alpha_array = np.array(alpha) / 255.0
    
    # Moduleer alpha met gradiënt
    modulated_alpha = alpha_array * (1.0 - gradient_array * strength)
    modulated_alpha = np.clip(modulated_alpha * 255, 0, 255).astype(np.uint8)
    
    # Maak nieuwe afbeelding met gemoduleerde alpha
    result = image.copy()
    result.putalpha(Image.fromarray(modulated_alpha))
    
    return result


def add_anatomical_variation(image, asymmetry=0.1, hotspots=True, gradients=True):
    """
    Voegt anatomische variatie toe voor realistische hersenactivatie.
    
    Args:
        image (PIL.Image): Afbeelding om variatie aan toe te voegen
        asymmetry (float): Asymmetrie factor (0.0-1.0)
        hotspots (bool): Voeg hotspot gebieden toe
        gradients (bool): Voeg anatomische gradiënten toe
        
    Returns:
        PIL.Image: Afbeelding met anatomische variatie
    """
    if not FMRI_REALISM.get('anatomical_variation', True):
        return image
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    width, height = image.size
    alpha = np.array(image.split()[-1]) / 255.0
    
    # Voeg asymmetrie toe
    if asymmetry > 0:
        alpha = _add_asymmetry(alpha, asymmetry)
    
    # Voeg hotspots toe
    if hotspots and FMRI_REALISM.get('anatomical_hotspots', True):
        alpha = _add_hotspots(alpha, width, height)
    
    # Voeg anatomische gradiënten toe
    if gradients and FMRI_REALISM.get('anatomical_gradients', True):
        alpha = _add_anatomical_gradients(alpha, width, height)
    
    # Maak nieuwe afbeelding
    result = image.copy()
    result.putalpha(Image.fromarray((alpha * 255).astype(np.uint8)))
    
    return result


def _add_asymmetry(alpha_array, asymmetry_factor):
    """Voegt asymmetrie toe aan activatie patroon."""
    height, width = alpha_array.shape
    
    # Maak asymmetrie masker
    asymmetry_mask = np.ones_like(alpha_array)
    
    # Linker/rechter asymmetrie
    center_x = width // 2
    for x in range(width):
        if x < center_x:
            # Linker kant
            factor = 1.0 + asymmetry_factor * (center_x - x) / center_x
        else:
            # Rechter kant
            factor = 1.0 - asymmetry_factor * (x - center_x) / center_x
        
        asymmetry_mask[:, x] *= factor
    
    # Pas asymmetrie toe
    result = alpha_array * asymmetry_mask
    
    return np.clip(result, 0.0, 1.0)


def _add_hotspots(alpha_array, width, height):
    """Voegt hotspot gebieden toe met verhoogde activatie."""
    # Genereer 2-4 willekeurige hotspots
    num_hotspots = random.randint(2, 4)
    
    for _ in range(num_hotspots):
        # Willekeurige hotspot positie
        hx = random.randint(width // 4, 3 * width // 4)
        hy = random.randint(height // 4, 3 * height // 4)
        
        # Hotspot radius
        radius = random.randint(10, 30)
        
        # Maak hotspot masker
        y_coords, x_coords = np.ogrid[:height, :width]
        distance = np.sqrt((x_coords - hx)**2 + (y_coords - hy)**2)
        hotspot_mask = np.exp(-distance**2 / (2 * radius**2))
        
        # Voeg hotspot toe
        alpha_array += hotspot_mask * 0.2
    
    return np.clip(alpha_array, 0.0, 1.0)


def _add_anatomical_gradients(alpha_array, width, height):
    """Voegt anatomische gradiënten toe."""
    # Anterior-posterior gradiënt
    ap_gradient = np.linspace(0.8, 1.2, height).reshape(-1, 1)
    ap_gradient = np.tile(ap_gradient, (1, width))
    
    # Superior-inferior gradiënt
    si_gradient = np.linspace(1.1, 0.9, width).reshape(1, -1)
    si_gradient = np.tile(si_gradient, (height, 1))
    
    # Combineer gradiënten
    combined_gradient = ap_gradient * si_gradient
    
    # Pas toe op activatie
    result = alpha_array * combined_gradient
    
    return np.clip(result, 0.0, 1.0)


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