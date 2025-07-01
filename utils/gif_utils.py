"""
GIF utilities voor het creëren en optimaliseren van animaties.
"""

import os
from PIL import Image
from config.constants import (
    OUTPUT_DIR, FRAMES_PER_SECOND, GIF_OPTIMIZE, GIF_LOOP,
    ANIMATION_DURATION, TOTAL_FRAMES
)


def ensure_output_directory():
    """
    Zorgt ervoor dat de output directory bestaat.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def create_gif_from_frames(frames, filename, duration=None, optimize=None, loop=None):
    """
    Creëert een GIF bestand uit een lijst van frames.
    
    Args:
        frames (list): Lijst van PIL.Image objecten
        filename (str): Naam van het output bestand
        duration (float): Duur per frame in seconden (standaard uit constants)
        optimize (bool): Of GIF geoptimaliseerd moet worden (standaard uit constants)
        loop (int): Loop count (0 = oneindige loop, standaard uit constants)
        
    Returns:
        str: Pad naar het gemaakte GIF bestand
    """
    if not frames:
        raise ValueError("Geen frames om GIF van te maken")
    
    # Gebruik standaard waarden als niet opgegeven
    if duration is None:
        duration = 1.0 / FRAMES_PER_SECOND
    if optimize is None:
        optimize = GIF_OPTIMIZE
    if loop is None:
        loop = GIF_LOOP
    
    # Zorg dat output directory bestaat
    ensure_output_directory()
    
    # Volledig pad naar output bestand
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    # Converteer alle frames naar RGB voor GIF compatibiliteit
    rgb_frames = []
    for frame in frames:
        if frame.mode == 'RGBA':
            # Converteer RGBA naar RGB met witte achtergrond
            rgb_frame = Image.new('RGB', frame.size, (255, 255, 255))
            rgb_frame.paste(frame, mask=frame.split()[-1])  # Gebruik alpha als mask
            rgb_frames.append(rgb_frame)
        elif frame.mode != 'RGB':
            rgb_frames.append(frame.convert('RGB'))
        else:
            rgb_frames.append(frame)
    
    # Maak GIF
    rgb_frames[0].save(
        output_path,
        save_all=True,
        append_images=rgb_frames[1:],
        duration=int(duration * 1000),  # PIL verwacht milliseconden
        loop=loop,
        optimize=optimize
    )
    
    return output_path


def create_test_gif():
    """
    Creëert een test GIF om de functionaliteit te testen.
    
    Returns:
        str: Pad naar test GIF
    """
    from utils.image_utils import load_background_image, create_oval_mask, apply_oval_mask
    from utils.color_utils import create_colored_circle, get_fmri_color
    
    # Laad achtergrond
    background = load_background_image()
    mask = create_oval_mask(background.size)
    
    frames = []
    
    # Maak test frames met bewegende cirkel
    for i in range(TOTAL_FRAMES):
        frame = background.copy()
        
        # Bereken positie voor bewegende cirkel
        progress = i / TOTAL_FRAMES
        x = int(300 + 100 * progress)  # Beweeg van links naar rechts
        y = 300
        
        # Maak gekleurde cirkel
        circle = create_colored_circle(20, get_fmri_color('primary'))
        
        # Plak cirkel op frame
        frame.paste(circle, (x - 10, y - 10), circle)
        
        # Pas ovaal masker toe
        masked_overlay = apply_oval_mask(frame, mask)
        
        # Combineer met originele achtergrond
        final_frame = background.copy()
        final_frame.paste(masked_overlay, (0, 0), masked_overlay)
        
        frames.append(final_frame)
    
    # Maak test GIF
    return create_gif_from_frames(frames, "test_animation.gif")


def optimize_gif(input_path, output_path=None, max_colors=256):
    """
    Optimaliseert een bestaand GIF bestand.
    
    Args:
        input_path (str): Pad naar input GIF
        output_path (str): Pad voor output GIF (standaard overschrijft input)
        max_colors (int): Maximum aantal kleuren in palet
        
    Returns:
        str: Pad naar geoptimaliseerd GIF
    """
    if output_path is None:
        output_path = input_path
    
    # Open GIF
    with Image.open(input_path) as gif:
        frames = []
        
        # Extraheer alle frames
        try:
            while True:
                frame = gif.copy()
                # Reduceer kleurenpalet
                if frame.mode != 'P':
                    frame = frame.quantize(colors=max_colors)
                frames.append(frame)
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass  # Einde van frames bereikt
    
    # Sla geoptimaliseerde versie op
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=gif.info.get('duration', int(1000 / FRAMES_PER_SECOND)),
            loop=gif.info.get('loop', 0),
            optimize=True
        )
    
    return output_path


def get_frame_count(gif_path):
    """
    Geeft het aantal frames in een GIF terug.
    
    Args:
        gif_path (str): Pad naar GIF bestand
        
    Returns:
        int: Aantal frames
    """
    with Image.open(gif_path) as gif:
        frame_count = 0
        try:
            while True:
                gif.seek(frame_count)
                frame_count += 1
        except EOFError:
            pass
    
    return frame_count


def get_animation_info():
    """
    Geeft informatie over animatie instellingen terug.
    
    Returns:
        dict: Dictionary met animatie informatie
    """
    return {
        'duration': ANIMATION_DURATION,
        'fps': FRAMES_PER_SECOND,
        'total_frames': TOTAL_FRAMES,
        'frame_duration_ms': int(1000 / FRAMES_PER_SECOND),
        'output_dir': OUTPUT_DIR,
        'optimize': GIF_OPTIMIZE,
        'loop': GIF_LOOP
    }