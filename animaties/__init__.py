"""
Animaties module voor BewegendeAnimaties fMRI-stijl GIF generator.

Deze module bevat verschillende animatie implementaties voor het visualiseren
van hersenactiviteit met bewegende elementen.

Beschikbare animaties:
- bewegend_mannetje: Figuur dat een ovaalroute volgt binnen hersenregio
- vlekken_verschijnend: Organische vlekken die geleidelijk verschijnen
- vlekken_verdwijnend: Organische vlekken die geleidelijk verdwijnen
- tekst_verschijnend: Tekst die letter voor letter zichtbaar wordt
"""

from .bewegend_mannetje import (
    genereer_bewegend_mannetje_animatie,
    create_demo_variations,
    calculate_oval_position,
    create_moving_figure
)

from .tekst_verschijnend import (
    create_text_appearing_animation,
    generate_text_appearing_gif,
    create_demo_animation as create_text_demo
)

__all__ = [
    'genereer_bewegend_mannetje_animatie',
    'create_demo_variations', 
    'calculate_oval_position',
    'create_moving_figure',
    'create_text_appearing_animation',
    'generate_text_appearing_gif',
    'create_text_demo'
]

__version__ = '1.1.0'