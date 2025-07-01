"""
Animaties module voor BewegendeAnimaties fMRI-stijl GIF generator.

Deze module bevat verschillende animatie implementaties voor het visualiseren
van hersenactiviteit met bewegende elementen.

Beschikbare animaties:
- bewegend_mannetje: Figuur dat een ovaalroute volgt binnen hersenregio
- (Toekomstige animaties worden hier toegevoegd)
"""

from .bewegend_mannetje import (
    genereer_bewegend_mannetje_animatie,
    create_demo_variations,
    calculate_oval_position,
    create_moving_figure
)

__all__ = [
    'genereer_bewegend_mannetje_animatie',
    'create_demo_variations', 
    'calculate_oval_position',
    'create_moving_figure'
]

__version__ = '1.0.0'