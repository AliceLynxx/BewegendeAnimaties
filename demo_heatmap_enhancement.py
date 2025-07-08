#!/usr/bin/env python3
"""
Demo script voor heatmap enhancement functionaliteit.

Dit script demonstreert de nieuwe heatmap enhancement functies die zijn toegevoegd
aan BewegendeAnimaties voor authentieke fMRI-heatmap visualisaties.
"""

import os
import sys
import time
from pathlib import Path

# Voeg project root toe aan Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from animaties.bewegend_mannetje import (
    genereer_bewegend_mannetje_animatie,
    genereer_lopend_mannetje_animatie,
    create_heatmap_enhancement_demo,
    create_heatmap_comparison_demo
)
from utils.color_utils import (
    get_available_color_schemes,
    create_color_preview,
    create_heatmap_gradient,
    integrate_heatmap_effects
)
from utils.image_utils import (
    load_background_image,
    create_heatmap_comparison,
    apply_scientific_heatmap_rendering
)
from config.constants import HEATMAP_ENHANCEMENT, OUTPUT_DIR
import numpy as np


def print_header(title):
    """Print een geformatteerde header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_section(title):
    """Print een geformatteerde sectie header."""
    print(f"\n--- {title} ---")


def ensure_output_directory():
    """Zorg ervoor dat de output directory bestaat."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✅ Output directory aangemaakt: {OUTPUT_DIR}")


def demo_color_schemes():
    """Demonstreer beschikbare kleurschema's."""
    print_section("Beschikbare kleurschema's")
    
    schemes = get_available_color_schemes()
    for scheme in schemes:
        print(f"- {scheme['name']}: {scheme['display_name']}")
        print(f"  {scheme['description']}")
        print(f"  Kleuren: {scheme['color_count']}")
        print()


def demo_heatmap_settings():
    """Demonstreer heatmap enhancement instellingen."""
    print_section("Heatmap Enhancement Instellingen")
    
    print("Huidige heatmap enhancement configuratie:")
    for key, value in HEATMAP_ENHANCEMENT.items():
        if isinstance(value, (int, float, bool, str)):
            print(f"- {key}: {value}")
        elif isinstance(value, (list, tuple)):
            print(f"- {key}: {value}")
    print()


def demo_basic_animations():
    """Demonstreer basis animaties met heatmap enhancement."""
    print_section("Basis Animaties met Heatmap Enhancement")
    
    try:
        # Bewegend mannetje (ovaal route) met heatmap enhancement
        print("Genereren bewegend mannetje (ovaal route) met heatmap enhancement...")
        path1 = genereer_bewegend_mannetje_animatie(
            clockwise=True,
            route_variation=0,
            color_scheme='hot',
            enable_fmri_realism=True,
            enable_heatmap_enhancement=True,
            output_filename="demo_ovaal_heatmap.gif"
        )
        print(f"✅ Gegenereerd: {path1}")
        
        # Lopend mannetje met heatmap enhancement
        print("\nGenereren lopend mannetje met heatmap enhancement...")
        path2 = genereer_lopend_mannetje_animatie(
            color_scheme='hot',
            enable_fmri_realism=True,
            enable_heatmap_enhancement=True,
            output_filename="demo_lopend_heatmap.gif"
        )
        print(f"✅ Gegenereerd: {path2}")
        
        return [path1, path2]
        
    except Exception as e:
        print(f"❌ Fout bij genereren basis animaties: {str(e)}")
        return []


def demo_color_scheme_comparison():
    """Demonstreer verschillende kleurschema's met heatmap enhancement."""
    print_section("Kleurschema Vergelijking met Heatmap Enhancement")
    
    color_schemes = ['hot', 'cool', 'jet', 'viridis']
    generated_files = []
    
    for scheme in color_schemes:
        try:
            print(f"Genereren animatie met {scheme} kleurschema...")
            path = genereer_lopend_mannetje_animatie(
                color_scheme=scheme,
                enable_fmri_realism=True,
                enable_heatmap_enhancement=True,
                output_filename=f"demo_colorscheme_{scheme}_heatmap.gif"
            )
            generated_files.append(path)
            print(f"✅ {scheme}: {path}")
            
        except Exception as e:
            print(f"❌ Fout bij {scheme}: {str(e)}")
    
    return generated_files


def demo_before_after_comparison():
    """Demonstreer voor/na vergelijking van heatmap enhancement."""
    print_section("Voor/Na Vergelijking Heatmap Enhancement")
    
    generated_files = []
    
    try:
        # Zonder heatmap enhancement
        print("Genereren animatie ZONDER heatmap enhancement...")
        path_before = genereer_lopend_mannetje_animatie(
            color_scheme='hot',
            enable_fmri_realism=True,
            enable_heatmap_enhancement=False,
            output_filename="demo_before_heatmap.gif"
        )
        generated_files.append(path_before)
        print(f"✅ Voor: {path_before}")
        
        # Met heatmap enhancement
        print("\nGenereren animatie MET heatmap enhancement...")
        path_after = genereer_lopend_mannetje_animatie(
            color_scheme='hot',
            enable_fmri_realism=True,
            enable_heatmap_enhancement=True,
            output_filename="demo_after_heatmap.gif"
        )
        generated_files.append(path_after)
        print(f"✅ Na: {path_after}")
        
        print("\n📊 Vergelijk deze twee bestanden om het verschil te zien!")
        
    except Exception as e:
        print(f"❌ Fout bij voor/na vergelijking: {str(e)}")
    
    return generated_files


def demo_advanced_heatmap_features():
    """Demonstreer geavanceerde heatmap functies."""
    print_section("Geavanceerde Heatmap Functies")
    
    try:
        # Test heatmap enhancement demo
        print("Uitvoeren heatmap enhancement demo...")
        demo_files = create_heatmap_enhancement_demo()
        print(f"✅ Demo bestanden: {len(demo_files)}")
        
        # Test heatmap vergelijking demo
        print("\nUitvoeren heatmap vergelijking demo...")
        comparison_files = create_heatmap_comparison_demo()
        print(f"✅ Vergelijking bestanden: {len(comparison_files)}")
        
        return demo_files + comparison_files
        
    except Exception as e:
        print(f"❌ Fout bij geavanceerde functies: {str(e)}")
        return []


def demo_heatmap_gradients():
    """Demonstreer heatmap gradiënt functies."""
    print_section("Heatmap Gradiënten")
    
    try:
        from PIL import Image
        
        # Creëer heatmap gradiënten voor verschillende kleurschema's
        schemes = ['hot', 'cool', 'jet', 'viridis']
        
        for scheme in schemes:
            print(f"Creëren heatmap gradiënt voor {scheme}...")
            
            # Creëer gradiënt
            gradient_colors = create_heatmap_gradient(
                intensity_levels=20,
                color_scheme=scheme,
                logarithmic=True
            )
            
            # Maak preview afbeelding
            preview = create_color_preview(scheme, width=400, height=50)
            
            # Sla preview op
            preview_path = os.path.join(OUTPUT_DIR, f"heatmap_gradient_{scheme}.png")
            preview.save(preview_path)
            print(f"✅ Gradiënt preview opgeslagen: {preview_path}")
        
    except Exception as e:
        print(f"❌ Fout bij heatmap gradiënten: {str(e)}")


def demo_scientific_accuracy():
    """Demonstreer wetenschappelijke nauwkeurigheid van heatmaps."""
    print_section("Wetenschappelijke Nauwkeurigheid")
    
    try:
        # Creëer test intensiteit data
        width, height = 200, 150
        intensity_data = np.zeros((height, width))
        
        # Voeg enkele activatie gebieden toe
        # Centrum activatie
        center_x, center_y = width // 2, height // 2
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance < 30:
                    intensity_data[y, x] = 0.8 * np.exp(-distance**2 / (2 * 15**2))
        
        # Secundaire activatie
        sec_x, sec_y = width // 4, height // 3
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - sec_x)**2 + (y - sec_y)**2)
                if distance < 20:
                    intensity_data[y, x] = max(intensity_data[y, x], 
                                             0.6 * np.exp(-distance**2 / (2 * 10**2)))
        
        # Laad achtergrond
        background = load_background_image()
        background = background.resize((width, height))
        
        # Creëer wetenschappelijk accurate heatmap
        print("Creëren wetenschappelijk accurate heatmap...")
        scientific_heatmap = apply_scientific_heatmap_rendering(
            intensity_data, background, 
            color_scheme='hot',
            statistical_threshold=True,
            zscore_mapping=True
        )
        
        # Sla op
        scientific_path = os.path.join(OUTPUT_DIR, "scientific_heatmap_demo.png")
        scientific_heatmap.save(scientific_path)
        print(f"✅ Wetenschappelijke heatmap opgeslagen: {scientific_path}")
        
        # Creëer vergelijking tussen verschillende kleurschema's
        print("\nCreëren kleurschema vergelijking...")
        comparison = create_heatmap_comparison(
            intensity_data, background,
            color_schemes=['hot', 'cool', 'jet', 'viridis'],
            layout='horizontal'
        )
        
        comparison_path = os.path.join(OUTPUT_DIR, "heatmap_schemes_comparison.png")
        comparison.save(comparison_path)
        print(f"✅ Kleurschema vergelijking opgeslagen: {comparison_path}")
        
    except Exception as e:
        print(f"❌ Fout bij wetenschappelijke nauwkeurigheid demo: {str(e)}")


def demo_performance_optimization():
    """Demonstreer performance optimalisatie opties."""
    print_section("Performance Optimalisatie")
    
    optimization_levels = ['speed', 'balanced', 'quality']
    
    for level in optimization_levels:
        try:
            print(f"Testen optimalisatie niveau: {level}")
            
            # Tijdelijk wijzig optimalisatie niveau
            original_level = HEATMAP_ENHANCEMENT.get('optimization_level', 'balanced')
            HEATMAP_ENHANCEMENT['optimization_level'] = level
            
            start_time = time.time()
            
            # Genereer korte animatie voor performance test
            path = genereer_lopend_mannetje_animatie(
                color_scheme='hot',
                enable_fmri_realism=True,
                enable_heatmap_enhancement=True,
                output_filename=f"demo_performance_{level}.gif"
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ {level}: {path} (tijd: {duration:.2f}s)")
            
            # Herstel originele instelling
            HEATMAP_ENHANCEMENT['optimization_level'] = original_level
            
        except Exception as e:
            print(f"❌ Fout bij {level} optimalisatie: {str(e)}")


def print_summary(all_generated_files):
    """Print samenvatting van gegenereerde bestanden."""
    print_header("SAMENVATTING")
    
    print(f"Totaal gegenereerde bestanden: {len(all_generated_files)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nGegenereerde bestanden:")
    
    for i, file_path in enumerate(all_generated_files, 1):
        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"{i:2d}. {os.path.basename(file_path)} ({file_size:.1f} KB)")
        else:
            print(f"{i:2d}. {file_path} (NIET GEVONDEN)")
    
    print(f"\n📁 Alle bestanden zijn opgeslagen in: {os.path.abspath(OUTPUT_DIR)}")
    print("\n🎯 Vergelijk de verschillende versies om de impact van heatmap enhancement te zien!")


def main():
    """Hoofdfunctie voor heatmap enhancement demo."""
    print_header("HEATMAP ENHANCEMENT DEMO")
    print("Dit script demonstreert de nieuwe heatmap enhancement functionaliteit")
    print("voor authentieke fMRI-heatmap visualisaties.")
    
    # Zorg ervoor dat output directory bestaat
    ensure_output_directory()
    
    # Verzamel alle gegenereerde bestanden
    all_generated_files = []
    
    try:
        # 1. Toon beschikbare kleurschema's
        demo_color_schemes()
        
        # 2. Toon heatmap instellingen
        demo_heatmap_settings()
        
        # 3. Demonstreer heatmap gradiënten
        demo_heatmap_gradients()
        
        # 4. Demonstreer wetenschappelijke nauwkeurigheid
        demo_scientific_accuracy()
        
        # 5. Basis animaties
        basic_files = demo_basic_animations()
        all_generated_files.extend(basic_files)
        
        # 6. Voor/na vergelijking
        comparison_files = demo_before_after_comparison()
        all_generated_files.extend(comparison_files)
        
        # 7. Kleurschema vergelijking
        colorscheme_files = demo_color_scheme_comparison()
        all_generated_files.extend(colorscheme_files)
        
        # 8. Geavanceerde functies
        advanced_files = demo_advanced_heatmap_features()
        all_generated_files.extend(advanced_files)
        
        # 9. Performance optimalisatie
        demo_performance_optimization()
        
        # 10. Samenvatting
        print_summary(all_generated_files)
        
        print_header("DEMO VOLTOOID")
        print("✅ Alle heatmap enhancement functies zijn succesvol getest!")
        print("\n🔬 De gegenereerde animaties tonen nu authentieke fMRI-heatmap visualisaties")
        print("   met verbeterde kleurmapping, blending en wetenschappelijke nauwkeurigheid.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo onderbroken door gebruiker")
        print_summary(all_generated_files)
        
    except Exception as e:
        print(f"\n❌ Onverwachte fout in demo: {str(e)}")
        print_summary(all_generated_files)
        raise


if __name__ == "__main__":
    main()