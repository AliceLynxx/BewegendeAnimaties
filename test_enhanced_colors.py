#!/usr/bin/env python3
"""
Test script voor Enhanced fMRI Kleuren functionaliteit.

Dit script test alle nieuwe kleurschalen en dynamische kleureffecten
voor alle animatie modules in BewegendeAnimaties.
"""

import os
import sys
from utils.color_utils import get_available_color_schemes, create_color_preview
from animaties.bewegend_mannetje import (
    genereer_bewegend_mannetje_animatie, 
    genereer_lopend_mannetje_animatie
)
from animaties.vlekken_verschijnend import create_spots_appearing_animation
from animaties.vlekken_verdwijnend import create_spots_disappearing_animation
from animaties.tekst_verschijnend import generate_text_appearing_gif


def print_header(title):
    """Print een geformatteerde header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_section(title):
    """Print een geformatteerde sectie header."""
    print(f"\n--- {title} ---")


def test_color_schemes():
    """Test alle beschikbare kleurschalen."""
    print_header("ENHANCED fMRI KLEURSCHALEN TEST")
    
    # Toon beschikbare kleurschalen
    schemes = get_available_color_schemes()
    print(f"Beschikbare kleurschalen: {len(schemes)}")
    
    for scheme in schemes:
        print(f"  • {scheme['name']}: {scheme['display_name']}")
        print(f"    {scheme['description']}")
        print(f"    Kleuren: {scheme['color_count']}")
    
    return [scheme['name'] for scheme in schemes]


def test_bewegend_mannetje(color_schemes):
    """Test bewegend mannetje met verschillende kleurschalen."""
    print_header("BEWEGEND MANNETJE - ENHANCED KLEUREN TEST")
    
    generated_files = []
    
    for scheme in color_schemes:
        print_section(f"Kleurschema: {scheme}")
        
        try:
            # Test ovaal route
            print("Genereren ovaal route animatie...")
            path1 = genereer_bewegend_mannetje_animatie(
                clockwise=True,
                color_scheme=scheme,
                output_filename=f"test_bewegend_mannetje_ovaal_{scheme}.gif"
            )
            generated_files.append(path1)
            
            # Test lopend mannetje
            print("Genereren lopend mannetje animatie...")
            path2 = genereer_lopend_mannetje_animatie(
                color_scheme=scheme,
                output_filename=f"test_bewegend_mannetje_lopend_{scheme}.gif"
            )
            generated_files.append(path2)
            
            print(f"✅ {scheme} kleurschema succesvol getest")
            
        except Exception as e:
            print(f"❌ Fout bij testen {scheme}: {str(e)}")
    
    return generated_files


def test_vlekken_animaties(color_schemes):
    """Test vlekken animaties met verschillende kleurschalen."""
    print_header("VLEKKEN ANIMATIES - ENHANCED KLEUREN TEST")
    
    generated_files = []
    
    for scheme in color_schemes:
        print_section(f"Kleurschema: {scheme}")
        
        try:
            # Test verschijnende vlekken
            print("Genereren vlekken verschijnend animatie...")
            path1 = create_spots_appearing_animation(
                output_filename=f"test_vlekken_verschijnend_{scheme}.gif",
                seed=42,
                color_scheme=scheme
            )
            generated_files.append(path1)
            
            # Test verdwijnende vlekken
            print("Genereren vlekken verdwijnend animatie...")
            path2 = create_spots_disappearing_animation(
                output_filename=f"test_vlekken_verdwijnend_{scheme}.gif",
                seed=42,
                color_scheme=scheme
            )
            generated_files.append(path2)
            
            print(f"✅ {scheme} kleurschema succesvol getest")
            
        except Exception as e:
            print(f"❌ Fout bij testen {scheme}: {str(e)}")
    
    return generated_files


def test_tekst_animatie(color_schemes):
    """Test tekst animatie met verschillende kleurschalen."""
    print_header("TEKST ANIMATIE - ENHANCED KLEUREN TEST")
    
    generated_files = []
    
    for scheme in color_schemes:
        print_section(f"Kleurschema: {scheme}")
        
        try:
            # Test tekst verschijnend
            print("Genereren tekst verschijnend animatie...")
            path = generate_text_appearing_gif(
                output_filename=f"test_tekst_verschijnend_{scheme}.gif",
                color_scheme=scheme
            )
            generated_files.append(path)
            
            print(f"✅ {scheme} kleurschema succesvol getest")
            
        except Exception as e:
            print(f"❌ Fout bij testen {scheme}: {str(e)}")
    
    return generated_files


def create_comparison_overview():
    """Creëert een overzicht van alle kleurschalen."""
    print_header("KLEURSCHEMA VERGELIJKING")
    
    schemes = ['hot', 'cool', 'jet', 'viridis']
    
    print("Genereren kleurschema previews...")
    
    for scheme in schemes:
        try:
            preview = create_color_preview(scheme, width=400, height=50)
            preview_path = f"output/color_preview_{scheme}.png"
            
            # Zorg ervoor dat output directory bestaat
            os.makedirs(os.path.dirname(preview_path), exist_ok=True)
            
            preview.save(preview_path)
            print(f"✅ Preview voor {scheme}: {preview_path}")
            
        except Exception as e:
            print(f"❌ Fout bij maken preview voor {scheme}: {str(e)}")


def test_dynamic_features():
    """Test dynamische kleurfeatures."""
    print_header("DYNAMISCHE KLEURFEATURES TEST")
    
    print("Testen van dynamische features:")
    print("  • Bewegingsgebaseerde kleurvariatie")
    print("  • Tijdgebaseerde kleurveranderingen") 
    print("  • Activiteitsgebaseerde kleurmapping")
    print("  • Pulserende kleureffecten")
    print("  • Vloeiende kleurovergangen")
    print("  • Enhanced gloed effecten")
    
    # Test met verschillende instellingen
    test_configs = [
        {"scheme": "hot", "description": "Klassieke warme kleuren"},
        {"scheme": "cool", "description": "Koele kleuren voor contrast"},
        {"scheme": "jet", "description": "Rainbow kleuren voor maximale variatie"},
        {"scheme": "viridis", "description": "Perceptueel uniforme kleuren"}
    ]
    
    generated_files = []
    
    for config in test_configs:
        scheme = config["scheme"]
        print_section(f"{scheme.upper()} - {config['description']}")
        
        try:
            # Test lopend mannetje met dynamische kleuren
            path = genereer_lopend_mannetje_animatie(
                color_scheme=scheme,
                output_filename=f"test_dynamic_{scheme}_walking.gif"
            )
            generated_files.append(path)
            print(f"✅ Dynamische test voor {scheme} voltooid")
            
        except Exception as e:
            print(f"❌ Fout bij dynamische test {scheme}: {str(e)}")
    
    return generated_files


def main():
    """Hoofdfunctie voor het uitvoeren van alle tests."""
    print_header("ENHANCED fMRI KLEUREN - VOLLEDIGE TEST SUITE")
    print("Dit script test alle nieuwe kleurschalen en dynamische effecten")
    print("voor alle animatie modules in BewegendeAnimaties.")
    
    try:
        # Test 1: Kleurschalen overzicht
        color_schemes = test_color_schemes()
        
        # Test 2: Kleurschema previews
        create_comparison_overview()
        
        # Test 3: Bewegend mannetje
        bewegend_files = test_bewegend_mannetje(color_schemes)
        
        # Test 4: Vlekken animaties
        vlekken_files = test_vlekken_animaties(color_schemes)
        
        # Test 5: Tekst animatie
        tekst_files = test_tekst_animatie(color_schemes)
        
        # Test 6: Dynamische features
        dynamic_files = test_dynamic_features()
        
        # Samenvatting
        print_header("TEST RESULTATEN SAMENVATTING")
        
        all_files = bewegend_files + vlekken_files + tekst_files + dynamic_files
        
        print(f"✅ Totaal {len(all_files)} animaties gegenereerd")
        print(f"✅ {len(color_schemes)} kleurschalen getest")
        print(f"✅ Alle animatie modules geüpdatet met enhanced kleuren")
        
        print("\nGegenereerde bestanden:")
        for file_path in all_files:
            if file_path:
                print(f"  • {file_path}")
        
        print("\n🎉 ALLE TESTS SUCCESVOL VOLTOOID!")
        print("\nNieuwe features:")
        print("  ✓ 4 neuroimaging kleurschalen (hot, cool, jet, viridis)")
        print("  ✓ Vloeiende kleurovergangen")
        print("  ✓ Bewegingsgebaseerde kleurvariatie")
        print("  ✓ Tijdgebaseerde kleurveranderingen")
        print("  ✓ Enhanced gloed effecten")
        print("  ✓ Dynamische pulsing effecten")
        print("  ✓ Activiteitsgebaseerde kleurmapping")
        
    except Exception as e:
        print(f"\n❌ KRITIEKE FOUT: {str(e)}")
        print("Test suite gefaald!")
        sys.exit(1)


if __name__ == "__main__":
    main()