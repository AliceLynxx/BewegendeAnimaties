#!/usr/bin/env python3
"""
Demo script voor bewegend mannetje animatie.

Dit script demonstreert het gebruik van de bewegend_mannetje module
en genereert verschillende voorbeeld animaties.
"""

import os
import sys
from animaties.bewegend_mannetje import (
    genereer_bewegend_mannetje_animatie,
    create_demo_variations
)


def print_header():
    """Print demo header."""
    print("=" * 60)
    print("🧠 BewegendeAnimaties - Bewegend Mannetje Demo")
    print("=" * 60)
    print()


def print_section(title):
    """Print sectie header."""
    print(f"\n📋 {title}")
    print("-" * 40)


def demo_basic_animation():
    """Demonstreer basis animatie."""
    print_section("Basis Animatie")
    
    try:
        output_path = genereer_bewegend_mannetje_animatie()
        print(f"✅ Basis animatie gegenereerd: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Fout bij basis animatie: {str(e)}")
        return False


def demo_custom_animations():
    """Demonstreer aangepaste animaties."""
    print_section("Aangepaste Animaties")
    
    success_count = 0
    total_count = 0
    
    # Linksom beweging
    try:
        total_count += 1
        path = genereer_bewegend_mannetje_animatie(
            clockwise=False,
            output_filename="demo_linksom.gif"
        )
        print(f"✅ Linksom animatie: {path}")
        success_count += 1
    except Exception as e:
        print(f"❌ Linksom animatie gefaald: {str(e)}")
    
    # Snelle beweging
    try:
        total_count += 1
        path = genereer_bewegend_mannetje_animatie(
            speed_multiplier=2.0,
            output_filename="demo_snel.gif"
        )
        print(f"✅ Snelle animatie: {path}")
        success_count += 1
    except Exception as e:
        print(f"❌ Snelle animatie gefaald: {str(e)}")
    
    # Elliptische route
    try:
        total_count += 1
        path = genereer_bewegend_mannetje_animatie(
            route_variation=1,
            output_filename="demo_elliptisch.gif"
        )
        print(f"✅ Elliptische animatie: {path}")
        success_count += 1
    except Exception as e:
        print(f"❌ Elliptische animatie gefaald: {str(e)}")
    
    # Smooth interpolatie
    try:
        total_count += 1
        path = genereer_bewegend_mannetje_animatie(
            smooth_interpolation=True,
            output_filename="demo_smooth.gif"
        )
        print(f"✅ Smooth animatie: {path}")
        success_count += 1
    except Exception as e:
        print(f"❌ Smooth animatie gefaald: {str(e)}")
    
    print(f"\n📊 Aangepaste animaties: {success_count}/{total_count} succesvol")
    return success_count == total_count


def demo_variations():
    """Demonstreer alle variaties."""
    print_section("Demo Variaties")
    
    try:
        variations = create_demo_variations()
        print(f"✅ {len(variations)} demo variaties gegenereerd:")
        for i, path in enumerate(variations, 1):
            print(f"   {i}. {path}")
        return True
    except Exception as e:
        print(f"❌ Demo variaties gefaald: {str(e)}")
        return False


def check_output_directory():
    """Controleer of output directory bestaat."""
    from config.constants import OUTPUT_DIR
    
    if os.path.exists(OUTPUT_DIR):
        files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.gif')]
        print(f"📁 Output directory: {OUTPUT_DIR}")
        print(f"📄 Bestaande GIF bestanden: {len(files)}")
        if files:
            for file in files[:5]:  # Toon eerste 5
                print(f"   - {file}")
            if len(files) > 5:
                print(f"   ... en {len(files) - 5} meer")
    else:
        print(f"📁 Output directory wordt aangemaakt: {OUTPUT_DIR}")


def main():
    """Hoofdfunctie voor demo."""
    print_header()
    
    # Controleer output directory
    check_output_directory()
    
    # Voer demos uit
    results = []
    
    # Basis animatie
    results.append(demo_basic_animation())
    
    # Aangepaste animaties
    results.append(demo_custom_animations())
    
    # Demo variaties
    results.append(demo_variations())
    
    # Samenvatting
    print_section("Samenvatting")
    success_count = sum(results)
    total_count = len(results)
    
    if success_count == total_count:
        print("🎉 Alle demos succesvol uitgevoerd!")
        print("\n💡 Tips:")
        print("   - Bekijk de gegenereerde GIF bestanden in de output directory")
        print("   - Experimenteer met verschillende parameters")
        print("   - Gebruik de animaties in je eigen projecten")
    else:
        print(f"⚠️  {success_count}/{total_count} demos succesvol")
        print("\n🔧 Troubleshooting:")
        print("   - Controleer of alle dependencies geïnstalleerd zijn")
        print("   - Zorg dat er voldoende schijfruimte is")
        print("   - Bekijk de error berichten voor meer details")
    
    print(f"\n📁 Controleer de output directory voor gegenereerde bestanden:")
    from config.constants import OUTPUT_DIR
    print(f"   {os.path.abspath(OUTPUT_DIR)}")
    
    return success_count == total_count


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo gestopt door gebruiker")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Onverwachte fout: {str(e)}")
        sys.exit(1)