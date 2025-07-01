#!/usr/bin/env python3
"""
Demo script voor het realistisch lopend mannetje.

Dit script demonstreert de nieuwe functionaliteit van het realistische lopende mannetje
met vrije beweging binnen het ovaal, verschillende poses en natuurlijke bewegingspatronen.
"""

import os
import sys
from animaties.bewegend_mannetje import (
    genereer_lopend_mannetje_animatie,
    genereer_bewegend_mannetje_animatie,
    create_demo_variations
)
from config.constants import OVAL_CENTER, OUTPUT_DIR


def ensure_output_directory():
    """Zorgt ervoor dat de output directory bestaat."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Output directory aangemaakt: {OUTPUT_DIR}")


def demo_basic_walking_figure():
    """Demonstreert de basis realistisch lopend mannetje animatie."""
    print("\n" + "="*60)
    print("DEMO: Basis Realistisch Lopend Mannetje")
    print("="*60)
    
    try:
        output_path = genereer_lopend_mannetje_animatie(
            output_filename="demo_basis_lopend_mannetje.gif"
        )
        print(f"✅ Basis demo succesvol gegenereerd: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Fout bij basis demo: {str(e)}")
        return None


def demo_walking_with_smooth_interpolation():
    """Demonstreert lopend mannetje met smooth interpolatie."""
    print("\n" + "="*60)
    print("DEMO: Lopend Mannetje met Smooth Interpolatie")
    print("="*60)
    
    try:
        output_path = genereer_lopend_mannetje_animatie(
            smooth_interpolation=True,
            output_filename="demo_lopend_mannetje_smooth.gif"
        )
        print(f"✅ Smooth interpolatie demo succesvol gegenereerd: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Fout bij smooth interpolatie demo: {str(e)}")
        return None


def demo_walking_from_center():
    """Demonstreert lopend mannetje startend vanaf het centrum."""
    print("\n" + "="*60)
    print("DEMO: Lopend Mannetje vanaf Centrum")
    print("="*60)
    
    try:
        output_path = genereer_lopend_mannetje_animatie(
            start_position=OVAL_CENTER,
            output_filename="demo_lopend_mannetje_centrum.gif"
        )
        print(f"✅ Centrum start demo succesvol gegenereerd: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Fout bij centrum start demo: {str(e)}")
        return None


def demo_comparison_old_vs_new():
    """Vergelijkt de oude cirkel animatie met de nieuwe lopende figuur."""
    print("\n" + "="*60)
    print("DEMO: Vergelijking Oude vs Nieuwe Animatie")
    print("="*60)
    
    results = []
    
    # Oude cirkel animatie
    try:
        print("\nGenereren oude cirkel animatie...")
        old_path = genereer_bewegend_mannetje_animatie(
            output_filename="demo_oude_cirkel_animatie.gif"
        )
        print(f"✅ Oude animatie gegenereerd: {old_path}")
        results.append(old_path)
    except Exception as e:
        print(f"❌ Fout bij oude animatie: {str(e)}")
    
    # Nieuwe lopende figuur animatie
    try:
        print("\nGenereren nieuwe lopende figuur animatie...")
        new_path = genereer_lopend_mannetje_animatie(
            output_filename="demo_nieuwe_lopende_figuur.gif"
        )
        print(f"✅ Nieuwe animatie gegenereerd: {new_path}")
        results.append(new_path)
    except Exception as e:
        print(f"❌ Fout bij nieuwe animatie: {str(e)}")
    
    return results


def demo_all_variations():
    """Genereert alle demo variaties."""
    print("\n" + "="*60)
    print("DEMO: Alle Variaties Genereren")
    print("="*60)
    
    try:
        variations = create_demo_variations()
        print(f"\n✅ Alle variaties succesvol gegenereerd:")
        for i, path in enumerate(variations, 1):
            print(f"   {i}. {path}")
        return variations
    except Exception as e:
        print(f"❌ Fout bij genereren alle variaties: {str(e)}")
        return []


def print_usage_info():
    """Print informatie over het gebruik van de demo."""
    print("\n" + "="*60)
    print("GEBRUIKSINFORMATIE")
    print("="*60)
    print("""
Dit demo script toont de nieuwe functionaliteit van het realistisch lopend mannetje:

NIEUWE FEATURES:
• Realistisch stick figure mannetje (hoofd, lichaam, armen, benen)
• 6 verschillende poses voor natuurlijke loop-cyclus
• Vrije beweging binnen ovaal met random walk algoritme
• Natuurlijke richtingsveranderingen en bounce detection
• Behoud van fMRI kleurenschema en gloed effecten

VERGELIJKING MET ORIGINEEL:
• Origineel: Abstracte cirkel die vaste ovaalroute volgt
• Nieuw: Herkenbaar mannetje dat vrij beweegt binnen ovaal

CONFIGURATIE:
Alle instellingen kunnen aangepast worden in config/constants.py:
• WALKING_FIGURE_SIZE: Grootte van het mannetje
• WALKING_SPEED: Bewegingssnelheid
• WALKING_DIRECTION_CHANGE_CHANCE: Kans op richtingsverandering
• WALKING_POSE_CHANGE_SPEED: Snelheid van pose verandering

OUTPUT:
Alle gegenereerde GIF bestanden worden opgeslagen in de 'output' directory.
    """)


def main():
    """Hoofdfunctie die alle demo's uitvoert."""
    print("🧠 DEMO: Realistisch Lopend Mannetje - BewegendeAnimaties")
    print("=" * 80)
    
    # Zorg ervoor dat output directory bestaat
    ensure_output_directory()
    
    # Print gebruiksinformatie
    print_usage_info()
    
    # Vraag gebruiker welke demo's uit te voeren
    print("\n" + "="*60)
    print("DEMO OPTIES")
    print("="*60)
    print("1. Basis lopend mannetje")
    print("2. Lopend mannetje met smooth interpolatie")
    print("3. Lopend mannetje vanaf centrum")
    print("4. Vergelijking oude vs nieuwe animatie")
    print("5. Alle variaties genereren")
    print("6. Alles uitvoeren")
    print("0. Afsluiten")
    
    try:
        choice = input("\nKies een optie (0-6): ").strip()
        
        if choice == "0":
            print("Demo afgesloten.")
            return
        elif choice == "1":
            demo_basic_walking_figure()
        elif choice == "2":
            demo_walking_with_smooth_interpolation()
        elif choice == "3":
            demo_walking_from_center()
        elif choice == "4":
            demo_comparison_old_vs_new()
        elif choice == "5":
            demo_all_variations()
        elif choice == "6":
            print("\n🚀 Alle demo's uitvoeren...")
            demo_basic_walking_figure()
            demo_walking_with_smooth_interpolation()
            demo_walking_from_center()
            demo_comparison_old_vs_new()
            demo_all_variations()
        else:
            print("❌ Ongeldige keuze. Probeer opnieuw.")
            return main()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo onderbroken door gebruiker.")
    except Exception as e:
        print(f"\n❌ Onverwachte fout: {str(e)}")
    
    print("\n" + "="*60)
    print("DEMO VOLTOOID")
    print("="*60)
    print(f"Controleer de '{OUTPUT_DIR}' directory voor gegenereerde GIF bestanden.")
    print("Bedankt voor het testen van de realistisch lopend mannetje functionaliteit!")


if __name__ == "__main__":
    main()