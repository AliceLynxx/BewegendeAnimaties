# BewegendeAnimaties 🧠

Een Python-tool voor het genereren van fMRI-stijl GIF animaties met bewegende elementen op een hersenachtergrond.

## 📋 Overzicht

BewegendeAnimaties creëert visueel aantrekkelijke animaties die hersenactiviteit simuleren. Het project gebruikt karakteristieke fMRI-kleuren (oranje/rood/geel gloed) en genereert verschillende soorten bewegende elementen binnen een gedefinieerde hersenregio.

### 🎯 Doelgroep
- Neurowetenschappers en onderzoekers
- Educatieve instellingen 
- Presentatie- en visualisatiedoeleinden
- Wetenschappelijke communicatie

## ✨ Features

### 🏃 Bewegend Mannetje Animatie ✅
- **Ovaalroute beweging**: Figuur volgt smooth ovaalroute binnen hersenregio
- **fMRI Styling**: Karakteristieke oranje/rood/geel kleuren met gloed effecten
- **Route Variaties**: Standaard, elliptisch en spiraal patronen
- **Configureerbaar**: Snelheid, richting, grootte en effecten instelbaar
- **Smooth Interpolatie**: Extra frames voor vloeiende beweging
- **Pulsing Effecten**: Dynamische kleurveranderingen tijdens beweging

### 🔮 Geplande Animaties
- **Vlekken Verschijnend**: Uitspreidende activatie patronen
- **Vlekken Verdwijnend**: Afnemende hersenactiviteit
- **Tekst Verschijnend**: Letter-voor-letter tekst animatie

## 🚀 Installatie

### Vereisten
- Python 3.7+
- pip package manager

### Dependencies installeren
```bash
pip install -r requirements.txt
```

**Benodigde packages:**
- Pillow (PIL) - Afbeelding manipulatie
- numpy - Numerieke berekeningen

## 📖 Gebruik

### Bewegend Mannetje Animatie

#### Basis gebruik
```python
from animaties.bewegend_mannetje import genereer_bewegend_mannetje_animatie

# Genereer standaard animatie
output_path = genereer_bewegend_mannetje_animatie()
print(f"Animatie gegenereerd: {output_path}")
```

#### Geavanceerd gebruik
```python
# Aangepaste configuratie
output_path = genereer_bewegend_mannetje_animatie(
    clockwise=False,           # Linksom beweging
    route_variation=1,         # Elliptische route
    speed_multiplier=2.0,      # Dubbele snelheid
    smooth_interpolation=True, # Extra smooth frames
    output_filename="custom_animatie.gif"
)
```

#### Demo script uitvoeren
```bash
python demo_bewegend_mannetje.py
```

Dit genereert verschillende voorbeeld animaties:
- Standaard rechtsom beweging
- Linksom beweging  
- Elliptische route
- Spiraal patroon
- Snelle smooth animatie

## ⚙️ Configuratie

Alle instellingen zijn configureerbaar via `config/constants.py`:

### Animatie Instellingen
```python
ANIMATION_DURATION = 3.0      # Duur in seconden
FRAMES_PER_SECOND = 15        # FPS voor GIF
MOVING_FIGURE_SIZE = 15       # Grootte bewegend figuur
MOVING_FIGURE_SPEED = 1.0     # Snelheid multiplier
```

### Hersenregio Instellingen
```python
OVAL_CENTER = (400, 300)      # Centrum van hersenregio
OVAL_WIDTH = 200              # Breedte van ovaal
OVAL_HEIGHT = 150             # Hoogte van ovaal
```

### fMRI Kleuren
```python
FMRI_COLORS = {
    'primary': (255, 140, 0),    # Oranje
    'secondary': (255, 69, 0),   # Rood-oranje  
    'accent': (255, 215, 0),     # Goud-geel
    'highlight': (255, 255, 0)   # Geel
}
```

### Gloed Effecten
```python
GLOW_RADIUS = 10              # Radius van gloed
GLOW_INTENSITY = 0.7          # Intensiteit (0.0-1.0)
```

## 📁 Project Structuur

```
BewegendeAnimaties/
├── animaties/
│   ├── __init__.py
│   └── bewegend_mannetje.py     # ✅ Bewegend figuur animatie
├── config/
│   └── constants.py             # Configuratie instellingen
├── utils/
│   ├── color_utils.py           # fMRI kleurenschema
│   ├── gif_utils.py             # GIF generatie
│   └── image_utils.py           # Afbeelding manipulatie
├── output/                      # Gegenereerde GIF bestanden
├── demo_bewegend_mannetje.py    # Demo script
├── requirements.txt             # Dependencies
└── README.md                    # Deze documentatie
```

## 🎨 Voorbeelden

### Route Variaties
- **Standaard (0)**: Perfecte ovaal route
- **Elliptisch (1)**: Meer uitgerekte ellips
- **Spiraal (2)**: Spiraal effect met variërende radius

### Bewegingsrichtingen
- **Rechtsom**: `clockwise=True` (standaard)
- **Linksom**: `clockwise=False`

### Snelheid Opties
- **Langzaam**: `speed_multiplier=0.5`
- **Normaal**: `speed_multiplier=1.0` (standaard)
- **Snel**: `speed_multiplier=2.0`

## 🔧 Troubleshooting

### Veelvoorkomende Problemen

**Fout: "No module named 'PIL'"**
```bash
pip install Pillow
```

**Fout: "No module named 'numpy'"**
```bash
pip install numpy
```

**Lege of corrupte GIF**
- Controleer of output directory bestaat
- Zorg voor voldoende schijfruimte
- Controleer bestandspermissies

**Animatie te snel/langzaam**
- Pas `ANIMATION_DURATION` aan in constants.py
- Gebruik `speed_multiplier` parameter
- Wijzig `FRAMES_PER_SECOND` voor smoothness

## 🚧 Ontwikkeling

### Volgende Stappen
1. **Vlekken Verschijnend** - Uitspreidende activatie patronen
2. **Vlekken Verdwijnend** - Afnemende hersenactiviteit  
3. **Tekst Verschijnend** - Letter-voor-letter animatie
4. **Centraal Script** - Batch generatie van alle animaties

### Bijdragen
Dit project is in actieve ontwikkeling. Suggesties en verbeteringen zijn welkom!

## 📄 Licentie

Dit project is ontwikkeld voor educatieve en onderzoeksdoeleinden.

---

**Status**: 🟢 Bewegend Mannetje Animatie - Volledig Functioneel  
**Laatste Update**: December 2024