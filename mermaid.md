flowchart TB

  %% Lanes
  subgraph PRE[Preprocessing]
    A[1. Bildladung<br/>• Validierung Eingabeformat<br/>• Größen- & Qualitätsprüfung<br/>• Helligkeits- & Sättigungsoptimierung]
    B[2. Farbraum-Vorverarbeitung<br/>• LAB-Konvertierung<br/>• CLAHE-Anwendung<br/>• HSV-Sättigungsanpassung<br/>• PIL-Konvertierung]
    C[3. Format-Normalisierung<br/>• Vorbereitung für KI-Verarbeitung<br/>• KI-Hintergrundentfernung]
    A --> B --> C
  end

  subgraph KI[KI-Verarbeitung]
    D[4. Deep Learning Modell<br/>• Präzise Objekterkennung<br/>• Alpha-Kanal-Generierung<br/>• Schatten-Artefakt-Entfernung]
  end

  subgraph POST[Postprocessing & Export]
    E[5. Transparenz-Analyse<br/>• Morphologische Operationen<br/>• Randglättung<br/>• Objektgrenzen-Ermittlung]
    F[6. Objektgeometrie<br/>• Alpha-Kanal-Auswertung<br/>• Bounding-Box-Berechnung<br/>• Zentrumsbestimmung<br/>• Zentrierung & Padding]
    G[7. Layout & Hintergrund<br/>• Quadrat-Erstellung<br/>• Objekt-Platzierung<br/>• Weißer Hintergrund]
    H[8. Finale Optimierung & Export<br/>• Denoising<br/>• Schärfung<br/>• Kontrastverstärkung<br/>• WebP-Kompression]
    E --> F --> G --> H
  end

  C --> D --> E
