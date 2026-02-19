# 📝 Guide d'intégration des graphiques dans votre rapport

## 📁 Fichiers générés

Tous les graphiques sont disponibles en **2 formats** :
- **PNG (300 DPI)** : Pour Word, PowerPoint, Google Docs
- **PDF vectoriel** : Pour LaTeX, publications académiques

---

## 📊 Graphiques disponibles

### 🔴 **INDISPENSABLES**

#### 1. `fig1_error_comparison.png/pdf` ⭐⭐⭐⭐⭐
**Description** : Comparaison de l'erreur de position entre Open-Loop et Closed-Loop RL

**À mettre** : 
- Dans la section "Résultats principaux"
- En première figure de votre rapport

**Légende suggérée** :
```
Figure 1 : Comparaison de la précision de suivi de trajectoire. 
L'approche Closed-Loop avec apprentissage par renforcement réduit 
l'erreur moyenne de 70% par rapport au contrôle Open-Loop classique 
(8.6 mm vs 28.4 mm).
```

---

#### 2. `table_summary.png/pdf` ⭐⭐⭐⭐⭐
**Description** : Tableau récapitulatif des métriques de performance

**À mettre** :
- Juste après la Figure 1
- Ou dans la section "Résultats quantitatifs"

**Légende suggérée** :
```
Tableau 1 : Métriques de performance comparatives entre les approches 
Open-Loop et Closed-Loop RL. L'amélioration est calculée comme 
(1 - RL/OpenLoop) × 100%.
```

---

### 🟡 **IMPORTANTS**

#### 3. `fig2_ee_position_x.png/pdf` ⭐⭐⭐⭐
**Description** : Suivi spatial détaillé sur l'axe X

**À mettre** :
- Section "Analyse détaillée"
- Pour illustrer la qualité du suivi

**Légende suggérée** :
```
Figure 2 : Suivi de trajectoire de l'effecteur final sur l'axe X. 
La trajectoire réalisée (vert) suit fidèlement la référence désirée (bleu), 
malgré le bruit d'observation (orange) dû aux capteurs simulés.
```

---

#### 4. `fig3_ee_speed.png/pdf` ⭐⭐⭐
**Description** : Vitesse de l'effecteur final

**À mettre** :
- Section "Analyse de sécurité" ou "Caractérisation du mouvement"
- Pour montrer la fluidité

**Légende suggérée** :
```
Figure 3 : Profil de vitesse de l'effecteur final durant la trajectoire. 
La vitesse moyenne de 188 mm/s et l'absence d'oscillations hautes fréquences 
démontrent la stabilité du contrôle.
```

---

### 🟢 **OPTIONNELS** (pour rapports techniques approfondis)

#### 5. `fig4_joint_j4_strategy.png/pdf` ⭐⭐
**Description** : Stratégie RL sur l'articulation J4

**À mettre** :
- Section "Analyse approfondie" ou "Stratégie de la politique"
- En annexe si le rapport est court

**Légende suggérée** :
```
Figure 4 : Comportement de l'articulation J4 (wrist_1). En Closed-Loop, 
la politique RL applique un biais constant de +0.63° par rapport à la 
référence pour compenser les erreurs cinématiques et améliorer la 
précision de l'effecteur final.
```

---

## 📄 Structure suggérée du rapport

### **Rapport court (5-8 pages)**

```markdown
1. Introduction
2. Méthodologie
   - 2.1 Architecture RL
   - 2.2 Environnement de simulation
3. Résultats
   - 3.1 Métriques principales
     ➜ [Figure 1: fig1_error_comparison.png]
     ➜ [Tableau 1: table_summary.png]
   - 3.2 Analyse spatiale
     ➜ [Figure 2: fig2_ee_position_x.png]
4. Discussion
5. Conclusion
```

---

### **Rapport standard (10-15 pages)**

```markdown
1. Introduction
2. État de l'art
3. Méthodologie
   - 3.1 Formulation du problème
   - 3.2 Architecture RL (PPO)
   - 3.3 Environnement IsaacLab
4. Expérimentations
   - 4.1 Configuration
   - 4.2 Protocole d'évaluation
5. Résultats
   - 5.1 Performance globale
     ➜ [Figure 1: fig1_error_comparison.png]
     ➜ [Tableau 1: table_summary.png]
   - 5.2 Analyse spatiale
     ➜ [Figure 2: fig2_ee_position_x.png]
   - 5.3 Caractérisation dynamique
     ➜ [Figure 3: fig3_ee_speed.png]
6. Discussion
   - 6.1 Interprétation des résultats
   - 6.2 Stratégie de la politique apprise
     ➜ [Figure 4: fig4_joint_j4_strategy.png]
7. Conclusion et perspectives
```

---

### **Rapport technique complet (15+ pages)**

Inclure **tous les graphiques** dans l'ordre numérique.

---

## 💻 Intégration par logiciel

### **Microsoft Word**

```
1. Insertion → Images → Sélectionner le PNG
2. Clic droit → Habillage du texte → "Rapproché" ou "Aligné"
3. Clic droit → Insérer une légende
4. Format de l'image → Taille → Largeur: 16 cm (pleine page)
```

**Astuce** : Pour une qualité optimale, utilisez les fichiers PNG (300 DPI)

---

### **LaTeX**

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{report_figures/fig1_error_comparison.pdf}
    \caption{Comparaison de la précision de suivi de trajectoire...}
    \label{fig:error_comparison}
\end{figure}

Comme illustré dans la Figure~\ref{fig:error_comparison}, l'approche RL...
```

**Astuce** : Utilisez les fichiers PDF vectoriels pour une qualité parfaite

---

### **Google Docs / Google Slides**

```
1. Insertion → Image → Importer depuis l'ordinateur
2. Sélectionner le PNG
3. Redimensionner : Largeur ~15 cm
4. Insertion → Légende (via Add-on ou manuellement)
```

---

### **Markdown (pour GitHub/sites web)**

```markdown
![Figure 1 : Comparaison erreur](report_figures/fig1_error_comparison.png)
*Figure 1 : Comparaison de la précision de suivi de trajectoire.*
```

---

## 📐 Recommandations de mise en page

### **Taille dans le document**
- **Rapport A4** : Largeur 14-16 cm (marge à marge)
- **Présentation** : Largeur 20-24 cm (plein écran)
- **Article 2 colonnes** : Largeur 8 cm (1 colonne) ou 17 cm (2 colonnes)

### **Position**
- Toujours placer les figures **après** leur première mention dans le texte
- Centrer les figures
- Laisser 0.5-1 cm d'espace avant/après

### **Légendes**
- **Police** : 1-2 points plus petite que le texte principal
- **Format** : "Figure X : [Description courte]. [Détails optionnels]."
- **Longueur** : 1-3 phrases maximum

### **Références dans le texte**
```
✅ BON : "Comme le montre la Figure 1, l'approche RL..."
✅ BON : "Les résultats (Fig. 1) démontrent..."
❌ ÉVITER : "Le graphique ci-dessous..."
```

---

## 🎨 Personnalisation

Si vous voulez modifier les graphiques :

1. **Modifier le script** : `plot_for_report.py`
2. **Changer les couleurs** : Lignes 35-40, 70-75, etc.
3. **Ajuster les titres** : Lignes avec `ax.set_title(...)`
4. **Modifier la taille** : Lignes avec `figsize=(10, 4)`
5. **Régénérer** :
   ```bash
   python plot_for_report.py \
     --openloop logs_deploy/openloop_log_*.npz \
     --deploy logs_deploy/deploy_log_*.npz \
     --output report_figures
   ```

---

## ✅ Checklist avant inclusion

- [ ] Les graphiques sont lisibles (police ≥ 10pt)
- [ ] Les axes sont étiquetés avec unités
- [ ] Les légendes sont claires
- [ ] Les couleurs sont différenciables (important pour impression N&B)
- [ ] Chaque figure a une légende dans le rapport
- [ ] Les figures sont référencées dans le texte
- [ ] La résolution est suffisante (300 DPI pour PNG)

---

## 📧 Support

Si vous avez besoin de graphiques supplémentaires ou de modifications :
1. Modifier `plot_for_report.py`
2. Ou utiliser directement `plot_deploy_log.py` pour des vues brutes

---

**Bon courage pour votre rapport ! 📚✨**
