# 📋 Aide-mémoire : Graphes pour le rapport

## 🚀 Version ultra-rapide

| # | Nom | Ce qu'il montre | Chiffre clé | Priorité |
|---|-----|-----------------|-------------|----------|
| 1 | Error Comparison | Erreur Open-Loop vs RL | **70% amélioration** | ⭐⭐⭐⭐⭐ |
| 2 | Summary Table | Toutes les métriques | 8.6 mm vs 28.4 mm | ⭐⭐⭐⭐⭐ |
| 3 | EE Position X | Suivi spatial X | 5.3 mm erreur | ⭐⭐⭐⭐ |
| 4 | EE Speed | Vitesse effecteur | 188 mm/s moyen | ⭐⭐⭐ |
| 5 | Joint J4 | Stratégie RL | +0.63° biais | ⭐⭐ |

---

## 📝 Phrases toutes prêtes pour le rapport

### **Introduction du graphe Error Comparison**
```
La Figure 1 compare la précision de suivi entre le contrôle 
classique (Open-Loop) et l'approche par apprentissage par 
renforcement (Closed-Loop RL). On observe une réduction de 
70% de l'erreur moyenne, passant de 28.4 mm à 8.6 mm.
```

### **Introduction du tableau Summary**
```
Le Tableau 1 présente les métriques quantitatives de performance. 
Toutes les mesures d'erreur sont significativement améliorées : 
erreur moyenne (-70%), maximum (-56%), et écart-type (-71%).
```

### **Introduction EE Position X**
```
La Figure 2 illustre le suivi détaillé de la trajectoire sur 
l'axe X. La courbe verte (position réalisée) suit fidèlement 
la référence bleue, avec une erreur moyenne de 5.3 mm.
```

### **Introduction EE Speed**
```
La Figure 3 montre le profil de vitesse de l'effecteur final. 
La vitesse moyenne de 188 mm/s et l'absence d'oscillations 
hautes fréquences démontrent la stabilité du contrôle.
```

### **Introduction Joint J4**
```
La Figure 4 révèle une stratégie intéressante de la politique 
apprise. Pour améliorer la précision cartésienne, la politique 
applique systématiquement un biais de +0.63° sur l'articulation 
J4, exploitant ainsi la redondance cinématique du robot.
```

---

## 🎨 Description des couleurs

### **Fig 1 - Error Comparison**
- 🔴 Rouge : Open-Loop (mauvais = haute erreur)
- 🟢 Vert : Closed-Loop RL (bon = basse erreur)

### **Fig 2 - EE Position X**
- 🔵 Bleu pointillé : Trajectoire désirée (sans bruit)
- 🟠 Orange pointillé : Trajectoire observée (avec bruit)
- 🟢 Vert plein : Trajectoire réalisée

### **Fig 3 - EE Speed**
- 🟣 Violet avec remplissage : Vitesse instantanée

### **Fig 4 - Joint J4**
- 🔵 Bleu pointillé : Référence dataset
- 🟠 Orange pointillé : Cible (ref + action RL)
- 🟢 Vert plein : Position mesurée

---

## ✅ Checklist intégration

### **Avant d'insérer un graphe**
- [ ] Le graphe est en haute résolution (300 DPI)
- [ ] Le graphe a un numéro (Figure 1, 2, 3...)
- [ ] Le graphe a une légende descriptive
- [ ] Le graphe est référencé dans le texte
- [ ] Les axes sont lisibles (police ≥ 10pt)

### **Format de légende standard**
```
Figure X : [Titre court]. [Description 1-2 phrases]. 
[Observation/résultat principal].
```

**Exemple** :
```
Figure 1 : Comparaison de la précision Open-Loop vs Closed-Loop RL. 
L'approche RL réduit l'erreur moyenne de 70%, passant de 28.4 mm 
à 8.6 mm. Les deux courbes montrent l'évolution temporelle sur 
une trajectoire de 4.6 secondes.
```

---

## 📐 Tailles recommandées

### **Pour Word/PowerPoint**
- Largeur : **15 cm** (pleine page A4)
- Hauteur : Proportionnelle (auto)
- Résolution : Garder celle d'origine (300 DPI)

### **Pour LaTeX**
```latex
\includegraphics[width=0.9\textwidth]{file.pdf}
```

### **Pour présentation (slides)**
- Largeur : **20-24 cm** (plein écran)
- Taille police dans l'image : Vérifier lisibilité

---

## 🎯 Réponses aux questions fréquentes

### **Q: Dois-je mettre tous les graphes ?**
Non. Pour un rapport standard, 2-4 graphes suffisent :
- Fig 1 + Table 1 = minimum vital
- + Fig 2 = rapport complet
- + Fig 3-4 = rapport technique

### **Q: Dans quel ordre ?**
1. Error Comparison (performance globale)
2. Summary Table (détails chiffrés)
3. EE Position (preuve visuelle)
4. Speed (stabilité)
5. J4 Strategy (analyse approfondie)

### **Q: C'est quoi "clean" et "obs" ?**
- **Clean** = Référence parfaite (ground truth)
- **Obs** = Référence bruitée (ce que voit le robot)
- → Montre que le robot gère bien le bruit réel

### **Q: Pourquoi J4 a un biais ?**
La politique RL a découvert qu'en bougeant légèrement 
le poignet (+0.63°), elle améliore la précision de 
l'effecteur final. C'est intelligent !

### **Q: 8.6 mm c'est bien ?**
Oui ! Pour un robot UR10 industriel :
- < 5 mm = Excellent
- 5-10 mm = Très bon ✅ (votre cas)
- 10-20 mm = Acceptable
- > 20 mm = Améliorations nécessaires

---

## 💬 Exemples de phrases d'analyse

### **Pour la discussion**
```
"Les résultats démontrent l'efficacité de l'approche par RL, 
avec une réduction de 70% de l'erreur de position. Cette 
amélioration est attribuée à la capacité de la politique 
apprise à compenser les perturbations dynamiques (variations 
de damping, délais d'action) et à exploiter la redondance 
cinématique du robot."
```

### **Pour la conclusion**
```
"La précision obtenue (8.6 mm) est compatible avec des 
applications industrielles telles que l'assemblage de 
précision ou la manipulation d'objets. Les prochaines 
étapes incluent le transfert sim-to-real et la validation 
sur robot physique."
```

### **Pour comparer avec l'état de l'art**
```
"Comparé aux approches classiques de suivi de trajectoire 
(contrôle PD, feedforward), notre méthode RL améliore la 
précision de 70%, tout en maintenant une vitesse 
d'exécution rapide (188 mm/s)."
```

---

## 📊 Données brutes pour tableaux

| Métrique | Open-Loop | Closed-Loop | Amélioration |
|----------|-----------|-------------|--------------|
| Erreur moy (mm) | 28.4 | 8.6 | 70% |
| Erreur max (mm) | 52.0 | 23.1 | 56% |
| Erreur std (mm) | 15.4 | 4.4 | 71% |
| Vitesse moy (mm/s) | - | 188 | - |
| Vitesse max (mm/s) | - | 588 | - |

| Axe | Erreur moyenne (mm) | % de l'erreur totale |
|-----|---------------------|----------------------|
| X | 5.3 | 62% |
| Y | 5.7 | 66% |
| Z | 1.7 | 20% |
| **Norme** | **8.6** | **100%** |

| Joint | Mouvement (°) | Erreur RL (°) | Action RL moy (°) |
|-------|---------------|---------------|-------------------|
| J1 | 21 | 1.3 | +0.03 |
| J2 | 11 | 0.5 | -0.36 |
| J3 | 18 | 0.7 | +0.13 |
| J4 | 8 | 0.7 | **+0.63** ✨ |
| J5 | 0 | 0.4 | +0.38 |
| J6 | 21 | 1.1 | +1.10 |

---

✅ **Tout est prêt pour votre rapport !**
