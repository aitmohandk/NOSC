# TODO — NOSC OSSE 3D multivariable

Suivi des travaux en suspens, issus de la revue de code et des discussions
d'architecture. Ordre indicatif par priorité au sein de chaque section.
(Voir CHANGES_OSSE.md pour ce qui est déjà fait.)

## 0. Mise en route (avant tout run long — bloquant)
- [ ] Exécuter `tests/test_argo_and_pseudo_obs.py` dans l'env conda du projet
      (les tests numpy passent déjà : 11/11).
- [ ] `python main.py xp=osse3d_gs21_multivar_unet --cfg job --resolve`
      (composition Hydra seule, coût nul — attrape les erreurs de config).
- [ ] Préparer les données : GLORYS régional surface + multi-profondeur,
      bathy, ARGO virtuels (`contrib.argo.virtual --depth-indices ...`),
      EOF si ablation modes (`contrib.multivar.vertical_modes`).
- [ ] Smoke test GPU (2 epochs, 5 batches) sur la config maîtresse PUIS sur
      chaque ablation (chaque ablation = un chemin de code distinct).
- [ ] Vérifier au smoke test la variance par niveau des sorties de la tête
      en modes verticaux (test d'équivalence numérique de la projection).
- [ ] Surveiller la RAM du chargement eager (~25–30 Go attendus à 21
      niveaux) ; basculer sur les variantes dask de contrib/data_loading/
      si nécessaire.

## 1. Robustesse au système d'observation
- [ ] **Masques réels 2010-2023** : brancher `l3_mask.nc`
      (altimetry_traces/2010_2023, si accessible) comme mask NetCDF daté de
      `make_pseudo_obs` — échantillonnage historique fidèle, constellation
      non stationnaire incluse ; comparer à la constellation synthétique
      (ablation réalisme d'échantillonnage).
- [ ] **Mission dropout** : à l'entraînement, retirer aléatoirement 1–2
      missions du masque de certains exemples (val/test figés). Deux voies :
      pré-générer quelques réalisations de pseudo-obs à sous-ensembles de
      missions et tirer parmi elles, ou réactiver un masquage additionnel
      léger côté augmentation. Double intérêt : robustesse opérationnelle
      + ablation OSSE classique (« que coûte la perte de Jason-3 ? »).
- [ ] Fenêtres d'activité historiques par mission dans `missions.py`
      (dates de lancement/fin : J3 2016, S3A 2016, S3B/HY-2B 2018, SARAL
      2013...) pour une constellation réaliste non stationnaire — ~20 lignes.
- [ ] Bruit d'observation ré-échantillonné à l'entraînement (augmentation),
      figé en val/test (actuellement : une seule réalisation partout).
- [ ] Intégrer SWOT : ajout à la constellation par défaut et/ou chemin
      simulateur officiel CNES/JPL (`build_masks_swot_official`, bruit
      KaRIn réel, `with_values=True`).

## 2. Réalisme des observations
- [ ] SST d'entrée masquée nuages (actuellement : vérité dense = hypothèse
      « L4 parfait ») — extension naturelle de `make_pseudo_obs`.
- [ ] Bruit instrumental + erreur de représentativité point-vs-maille sur
      les ARGO virtuels (actuellement parfaits à l'échelle de la grille).
- [ ] Canaux de présence d'observation ARGO par profondeur (analogue de
      `obs_mask` SSH) si le réseau sous-utilise les entrées in situ éparses.
- [ ] Heure de passage intra-journalière des traces (les masques sont
      journaliers — simplification standard, légèrement optimiste).

## 3. Architecture (ablations restantes)
- [ ] **Ablation GradSolver** : porter le prior variationnel multivarié
      (`multivar_costs`) vers le nouveau jeu de variables pour comparer
      UNet direct vs 4DVarNet itératif (la comparaison de l'article de
      référence).
- [ ] Variante temporelle : conv 3D (temps comme dimension), récurrence ou
      attention temporelle au goulot, au lieu du temps-en-canaux.
- [ ] Formulation générative (diffusion conditionnelle) si la baseline
      confirme le flou en profondeur attendu de la médiane conditionnelle
      — direction de recherche, pas un prérequis.

## 4. Protocole & évaluation
- [ ] Variante SLA (soustraction de la moyenne temporelle de zos) pour la
      comparabilité stricte avec l'article de référence.
- [ ] Deuxième région (régime dynamique contrasté, p.ex. gyre subtropical
      calme) pour dé-conditionner les conclusions du Gulf Stream.
- [ ] Volet OSSE→OSE : transfert aux vraies observations, validation contre
      ARGO réels (`colocate.py`) et drifters indépendants (`ose2osse/`).
- [ ] Métriques spectrales par profondeur intégrées au flux de test (le
      script `depth_profile_metrics.py` est offline pour l'instant) ;
      score de cohérence verticale des profils reconstruits.
- [ ] Score Lagrangien sur les courants reconstruits (outillage existant
      dans `metric/lagrangian/`).

## 5. Dette technique héritée
- [x] `MultivarBatchSelector` : `.cuda()` codé en dur → `default_device()`
      (fait, 3e lot ; chemin OSSE agnostique CPU/GPU). Reste : patron
      singleton fragile du sélecteur.
- [ ] Conversion pression(dbar)→profondeur(m) TEOS-10 dans le pipeline ARGO
      (~1 % d'erreur à 200 m, actuellement assimilées).
- [ ] `colocate.py` : vectoriser (iterrows lent à l'échelle ARGO).
- [ ] Nettoyage des configs dépréciées (`unet_uv_full_integration_*`) une
      fois les comparaisons faites.

## 6. Dépendances externes (versions à surveiller)
- [x] `copernicusmarine` : épinglage 1.3.3 (v1, rejeté par le backend actuel
      — erreur "Client version is not compatible") corrigé en `>=2.4,<3`
      dans env/4dvarnet-daniel.yaml. Si l'erreur revient : le service évolue
      vite, mettre à jour (`pip install -U copernicusmarine` /
      `conda update copernicusmarine`) et vérifier le guide de migration
      officiel pour d'éventuels renommages de paramètres.

- [x] `dask==2023.12.1` (pip) désynchronisé de `distributed=2024.9.0` (conda) —
      bug préexistant dans l'env du projet parent, révélé par l'ajout de la
      contrainte `copernicusmarine>=2.4` (qui exige `dask>=2024.8.1`).
      Corrigé : `dask==2024.9.0`, aligné sur `distributed` ET `dask-expr=1.1.14`
      (sortis la même semaine — la paire cohérente).
