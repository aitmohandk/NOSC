# Refonte OSSE propre — journal des modifications

Objectif : implémenter le protocole "vérité GLORYS, pseudo-observations
synthétiques" (dans la lignée d'arXiv:2512.22152), étendu à la 3D et au
multi-variable, en remplacement du protocole hybride des configs
`unet_uv_full_integration_*` (produits L4 réels mélangés à des masques
synthétiques + fuite entrées/cibles via ugos/vgos).

## Corrections de bugs

1. **`contrib/synthetic_obs/missions.py` — paramètres orbitaux faux (×2)**
   Jason-3/Sentinel-6 : 254 *passes* par cycle avaient été pris pour 254
   *orbites* → période de 56 min (physiquement impossible) et densité
   d'échantillonnage doublée. Corrigé : 127 orbites (T≈112,4 min). HY-2B :
   228 → 193 orbites (T≈104,5 min). Un garde-fou `validate_missions()`
   (appelé à l'import et par `build_masks`) refuse désormais toute période
   hors de la bande LEO plausible [90, 130] min.

2. **`contrib/synthetic_obs/orbits.py` — cycle de répétition non fermé**
   La dérive du nœud ascendant était calculée avec le jour sidéral, en
   négligeant la précession nodale J2 → erreur de fermeture ≈ 20°/cycle.
   Remplacé par la paramétrisation standard (N orbites, D jours nodaux) :
   décalage/orbite = 360·D/N, fermeture exacte par construction (testée).

3. **`contrib/synthetic_obs/sampling.py` — traces pointillées**
   720 points/orbite ≈ 56 km entre échantillons > taille de cellule à 1/4°
   → la moitié des cellules traversées n'étaient pas marquées. Le nombre
   d'échantillons est maintenant auto-dimensionné sur la résolution de la
   grille (`n_samples_for_grid`, ~2900 à 1/4°) ; continuité testée (>99 %
   de cellules consécutives adjacentes).

4. **`contrib/data_loading/data.py` — masque journalier tiré AU HASARD**
   `mask_input` tirait un masque aléatoire par pas de temps
   (`np.random.randint`) : progression jour-à-jour des traces détruite,
   entrée non reproductible. `open_var_dataset` applique désormais
   `mask_input_sequential` (masque du jour t → champ du jour t), avec
   vérifications de longueur et de forme. L'ancien comportement reste
   disponible (documenté comme legacy) pour `open_glorys12_data`.

5. **`contrib/argo/qc.py` — tri manquant avant le test d'inversion**
   `reject_pressure_inversions` exige un tri (profil, cycle, pression) que
   `run_pipeline` ne faisait jamais → rejets erronés de points valides sur
   la sortie argopy (non triée). `apply_standard_qc` trie désormais
   systématiquement (`sort_pointcloud`) ; régression testée.

6. **`contrib/argo/download.py` — téléchargement décennal en un appel**
   Ajout de `fetch_argo_profiles_chunked` (par mois, avec reprise sur
   erreur de chunk) — un appel région unique sur 10 ans échoue en pratique.

## Nouvelles briques (protocole OSSE)

- **`contrib/synthetic_obs/make_pseudo_obs.py`** : fabrique une fois pour
  toutes le fichier de pseudo-observations = vérité masquée le long des
  traces + bruit instrumental gaussien (σ configurable, graine fixée),
  identique en train/val/test. Contient aussi `obs_mask` (1/0), utilisable
  tel quel comme canal d'entrée de présence d'observation (aucune
  modification du cœur nécessaire).
- **`contrib/argo/virtual.py`** : ARGO *virtuels* — géométrie réelle des
  flotteurs (positions, dates, couverture verticale par profil, après QC),
  valeurs échantillonnées dans la vérité GLORYS (sélection vectorisée,
  tolérance temporelle explicite). Grillage inchangé en aval. Résout
  l'incohérence "valeurs ARGO réelles vs vérité GLORYS" du pipeline
  précédent, qui reste disponible pour la validation OSE réelle.
- **`contrib/synthetic_obs/build_masks.py`** : option `time_from` (n_days
  et jour 0 pris sur l'axe temporel du fichier de vérité → alignement par
  construction avec l'application séquentielle des masques) ; option
  `output_netcdf` (masque 1/0 daté).

## Configuration

- **`config/xp/osse3d_gs_multivar_unet.yaml`** : la nouvelle config de
  référence. OSSE Gulf Stream (32–44°N, 66–54°W, GLORYS natif 1/12°),
  entrées = SSH pseudo-obs 6 nadirs bruitée + canal de masque + SST vérité
  dense + ARGO virtuels (5 profondeurs) + bathy/lat ; cibles = zos +
  thetao/uo/vo × 5 profondeurs GLORYS (16 sorties). Bloc `paths:` unique à
  adapter. Les entrypoints régénèrent masques et pseudo-obs depuis l'axe
  temporel du fichier de vérité avant l'entraînement.
- **`config/vars/argo_virtual_thetao_depths.yaml`** : fragment d'entrées
  ARGO virtuelles (`prior_input`) — plus AUCUNE cible ARGO redondante : une
  grandeur physique = un seul canal de sortie (les cibles restent les
  thetao_* GLORYS).
- Fragments `config/vars/{thetao,uo,vo}_depths.yaml` paramétrés par
  `${paths.glorys_multidepth}` ; les 4 anciennes configs qui les utilisent
  reçoivent un bloc `paths:` de compatibilité et une note de dépréciation
  (protocole hybride non interprétable — conservées pour comparaison).

## Tests (`tests/`)

- `test_synthetic_obs.py` (numpy seul — **exécutés ici, 6/6 OK**) :
  périodes orbitales plausibles ; le garde-fou attrape la confusion
  passes/orbites ; couverture journalière nadir dans [1,5 ; 4,5] % à 1/4°
  (mesuré : Jason-3 3,6 %) ; régression explicite vs le bug ×2 ; continuité
  along-track > 99 % ; fermeture exacte du cycle de répétition.
- `test_argo_and_pseudo_obs.py` (nécessite l'env conda du projet) : le tri
  corrige les faux rejets d'inversion sur données entrelacées ; les ARGO
  virtuels échantillonnent bien la vérité (valeurs réelles jetées,
  couverture verticale préservée) ; pseudo-obs : NaN hors-trace, bruit du
  bon ordre, alignement temporel, déterminisme par graine.

## Limites restantes / suites proposées

- Pseudo-obs SST : la SST d'entrée est la vérité dense (approximation
  "SST L4 bien observée") ; une version masquée nuages serait plus réaliste.
- SWOT n'est pas dans la constellation par défaut de la config (ajouter
  `swot` aux missions + le chemin officiel `build_masks_swot_official`
  pour le bruit KaRIn réel).
- Variantes têtes/incertitude : réutilisables telles quelles sur la
  nouvelle config (mêmes mécanismes `head_group`/`combine_losses`).
- Métriques : ajouter nRMSE et résolution effective PAR PROFONDEUR
  (la courbe skill-vs-profondeur est le résultat central de l'extension).
- `MultivarBatchSelector` : `.cuda()` codé en dur (hérité) — GPU requis
  même pour un smoke test.

---

# Extension 21 niveaux + système d'ablations (2e lot)

## Sélection verticale par indices (source unique de vérité)
- `config/depths/gs21_indices.yaml` : LA liste des 21 positions (indices,
  base 0) dans l'axe `depth` du fichier GLORYS. À éditer ici et nulle part
  ailleurs.
- `contrib/data_loading/data.py` : nouvelle clé `depth_index` des entrées
  multivar (sélection exacte par `isel`, bornes vérifiées) — remplace
  `depth_level` (nearest, risque d'aliasing silencieux de deux profondeurs
  sur le même niveau natif) pour les configs multi-niveaux.
- `contrib/data_loading/depth_fragments.py` : codegen des fragments
  `config/vars/{thetao,uo,vo,argo_virtual_thetao}_gs21.yaml` (21 entrées
  chacun, nommées `{var}_d{index:02d}`) depuis la liste d'indices ; valide
  unicité/bornes, annote les valeurs en mètres avec `--truth-path`.
  Head groups : `temperature_glorys`, `currents_glorys_u`,
  `currents_glorys_v` (u et v séparés — requis pour des bases EOF propres).
- `contrib/argo/virtual.py` : option `--depth-indices` (résolution des
  valeurs depuis le fichier vérité, fichiers de sortie nommés par indice
  pour correspondre aux fragments générés).

## Équilibrage de la perte (obligatoire à 21 niveaux)
- `contrib/multivar/loss_grouping.py` : combinaison des pertes par variable
  avec mode `flat_sum` (historique) ou `group_mean` (moyenne intra-groupe
  puis somme pondérée inter-groupes — chaque grandeur physique pèse pareil,
  la SSH n'est plus noyée à 1/64). Pur Python, testé.
- `MultivarUNet_mae` : nouveaux flags `loss_group_mode`, `loss_groups`
  (générés depuis les tags head_group), `loss_group_weights`,
  `grad_loss_weight` (réintroduction du terme de gradient Sobel que la
  variante MAE avait perdu ; pondéré, loggé par variable). Défauts =
  comportement historique exact. Log des pertes par groupe.

## Métriques par profondeur
- `metric/eulerian/depth_profile_metrics.py` : nRMSE, biais et résolution
  effective (PSD isotrope, seuil 0,5 — définition standard du mapping SSH)
  PAR canal de sortie, comparés directement aux fichiers vérité GLORYS ;
  CSV trié par (variable, profondeur) + résumé par famille. Le modèle
  exporte désormais `output_var_names.json` au début du test pour le
  mapping dim→variable.

## Tête en modes verticaux
- `contrib/multivar/vertical_modes.py` : calcul des EOF verticales
  standardisées par niveau sur la période train (CLI) + solveur
  `VerticalModesUNet` (tronc UNet → K coefficients par groupe/pas de temps
  → projection figée vers les niveaux physiques). Orthonormalité, ordre de
  variance et reconstruction testés (numpy).

## Ablations (groupe Hydra `config/ablation/`, CLI `ablation=<nom>`)
- `baseline` : UNet + group_mean + gradient Sobel (référence).
- `flat_sum_loss`, `no_grad_loss` : ablations de perte.
- `uncertainty` : pondération par incertitude apprise (supersède le
  groupage — combine_losses surchargée).
- `heads` : têtes convolutives par groupe.
- `vertical_modes` : tête en modes verticaux (EOF à générer d'abord, voir
  le fichier).
- `attention` : self-attention aux échelles grossières du UNet.
- Variante surface seule : `xp=osse3d_gs21_surface_only` (identique sans
  les entrées ARGO virtuelles — l'écart de skill par profondeur quantifie
  l'apport du réseau in situ).
- Les deux solveurs alternatifs ignorent explicitement la clé
  `out_channels` héritée par fusion Hydra du solveur de base (collision de
  kwargs sinon).

## Configs maîtresses
- `config/xp/osse3d_gs21_multivar_unet.yaml` : 64 cibles (zos + 3×21
  niveaux), ~110 canaux d'entrée dont 21 ARGO virtuels, group_mean +
  gradient par défaut, `- /ablation: baseline` dans les defaults.
- `config/xp/osse3d_gs21_surface_only.yaml` : idem sans in situ.

## Ordre de préparation (21 niveaux)
1. Éditer `config/depths/gs21_indices.yaml`, régénérer les fragments
   (`depth_fragments.py --truth-path ...` pour valider les valeurs).
2. ARGO virtuels : `python -m contrib.argo.virtual ... --depth-indices <liste>`.
3. (ablation modes) EOF : `python -m contrib.multivar.vertical_modes ...`
   pour thetao, uo, vo, mêmes indices.
4. `python main.py xp=osse3d_gs21_multivar_unet [ablation=...]`.
5. Métriques : `python -m metric.eulerian.depth_profile_metrics ...`.

## Tests (tous numpy, exécutés : 11/11 OK)
- test_loss_and_modes.py : flat_sum ≡ historique ; group_mean équilibre les
  familles ; poids de groupe ; EOF orthonormales/ordonnées/reconstructives ;
  rejet des entrées sous-déterminées.

## Restes connus
- Ablation GradSolver (4DVarNet variationnel multivarié) : non câblée — le
  prior multivarié (`multivar_costs`) doit être porté vers le nouveau jeu de
  variables ; prévu comme lot suivant.
- Ablation temporelle (conv 3D / récurrence) : non implémentée (déclarée
  comme suite dans la discussion d'architecture).
- Chargement eager : ~25-30 Go RAM attendus à 21 niveaux sur le domaine GS —
  surveiller ; variantes lazy/dask disponibles dans contrib/data_loading/.

---

# Support CPU / agnosticité au GPU (3e lot)

Objectif : le chemin de code OSSE tourne désormais tel quel sur machine sans
GPU (portable, CI, smoke test), sans crash "Torch not compiled with CUDA" /
"no CUDA device".

- `contrib/multivar/device_utils.py` : helper central `default_device()`
  (auto CUDA si dispo, sinon CPU ; surchargeable par la variable
  d'environnement `NOSC_DEVICE`, p.ex. `NOSC_DEVICE=cpu` force le CPU même
  sur une machine GPU) et `to_device()`.
- `.cuda()` codés en dur remplacés sur tout le chemin actif :
  `multivar_utils.py` (5 buffers d'indices du sélecteur → `default_device()`),
  `multivar_data.py` (3 tenseurs de reconstruction → `device=default_device()`),
  hooks de test de `multivar_models_unet_mae.py` et des classes de base/UNet
  (le `test_data` est déjà sur CPU à l'export, `.cuda()` superflu retiré).
- `clear_gpu_mem` : `torch.cuda.empty_cache()` gardé par
  `torch.cuda.is_available()`.
- Configs OSSE : `accelerator: gpu` → `accelerator: auto` (Lightning
  détecte GPU/CPU). Les `.cuda()` restants du dépôt sont dans des modules
  hors chemin OSSE (`_rec`, `_finescale`, `mapping_fastrec`, `forecast_plus`,
  `multiprior`…) ou déjà commentés — laissés tels quels.
- Test `tests/test_device_agnostic.py` : scan statique (aucun `.cuda()` non
  commenté sur le chemin actif) + override `NOSC_DEVICE=cpu` (exécuté ; 12
  tests numpy au total OK).

ATTENTION perf : sur CPU, le smoke test (2 epochs × 5 batches) est utilisable
pour vérifier l'intégration, mais un entraînement complet à 21 niveaux y est
irréaliste — le CPU sert à valider la mécanique, pas à produire des résultats.


---

# Support CPU (exécution sans GPU) — 3e lot

Objectif : le chemin OSSE tourne à l'identique sur GPU ou sur une machine
sans GPU (dev, CI, smoke tests). L'helper `contrib/multivar/device_utils.py`
existait mais n'avait jamais été câblé (`.cuda()` toujours partout) — corrigé.

- **`MultivarBatchSelector`** : les tenseurs d'index ne sont plus épinglés à
  un device deviné à la construction ; ils suivent désormais le device du
  batch entrant (`_idx_on`), ce qui est correct sur CPU, mono-GPU et DDP.
  Import `default_device` devenu inutile, retiré.
- **`src/train.py`** : `torch.load(..., map_location=...)` — un checkpoint
  entraîné sur GPU se recharge sur une machine CPU (et inversement).
- **`multivar_models_unet_psi.py`** (hors chemin OSSE, corrigé par
  prudence) : `dx/dy/coriolis` en buffers (suivent le device du module),
  plus de `.to("cuda")`.
- **Configs** : `accelerator: auto` (auto-détection Lightning) dans les
  configs OSSE ; nouvelle `config/xp/osse3d_gs21_cpu_smoke.yaml` (CPU forcé,
  1 epoch, 2 batches) ; variable `NOSC_DEVICE=cpu` pour forcer le CPU même
  sur une machine GPU.
- Vérifié : plus aucun `.cuda()`/`.to("cuda")` actif dans le chemin OSSE
  (sélecteur, modèle MAE, ablations heads/uncertainty/modes, train). Les
  hooks hérités `*_theo` (chemin de test non utilisé par MAE) et de nombreux
  modules `multivar_*` inutilisés conservent des `.cuda()` — sans effet sur
  les configs OSSE.
- **`tests/test_device_agnostic.py`** (nécessite torch) : override
  `NOSC_DEVICE`, cohérence avec le matériel, et surtout le sélecteur tourne
  sur CPU en alignant ses index sur le device du batch.

Note : ceci rend enfin exécutable un smoke test d'intégration SANS GPU —
`python main.py xp=osse3d_gs21_cpu_smoke paths.data_root=$DATA_ROOT` — utile
pour valider tout le câblage avant de réserver un GPU.

---

# Réanalyse avec process_data/ (4e lot) — cohérence pipeline de données

L'ajout du répertoire process_data/ (182 fichiers : la préparation de données
du projet parent) a permis de vérifier la cohérence de la refonte contre le
pipeline réel et a révélé plusieurs faits importants.

## Découvertes (et corrections de mon propre diagnostic)
1. **Les masques d'origine étaient RÉELS, pas synthétiques** :
   `mask/glorys_masking.ipynb` grille de vraies traces L3 6-satellites
   (année 2019, aussi 2022) en 1/NaN. Le tirage ALÉATOIRE de `mask_input`
   que j'avais qualifié de bug était le mécanisme voulu pour recycler 365
   masques réels sur 3653 jours — critiquable (non daté, non reproductible)
   mais intentionnel. Correction du correctif : nouveau `mask_mode: cycle`
   (recyclage modulo déterministe, préserve la progression intra-annuelle)
   pour ce cas d'usage ; `sequential` reste le défaut ; message d'erreur
   orienté vers les deux issues. Testé (numpy).
2. **Des masques réels PLEINE PÉRIODE existent** : ssh_osse.ipynb référence
   `altimetry_traces/2010_2023/gridded/l3_mask.nc` (+ SLA L3 grillée
   2010-2023 à 1/4°). `make_pseudo_obs` accepte désormais un masque NetCDF
   DATÉ (aligné par .sel, tolérance 1 jour, régrillage nearest si besoin,
   conventions 1/NaN ou 1/0) en plus du pickle indexé → l'OSSE peut tourner
   avec l'échantillonnage réel historique (constellation non stationnaire
   comprise), option supérieure au générateur synthétique quand le fichier
   est accessible. Le générateur reste utile hors cluster et pour les
   études de constellation (dropout, ajout SWOT).
3. **Deux conventions de grille 1/4° coexistent** dans process_data :
   coarsening historique 2041//3 (680×1440, ssh_osse/glorys_9y_4th) vs
   linspace(-90,90,721)[:-1] (720×1440, make_train_dataset_glorys). Piège :
   masques et données préparés sur des grilles différentes ne s'alignent
   pas (make_pseudo_obs régrille désormais en nearest, mais vérifier la
   provenance). Le chemin OSSE GS (1/12° natif régional) n'est pas concerné.
4. **`pad/add_pad.py`** : padding wrap ±4 en longitude pour les convolutions
   globales — le parent avait une solution à la couture du méridien 180°
   que j'avais signalée. Sans objet en régional ; à réutiliser tel quel
   pour un futur run global.
5. **`mask/mask_uv.ipynb`** : masque statique des courants |lat|<10°
   (géostrophie invalide à l'équateur) et |lat|>70°. Sans objet sur le
   Gulf Stream ; indispensable à reporter dans les cibles uv d'un run
   global.
6. Modules OSE du parent, cohérents avec le plan OSSE→OSE et l'env
   (jax/jaxparrow présents) : drifters (cartes journalières AOML/CMEMS),
   ageostrophy (uv géostrophiques + résidu), cyclogeo (cyclogéostrophie
   jaxparrow), mld, era5, interpolation (DUACS/NeurOST/SST L3-L4 vers
   1/4°-1/8°), compress (float64→32), noise (écarts GLORYS vs DUACS vs
   NeurOST — désaccords de produits O(20 cm), sans lien avec le bruit
   instrumental le long de trace : le σ=2 cm des pseudo-obs reste le bon
   ordre de grandeur).

## Cohérence globale après réanalyse
Aucune incompatibilité entre la refonte et process_data : le contrat des
masques pickle est respecté (même format), les configs dépréciées annotées
(mask_mode), et le chemin OSSE (GS 1/12° natif) est orthogonal au pipeline
1/4° historique. Les nouveaux points d'attention sont documentés ci-dessus.
