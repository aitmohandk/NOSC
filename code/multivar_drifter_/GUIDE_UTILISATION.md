# Mode d'emploi — NOSC (Neural Ocean Surface Currents)

Ce document explique comment utiliser le projet : préparer les données, configurer et lancer un entraînement, faire de l'inférence/reconstruction, et où se trouve chaque nouvelle brique (multi-variable, 3D, observations synthétiques, têtes spécialisées, repondération) ajoutée pour étendre l'architecture d'origine.

Pour le raisonnement architectural complet derrière ces extensions, voir `/home/k24aitmo/.claude/plans/staged-meandering-ocean.md`.

---

## 1. Vue d'ensemble

NOSC repose sur **4DVarNet** : un réseau de neurones d'assimilation de données (Hydra + PyTorch Lightning) qui reconstruit des champs océaniques denses à partir d'observations satellite éparses. Le cœur du framework est dans `src/` (dataset générique, solveur variationnel `GradSolver`, boucle d'entraînement). Toutes les extensions (multi-variable, 3D, observations synthétiques, etc.) vivent dans `contrib/` et sont assemblées par composition de configuration Hydra — **aucune donnée réelle n'est modifiée dans `src/`**.

Le point d'entrée unique est `main.py` :

```bash
python main.py xp=<nom_de_la_config>
```

`<nom_de_la_config>` correspond à un fichier `config/xp/<nom_de_la_config>.yaml`.

---

## 2. Installation

```bash
conda env create -f ../../env/4dvarnet-daniel.yaml   # depuis code/multivar_drifter_
conda activate 4dvarnet-daniel
```

L'environnement contient désormais aussi `argopy` (pipeline ARGO) et `swot-simulator` (simulateur SWOT officiel CNES/JPL), ajoutés pour les extensions décrites plus bas.

---

## 3. Préparation des données

### 3.1 Données brutes (Glorys, SSH/SST L4, ERA5, dérives, bathymétrie)

Scripts dans `code/download_data_/` (utilisent `copernicusmarine`) :
- `import_data_glorys_0m_year.py <année>` / `import_data_glorys_15m_year.py <année>` — Glorys à un seul niveau de profondeur (usage historique).
- `import_data_glorys_multidepth.py <année> [profondeur_min] [profondeur_max]` — **nouveau**, télécharge une plage de profondeurs Glorys en un seul appel (nécessaire pour les configs 3D, section 6).
- `import_data_ssh_L4.py`, `import_data_sst_L4.py`, `import_data_era5.py` — produits gridés L4 utilisés comme `prior_input`.

Voir `code/process_data/drifters/make_daily_uv_map_aoml.py` pour le motif de grillage des dérives de surface (repris pour ARGO, section 3.3).

### 3.2 Masques d'observation (traces satellite)

Un masque est une liste Python de tableaux 2D journaliers (1.0 = observé, NaN = non observé), sérialisée en pickle et référencée via la clé `mask_path` d'une entrée `multivar:`. Trois façons de le produire :

**a. Réutiliser un vrai produit multi-satellite déjà grillé** (méthode d'origine du projet) : `code/process_data/mask/glorys_masking.ipynb`.

**b. Générateur de traces synthétiques en formule fermée** (`contrib/synthetic_obs/`) — simule un ensemble de missions nadir (Jason-3, Sentinel-3A/B, SARAL, HY-2B) + SWOT à fauchée large, sans dépendance à un fichier externe :

```bash
python -m contrib.synthetic_obs.build_masks \
  --grid-from <fichier_de_référence_avec_lat_lon> \
  --n-days 3653 \
  --missions jason3 sentinel3a sentinel3b saral hy2b swot \
  --output <chemin_de_sortie.pickle>
```

**c. Simulateur SWOT officiel (CNES/JPL)** — orbite réelle (vraies éphémérides SWOT) et vrai modèle de bruit instrumental KaRIn, uniquement pour la plateforme SWOT :

```python
from contrib.synthetic_obs.swot_official_settings import write_settings_file
from contrib.synthetic_obs.build_masks_swot_official import run_swot_simulator, build_and_serialize_swot_official_masks

write_settings_file(
    "settings.py", glorys_path="<fichier_glorys.nc>", glorys_var="zos",
    working_directory="<répertoire_de_sortie_swot_simulator>",
)
run_swot_simulator("settings.py", first_date="2019-01-01", last_date="2019-12-31", n_workers=4)
build_and_serialize_swot_official_masks(
    "<masques_swot_officiels.pickle>", "<répertoire_de_sortie_swot_simulator>",
    lat_grid, lon_grid, "2019-01-01", "2019-12-31", with_values=False,  # True pour de vraies valeurs SSH bruitées, pas juste un masque
)
```

`swot_official_settings.py` ne couvre que SWOT (pas de Jason/Sentinel-3/SARAL/HY-2B — ces missions historiques n'ont pas d'éphémérides embarquées dans l'outil) ; pour elles, l'option (a) ou (b) reste pertinente.

### 3.3 Pipeline ARGO (profils de température/salinité en profondeur)

```bash
python -m contrib.argo.run_pipeline \
  --grid-from <fichier_de_référence> \
  --start-date 2010-01-01 --end-date 2020-01-01 \
  --lat-min -70 --lat-max 70 --lon-min -180 --lon-max 180 \
  --depths 0.49 15 50 100 200 \
  --output-dir <répertoire_de_sortie>
```

Chaîne : téléchargement (`contrib/argo/download.py`) → contrôle qualité (`contrib/argo/qc.py`, flags QC standards + rejet des inversions de pression) → interpolation verticale sur les niveaux de profondeur du modèle (`contrib/argo/vertical_interp.py`) → grillage journalier (`contrib/argo/build_argo_dataset.py`, un fichier `.nc` par couple variable/profondeur). Produit les fichiers attendus par `config/vars/thetao_argo_depths.yaml`.

Pour une **validation indépendante** (sans entraîner, sans grillage) contre une reconstruction déjà produite : `contrib/argo/colocate.py` (`colocate_profiles_pointwise` + `compute_validation_metrics`, RMSE/biais par profondeur).

### 3.4 Fusion des fichiers Glorys multi-profondeur

Après `import_data_glorys_multidepth.py`, fusionner les fichiers annuels en un seul (attendu par `config/vars/thetao_depths.yaml`, `uo_depths.yaml`, `vo_depths.yaml`) :

```bash
python -c "
import xarray as xr
xr.open_mfdataset('<dossier>/*.nc', combine='by_coords').to_netcdf('<dossier>/glorys_multidepth_2010-2020.nc')
"
```

---

## 4. Configurer une expérience (Hydra)

Une config d'expérience (`config/xp/*.yaml`) décrit tout : quelles variables entrent/sortent du modèle, le domaine spatio-temporel, le solveur, l'optimiseur, les points d'entrée (`entrypoints:`).

### 4.1 Le dictionnaire `multivar:`

Chaque variable (ou couple variable/profondeur) est une entrée :

```yaml
multivar:
  thetao_50m:
    var_path: /chemin/vers/fichier.nc
    var_name: thetao          # nom de la variable dans le fichier
    input_arch: no_input       # full_input | prior_input | no_input
    output_arch: full_output   # full_output | no_output
    broadcast_time: False      # True si le champ n'a qu'un pas de temps à dupliquer (ex. bathymétrie)
    depth_level: 50            # optionnel : sélectionne un niveau dans un fichier multi-profondeur
    mask_path: /chemin/masques.pickle  # optionnel : masque l'entrée pour simuler des observations éparses
    head_group: temperature_glorys      # optionnel : voir section 6.4 (têtes spécialisées)
```

- `prior_input` : informe le modèle mais n'est pas reconstruit.
- `full_output` : cible de reconstruction (perte calculée dessus).
- Une variable peut être les deux à la fois (rare) en dupliquant l'entrée sous deux clés différentes (voir `ssh_tgt` dans les configs Phase 0/5, qui duplique `ssh`).

### 4.2 Groupes de variables réutilisables (`config/vars/`)

Pour éviter de dupliquer à la main une entrée par profondeur, `contrib/data_loading/multidepth.py` génère des fragments YAML :

```python
from contrib.data_loading.multidepth import write_depth_levels_yaml
write_depth_levels_yaml(
    "config/vars/thetao_depths.yaml", "thetao",
    dict(var_path="...", var_name="thetao", input_arch="no_input", output_arch="full_output",
         broadcast_time=False, head_group="temperature_glorys"),
    depth_levels=[0.49, 15, 50, 100, 200],
)
```

Puis dans la config d'expérience :

```yaml
defaults:
  - /vars/thetao_depths     # le slash initial est important (groupe racine, pas relatif à xp/)
  - _self_
```

### 4.3 Créer une nouvelle expérience

Le plus simple : copier une config existante proche de votre besoin (voir liste section 5) et modifier `multivar:`, `datamodule.domains`, ou `model.solver`.

---

## 5. Configs disponibles

| Config | Ce qu'elle teste |
|---|---|
| `unet_uv_aoml_15m_10y_11d_bathy_sst_mae_duacs_RonanUnet` | Référence d'origine : courants de surface (u/v des dérives) |
| `unet_uv_ssh_sst_aoml_15m_10y_11d_bathy_mae_duacs_RonanUnet` | + SSH et température en sortie (Phase 0) |
| `unet_uv_temp3d_aoml_15m_10y_11d_bathy_mae_duacs_RonanUnet` | + température/courants Glorys sur 5 profondeurs (Phase 2) |
| `unet_uv_full_integration_15m_10y_11d_bathy_mae_duacs_RonanUnet` | Tout combiné : multi-variable + 3D + ARGO + masques synthétiques (Phase 5, référence pour les comparaisons) |
| `unet_uv_full_integration_heads_15m_10y_11d_bathy_mae_duacs_RonanUnet` | Identique, avec têtes spécialisées par groupe de variables |
| `unet_uv_full_integration_uncertainty_15m_10y_11d_bathy_mae_duacs_RonanUnet` | Identique, avec repondération de perte par incertitude apprise |

Les autres fichiers dans `config/xp/` (hors `old/`) sont des variantes historiques du projet, antérieures à cette session.

---

## 6. Entraînement

### 6.1 En local (test rapide)

```bash
python main.py xp=<nom> --cfg job --resolve   # valide juste que la config se compose (aucun calcul)
python main.py xp=<nom> ++trainer.max_epochs=2 ++trainer.limit_train_batches=5   # smoke test réel, GPU requis
```

`MultivarBatchSelector` (le sélecteur de canaux multi-variable) appelle `.cuda()` en dur — **un GPU est requis**, y compris pour un smoke test.

### 6.2 Sur le cluster (SLURM/Odyssey)

Utiliser `launch_architecture_extension.sh` (à la racine de `code/multivar_drifter_/`) :

```bash
# adapter ODYSSEY_USER en tête de fichier avant toute chose
./launch_architecture_extension.sh glorys-download   # 1 job par année
./launch_architecture_extension.sh glorys-merge       # après la fin des téléchargements
./launch_architecture_extension.sh masks-dryrun        # test local rapide (5 jours, pas de sbatch)
./launch_architecture_extension.sh masks               # génération complète
./launch_architecture_extension.sh argo                 # pipeline ARGO complet
./launch_architecture_extension.sh smoke                 # Phase 0, 2 epochs — détecte les erreurs vite
./launch_architecture_extension.sh scale-2d3d            # Phase 2 en réel
./launch_architecture_extension.sh scale-full            # Phase 5 en réel
./launch_architecture_extension.sh compare                # référence + têtes + incertitude en parallèle
```

Pour une expérience hors catalogue, s'inspirer des blocs `sbatch <<EOF ... EOF` du script et changer `xp=<nom>`.

### 6.3 Suivi

Chaque run écrit dans `outputs/<date>/<heure>/<nom_xp>/` (répertoire Hydra standard) : logs CSV (`CSVLogger`), checkpoints (`ModelCheckpoint`, top-3 sur `val_total_mse`). Comparer plusieurs runs : lire les `metrics.csv` de chaque run, ou comparer les `{phase}_{variable}_mse` loggés par variable (déjà présents pour chaque sortie, y compris avec les têtes spécialisées et la repondération — cette dernière logue en plus `{phase}_{variable}_log_var`).

### 6.4 Têtes spécialisées et repondération (rappel)

- **Têtes** (`contrib/multivar/multivar_models_unet_heads.py`) : un tronc partagé (`UNetModel`, inchangé) jusqu'à un "cou" étroit, puis une petite tête convolutive indépendante par `head_group`. Les variables d'un même groupe doivent être **contiguës** dans `multivar:` (l'ordre des canaux de sortie doit correspondre exactement à l'ordre des cibles) — une erreur explicite est levée sinon.
- **Repondération** (`contrib/multivar/multivar_models_unet_uncertainty.py`) : un poids appris par variable (`log_vars`), initialisé à zéro (aucun effet au départ). Nécessite `cosanneal_lr_adam_unet_uncertainty` comme `opt_fn` (l'optimiseur standard n'inclurait pas `log_vars`).

---

## 7. Inférence / reconstruction

`src.train.base_training` (appelé par tous les `entrypoints:`) enchaîne entraînement puis test automatiquement. Pour reconstruire uniquement à partir d'un checkpoint existant (sans réentraîner), décommenter/ajouter dans la config :

```yaml
entrypoints:
  - _target_: src.train.base_training
    trainer: ${trainer}
    lit_mod: ${model}
    dm: ${datamodule}
    ckpt: /chemin/vers/checkpoint.ckpt
    only_rec: True
```

Sortie : un fichier `.nc` par dimension de sortie (`test_data_dim0.nc`, `test_data_dim1.nc`, ...), dans l'ordre de `multivar_output_var_names()` (l'ordre des entrées `full_output` dans `multivar:`). Chaque fichier contient les champs `inp`/`tgt`/`out` reconstruits sur le domaine de test.

### 7.1 Validation ARGO indépendante

Une fois des `test_data_dimN.nc` produits pour les profondeurs de température (Phase 2/5), les comparer aux profils ARGO réels sans réentraîner :

```python
from contrib.argo.colocate import colocate_profiles_pointwise, compute_validation_metrics
# interp_df : sortie de contrib.argo.vertical_interp.interp_argo_profiles sur des profils QC'd
colocated = colocate_profiles_pointwise(interp_df, {50: "test_data_dim2.nc", 100: "test_data_dim3.nc", ...}, value_col="TEMP")
print(compute_validation_metrics(colocated))  # RMSE/biais/écart-type par profondeur
```

---

## 8. Pièges connus

- **Chemins à adapter** : toutes les nouvelles configs pointent vers des chemins `/Odyssey/private/t22picar/...` — ce sont des emplacements plausibles, pas garantis exister. Les adapter à votre propre stockage avant de lancer quoi que ce soit.
- **GPU obligatoire**, y compris pour des tests courts (voir 6.1).
- **`thetao_argo_*` vs `thetao_*`** : ce sont deux cibles de sortie *distinctes* (Glorys dense vs ARGO épars), pas la même variable dédoublée — voir la note dans `config/vars/thetao_argo_depths.yaml`.
- **`sst` ≠ température brute** : l'entrée `sst` historique (`sst_transfo: True`) est transformée en `log|∇T|` (une feature de fronts), pas la température elle-même. Pour une vraie cible de température de surface, utiliser `sst_tgt` (sans `sst_transfo`).
- **Ordre des `head_group`** : doit être contigu dans `multivar:`, sinon `get_multivar_head_groups` lève une erreur explicite plutôt que de mélanger silencieusement les canaux.

---

## 9. Protocole OSSE propre (mise à jour)

Les configs `unet_uv_full_integration_*` sont **dépréciées** (protocole
hybride : voir CHANGES_OSSE.md). La config de référence est désormais
`osse3d_gs_multivar_unet` (OSSE Gulf Stream, vérité GLORYS de bout en bout).

Préparation (une fois, chemins dans le bloc `paths:` de la config) :

```bash
# 1. vérité GLORYS régionale (surface + multi-profondeur)
python ../download_data_/import_data_glorys_multidepth.py <année> 0 200   # puis subset domaine + merge (section 3.4)
# 2. ARGO virtuels (géométrie réelle, valeurs GLORYS)
python -m contrib.argo.virtual \
  --grid-from <glorys_surface.nc> --truth-path <glorys_multidepth.nc> \
  --start-date 2010-01-01 --end-date 2020-01-01 \
  --lat-min 32 --lat-max 44 --lon-min -66 --lon-max -54 \
  --depths 0.49 15 50 100 200 --output-dir <data_root>/argo_virtual/gridded
# 3. masques + pseudo-obs : générés par les entrypoints de la config
python main.py xp=osse3d_gs_multivar_unet
```

Tests : `python tests/test_synthetic_obs.py` (numpy seul) puis, dans l'env
conda, `python tests/test_argo_and_pseudo_obs.py`.
