# NOSC sur Jean Zay — guide pas à pas

De la construction de l'image au premier entraînement, dans l'ordre. Chaque
étape se valide avant de passer à la suivante.

**Contexte.** Projet `yrf`, login `uag47zl`. Le dépôt est dans
`$WORK/NOSC`, le code d'expérience dans `$WORK/NOSC/code/multivar_drifter_`.

**Deux contraintes de Jean Zay que le guide du projet n'aborde pas**, car il vise
le cluster Odyssey :

- **Les nœuds de calcul n'ont pas d'accès Internet**, et les frontales
  n'exécutent pas de conteneurs. Tout téléchargement — GLORYS via
  `copernicusmarine`, profils ARGO via `argopy` — passe donc par un nœud de
  **pré/post-traitement** (`--partition=prepost`), seul endroit qui cumule accès
  Internet, mémoire abondante et exécution de conteneurs.
- **Les quotas sont serrés** : 500 000 inodes sur le `$WORK`, partagés entre tous
  les membres du projet. D'où le conteneur, et d'où le placement des données et
  des sorties sur `$SCRATCH`.

---

# A. Construire l'image (sur votre poste Ubuntu)

Jean Zay n'autorise pas la construction depuis un fichier de définition : elle
exige les droits root, indisponibles sur le calculateur.

### A.1 Installer Apptainer

```bash
lsb_release -a
```

**Ubuntu 22.04 ou antérieur** — le PPA officiel :

```bash
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update && sudo apt install -y apptainer
```

**Ubuntu 24.04 ou plus récent** — le paquet `.deb` des releases GitHub, qui
installe le profil AppArmor autorisant les espaces de noms non privilégiés,
absent du paquet PPA :

```bash
# dernier apptainer_<version>_amd64.deb sur
# https://github.com/apptainer/apptainer/releases
sudo apt install -y ./apptainer_*_amd64.deb
apptainer --version
```

### A.2 Contrôler le YAML avant de lancer

```bash
cd ~/NOSC/env          # impératif : %files lit le YAML dans le répertoire courant
grep -n "oceanbench\|loguru" nosc-jeanzay.yaml
```

Vous devez voir `- git+https://github.com/jejjohnson/oceanbench.git` et **aucune**
ligne `- oceanbench` seule : le paquet homonyme de PyPI est celui de Mercator
Ocean, sans rapport avec ce projet. Et `- loguru`, dépendance rangée dans un
groupe Poetry qu'un `pip install git+…` ignore.

### A.3 Construire et valider localement

```bash
apptainer build --fakeroot nosc-$(date +%Y-%m).sif nosc.def
apptainer exec nosc-*.sif \
  python -c "import ocean4dvarnet, oceanbench, ocn_tools, xrpatcher, jaxparrow; print('TOUT OK')"
```

Comptez 30 à 40 minutes. Apptainer ne construit pas de façon incrémentale :
toute modification du YAML relance l'intégralité du processus.

---

# B. Transférer vers Jean Zay

L'accès passe par un rebond. À configurer une fois dans `~/.ssh/config` :

```
Host imt
    HostName ssh.telecom-bretagne.eu
    User <votre-login-imt>

Host jz
    HostName jean-zay.idris.fr
    User uag47zl
    ProxyJump imt
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

```bash
rsync -avP nosc-2026-08.sif jz:/lustre/fswork/projects/rech/yrf/uag47zl/
```

`rsync -P` affiche la progression et reprend un transfert interrompu. `scp` reste
silencieux jusqu'à la fin.

---

# C. Importer l'image

```bash
ssh jz
module load singularity
idrcontmgr cp $WORK/nosc-2026-08.sif
idrcontmgr ls
rm $WORK/nosc-2026-08.sif        # récupérer l'espace du WORK
```

`idrcontmgr` vérifie les contraintes de sécurité et place l'image dans
`$SINGULARITY_ALLOWED_DIR`, seul répertoire depuis lequel elle peut s'exécuter.
Ce chemin n'est pas modifiable et l'espace est limité à 20 conteneurs. Ne touchez
pas à `$SINGULARITY_CACHEDIR`, créé automatiquement dans votre `$SCRATCH`.

---

# D. Vérifier le conteneur

Aucune donnée n'est nécessaire à cette étape. Les conteneurs ne s'exécutent
**jamais sur les frontales** : ouvrez une session sur un nœud de calcul. V100 en
QoS de développement suffit — pas de module `arch/` à charger, une variable de
moins pendant le débogage.

### D.1 Session interactive

```bash
srun -A yrf@v100 --qos=qos_gpu-dev --gres=gpu:1 --cpus-per-task=10 \
     --time=00:30:00 --hint=nomultithread --pty bash
module load singularity
```

### D.2 Le GPU est-il visible ?

```bash
singularity exec --nv --bind $WORK:$WORK,$SCRATCH:$SCRATCH \
  $SINGULARITY_ALLOWED_DIR/nosc-2026-08.sif \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Attendu : `True NVIDIA V100-...`. Si `False`, c'est l'option `--nv` manquante.

### D.3 Les imports du projet passent-ils ?

```bash
singularity exec --nv --bind $WORK:$WORK,$SCRATCH:$SCRATCH \
  $SINGULARITY_ALLOWED_DIR/nosc-2026-08.sif \
  python -c "import ocean4dvarnet, oceanbench, ocn_tools, xrpatcher, jaxparrow; print('imports OK')"
```

Sortez avec `exit`.

---

# E. Organiser les espaces de stockage

| Espace | Volume | Purge | Sauvegarde | Usage |
|---|---|---|---|---|
| `$HOME` | 3 Go, 150 000 inodes | non | oui | configuration — **jamais de données** |
| `$WORK` | 5 To, 500 000 inodes (projet) | non | non | code |
| `$SCRATCH` | très grand | **30 jours sans accès** | non | **données, sorties, checkpoints** |
| `$STORE` | très grand, peu d'inodes | non | oui | archives `.tar` |
| `$JOBSCRATCH` | disque local du nœud | fin du job | non | pré-staging, le plus rapide |
| `$DSDIR` | — | — | — | jeux publics IDRIS, lecture seule |

```bash
export DATA_ROOT=$SCRATCH/nosc/data
mkdir -p $DATA_ROOT $SCRATCH/nosc_runs
echo "export DATA_ROOT=$SCRATCH/nosc/data" >> ~/.bashrc
```

**Se prémunir de la purge.** `$SCRATCH` est effacé après trente jours sans accès.
Gardez la copie maître archivée, en une seule archive puisque `$STORE` offre
beaucoup de volume mais peu d'inodes :

```bash
tar cf $STORE/nosc_data_2026-08.tar -C $SCRATCH/nosc data/
```

**Surveiller le nombre de fichiers, pas seulement le volume.** Un jeu découpé en
un NetCDF par jour et par capteur produit vite des dizaines de milliers de
fichiers, ce qui dégrade Lustre pour tous et consomme vos inodes. Privilégiez
l'agrégation, avec des chunks de quelques dizaines à quelques centaines de
mégaoctets alignés sur le motif de lecture — pour du 4DVarNet, des patchs
spatiaux parcourus dans le temps.

```bash
idr_quota_user
idr_quota_project
```

---

# F. Préparer les données

**C'est l'étape la plus longue, et elle doit précéder tout entraînement.** Le
pipeline reconstruit des champs océaniques denses à partir d'observations éparses :
il lui faut une vérité GLORYS, des ARGO virtuels, et des masques d'observation.

Le protocole de référence est l'OSSE Gulf Stream, avec vérité GLORYS de bout en
bout. Domaine : latitude 32 à 44, longitude −66 à −54.

### F.1 Valider la composition de la configuration

À faire **en premier** : cela ne demande ni GPU ni données, et révèle
immédiatement les chemins à corriger.

```bash
cd $WORK/NOSC/code/multivar_drifter_
export XP=osse3d_gs_multivar_unet        # config de référence du protocole OSSE

singularity exec --bind $WORK:$WORK,$SCRATCH:$SCRATCH \
  $SINGULARITY_ALLOWED_DIR/nosc-2026-08.sif \
  python main.py xp=$XP --cfg job --resolve
```

Notez tous les chemins affichés : ce sont les fichiers que les étapes suivantes
doivent produire.

### F.2 Adapter les chemins

Les configurations du projet pointent vers `/Odyssey/private/t22picar/…`, des
emplacements du cluster d'origine qui n'existent pas ici.

```bash
grep -rn "Odyssey\|data_root" config/ | head -30
```

Reportez tout sous `${paths.data_root}`, que vous passerez en ligne de commande
via `paths.data_root=$DATA_ROOT`.

### F.3 Télécharger GLORYS — sur un nœud de pré/post-traitement

**Pas sur une frontale.** Le temps CPU y est limité et un téléchargement de
plusieurs heures sera interrompu ; ces machines sont partagées par tous les
utilisateurs ; et surtout, **les conteneurs ne peuvent pas y être exécutés**,
alors que `copernicusmarine` est dans l'image.

**Pas sur un nœud de calcul non plus** : ils n'ont pas d'accès Internet.

Le nœud de pré/post-traitement cumule les trois propriétés nécessaires — accès
Internet, mémoire abondante, exécution de conteneurs autorisée — et n'entame pas
vos heures GPU.

```bash
srun -A yrf@cpu --partition=prepost --time=10:00:00 --pty bash
module load singularity
cd $WORK/NOSC/code/multivar_drifter_
```

Le compte est **obligatoire** : dès que plusieurs allocations existent, Slurm
refuse la soumission avec « Multiple accounts available ». `yrf@cpu` est le bon
choix ici, les nœuds de pré/post-traitement n'étant pas des nœuds GPU.

Vérifiez la durée maximale de la partition avant de vous fier à ces dix heures :

```bash
scontrol show partition prepost | grep -i maxtime
```

Première exécution : enregistrez vos identifiants Copernicus Marine une fois pour
toutes, sinon chaque année du téléchargement les redemandera.

```bash
singularity exec --bind $WORK:$WORK,$SCRATCH:$SCRATCH \
  $SINGULARITY_ALLOWED_DIR/nosc-2026-08.sif \
  copernicusmarine login
```

Puis le téléchargement, année par année :

```bash
for y in $(seq 2010 2020); do
  echo "=== $y ==="
  singularity exec --bind $WORK:$WORK,$SCRATCH:$SCRATCH \
    $SINGULARITY_ALLOWED_DIR/nosc-2026-08.sif \
    python ../download_data_/import_data_glorys_multidepth.py $y 0 200
done
```

Le volume attendu se compte en dizaines à centaines de gigaoctets selon le
domaine et le nombre de niveaux. Vérifiez l'espace disponible avant de lancer :

```bash
idr_quota_user
du -sh $DATA_ROOT
```

Si la durée risque de dépasser la limite de la session interactive, soumettez
plutôt un job batch sur cette même partition, avec `#SBATCH -A yrf@cpu` et
`#SBATCH --partition=prepost`,
et surveillez-le avec `squeue`. La boucle année par année permet en outre de
reprendre là où elle s'est arrêtée.

### F.4 Sous-domaine et fusion

Le fichier attendu par le code — celui dont l'absence a fait échouer votre
premier essai — est le champ de surface servant de grille de référence :

```bash
singularity exec --bind $WORK:$WORK,$SCRATCH:$SCRATCH \
  $SINGULARITY_ALLOWED_DIR/nosc-2026-08.sif python - << 'PYEOF'
import xarray as xr, os
root = os.environ["DATA_ROOT"]
ds = xr.open_mfdataset(f"{root}/glorys_raw/*.nc", combine="by_coords")
gs = ds.sel(latitude=slice(32, 44), longitude=slice(-66, -54))
gs.isel(depth=0).to_netcdf(f"{root}/glorys_gs_surface_2010-2020.nc")   # grille de référence
gs.to_netcdf(f"{root}/glorys_gs_multidepth_2010-2020.nc")              # vérité 3D
PYEOF
```

Cette étape est gourmande en mémoire : restez sur le nœud de pré/post-traitement.

```bash
ls -lh $DATA_ROOT/*.nc
```

### F.5 ARGO virtuels

Géométrie réelle des profils, valeurs issues de GLORYS.

```bash
singularity exec --bind $WORK:$WORK,$SCRATCH:$SCRATCH \
  $SINGULARITY_ALLOWED_DIR/nosc-2026-08.sif \
  python -m contrib.argo.virtual \
    --grid-from $DATA_ROOT/glorys_gs_surface_2010-2020.nc \
    --truth-path $DATA_ROOT/glorys_gs_multidepth_2010-2020.nc \
    --start-date 2010-01-01 --end-date 2020-01-01 \
    --lat-min 32 --lat-max 44 --lon-min -66 --lon-max -54 \
    --depths 0.49 15 50 100 200 \
    --output-dir $DATA_ROOT/argo_virtual/gridded
```

**Point de vigilance.** Si cette étape récupère la géométrie des profils en
ligne, elle exige Internet — donc le nœud de pré/post-traitement — et se heurtera aux
avertissements `argopy` de l'image : une incompatibilité entre les versions
d'`argopy` et d'`erddapy` désactive les *fetchers* ERDDAP, ArgoVis et GDAC. Si
l'étape échoue pour cette raison, il faudra épingler des versions compatibles
dans le YAML et reconstruire l'image. Si la géométrie provient d'un fichier
local, la question ne se pose pas.

### F.6 Masques d'observation

Ils sont produits par les `entrypoints:` de la configuration, au premier
lancement de `main.py`. C'est l'étape qui échouait faute de la grille de
référence produite en F.4.

Pour les générer indépendamment, hors du protocole OSSE :

```bash
singularity exec --bind $WORK:$WORK,$SCRATCH:$SCRATCH \
  $SINGULARITY_ALLOWED_DIR/nosc-2026-08.sif \
  python -m contrib.synthetic_obs.build_masks \
    --grid-from $DATA_ROOT/glorys_gs_surface_2010-2020.nc \
    --n-days 3653 \
    --missions jason3 sentinel3a sentinel3b saral hy2b swot \
    --output $DATA_ROOT/masks_synthetic.pickle
```

### F.7 Tests fournis

```bash
singularity exec --bind $WORK:$WORK,$SCRATCH:$SCRATCH \
  $SINGULARITY_ALLOWED_DIR/nosc-2026-08.sif python tests/test_synthetic_obs.py

singularity exec --bind $WORK:$WORK,$SCRATCH:$SCRATCH \
  $SINGULARITY_ALLOWED_DIR/nosc-2026-08.sif python tests/test_argo_and_pseudo_obs.py
```

### F.8 Archiver

```bash
tar cf $STORE/nosc_data_2026-08.tar -C $SCRATCH/nosc data/
```

---

# G. Lancer un entraînement

### G.1 Smoke test interactif

Un GPU est **obligatoire**, même pour un test de deux itérations : le sélecteur
de canaux multi-variable appelle `.cuda()` en dur.

```bash
srun -A yrf@v100 --qos=qos_gpu-dev --gres=gpu:1 --cpus-per-task=10 \
     --time=00:30:00 --hint=nomultithread --pty bash
module load singularity
cd $WORK/NOSC/code/multivar_drifter_

singularity exec --nv --bind $WORK:$WORK,$SCRATCH:$SCRATCH \
  $SINGULARITY_ALLOWED_DIR/nosc-2026-08.sif \
  python -u main.py xp=$XP \
    paths.data_root=$DATA_ROOT \
    hydra.run.dir=$SCRATCH/nosc_runs/$(date +%Y%m%d-%H%M%S) \
    ++trainer.max_epochs=2 ++trainer.limit_train_batches=5 ++trainer.limit_val_batches=2
```

**`hydra.run.dir` n'est pas optionnel.** Par défaut, Hydra écrit dans
`outputs/<date>/<heure>/` relativement au répertoire courant, donc dans le
`$WORK` — et le job meurt sur un dépassement de quota avant la première époque,
avec une erreur qui ne mentionne que `os.mkdir`. L'horodatage est calculé par
`date` côté shell plutôt qu'avec la syntaxe `${now:…}` de Hydra, que bash
tenterait d'interpréter.

En cas d'échec, `SINGULARITYENV_HYDRA_FULL_ERROR=1` donne la trace complète : le
préfixe est nécessaire pour qu'une variable traverse le conteneur.

### G.2 Choisir la partition

`-A` désigne l'allocation débitée et conditionne l'accès ; `-C` désigne le type
de nœud. Le projet `yrf` a accès aux quatre familles.

| | V100 (défaut) | A100 (gpu_p5) | H100 (gpu_p6) |
|---|---|---|---|
| `-A` | `yrf@v100` | `yrf@a100` | `yrf@h100` |
| `-C` | rien, ou `v100-16g` / `v100-32g` | `a100` | `h100` |
| GPU par nœud | 4 | 8 | 4 |
| Mémoire par GPU | 16 ou 32 Go | 80 Go | 80 Go |
| `--cpus-per-task` | 10 | 8 | 24 |
| Durée maximale | 100 h | **20 h** | 100 h |
| Module préalable | aucun | `arch/a100` | `arch/h100` |
| QoS test | `qos_gpu-dev` | `qos_gpu_a100-dev` | `qos_gpu_h100-dev` |
| QoS standard | `qos_gpu-t3` | `qos_gpu_a100-t3` | `qos_gpu_h100-t3` |
| QoS longue | `qos_gpu-t4` | *(inexistante)* | `qos_gpu_h100-t4` |

**Recommandation : H100.** QoS longue disponible, 80 Go par GPU, processeurs
Intel compatibles avec la build MKL de l'image, 364 nœuds contre 52 en A100.
Éviter A100 : plafond de 20 heures et processeurs AMD moins favorables.

### G.3 Script de soumission

`$WORK/NOSC/jobs/train_h100.slurm` :

```bash
#!/bin/bash
#SBATCH --job-name=nosc_train
#SBATCH --account=yrf@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -x
cd $WORK/NOSC/code/multivar_drifter_

module purge
module load arch/h100        # IMPÉRATIF avant tout autre module sur gpu_p6
module load singularity

srun singularity exec --nv \
  --bind $WORK:$WORK,$SCRATCH:$SCRATCH \
  $SINGULARITY_ALLOWED_DIR/nosc-2026-08.sif \
  python -u main.py xp=osse3d_gs_multivar_unet \
    paths.data_root=$SCRATCH/nosc/data \
    hydra.run.dir=$SCRATCH/nosc_runs/$SLURM_JOB_ID
```

À ne pas modifier à la légère :

- `module load arch/h100` **avant** `singularity` : sur gpu_p5 et gpu_p6, les
  modules par défaut ne sont pas compatibles avec la partition.
- `--cpus-per-task=24` respecte le ratio cœurs/GPU du nœud H100. Le réduire
  étrangle les *dataloaders*, l'augmenter réserve inutilement le nœud entier.
- `python -u` désactive la bufférisation : sans cette option, les logs
  n'apparaissent qu'à la fin du job.
- `hydra.run.dir` pointe vers `$SCRATCH` avec `$SLURM_JOB_ID`, ce qui relie
  chaque sortie au job qui l'a produite.

### G.4 Soumettre et surveiller

```bash
cd $WORK/NOSC       # soumettre depuis la racine : les chemins des directives
mkdir -p logs       # #SBATCH sont relatifs au répertoire de soumission
sbatch jobs/train_h100.slurm

squeue -u $USER                      # état de la file
squeue -u $USER --start              # démarrage estimé
tail -f logs/nosc_train_<jobid>.out  # logs en direct
```

Codes d'état : `PD` en attente, `R` en cours, `CG` en fin. Si `PD` persiste, la
colonne `NODELIST(REASON)` en donne la cause.

### G.5 Après le job

```bash
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
ls $SCRATCH/nosc_runs/<jobid>/
```

`State=COMPLETED` et `ExitCode=0:0` signalent une fin normale. `TIMEOUT` signifie
que `--time` était trop court : reprenez depuis le dernier checkpoint.

Chaque run écrit ses logs CSV et ses checkpoints — les trois meilleurs sur
`val_total_mse` — dans le répertoire Hydra. Pour comparer plusieurs runs, lisez
les `metrics.csv` respectifs.

Pour enchaîner au-delà de 100 heures :

```bash
sbatch --dependency=afterok:<jobid> jobs/train_h100.slurm
```

---

# H. Inférence à partir d'un checkpoint

Pour reconstruire sans réentraîner, ajoutez dans la configuration :

```yaml
entrypoints:
  - _target_: src.train.base_training
    trainer: ${trainer}
    lit_mod: ${model}
    dm: ${datamodule}
    ckpt: /chemin/vers/checkpoint.ckpt
    only_rec: True
```

Sortie : un `test_data_dimN.nc` par dimension de sortie, dans l'ordre des entrées
`full_output` du bloc `multivar:`, contenant les champs `inp`, `tgt` et `out`.

---

# I. Pièges connus

**Les montages.** Seuls les répertoires liés par `--bind` sont visibles depuis le
conteneur. C'est la première cause de « fichier introuvable » sur un chemin
pourtant correct.

**Les sorties d'Hydra.** Sans `hydra.run.dir`, tout atterrit dans `outputs/` sous
le répertoire courant, donc dans le `$WORK`, et le job meurt sur le quota.

**Le HOME.** Monté automatiquement, il fait fuir caches et paquets locaux dans le
conteneur. `PYTHONNOUSERSITE=1` est défini dans l'image pour neutraliser
`~/.local`. En cas de comportement inexplicable, ajoutez `--no-home`.

**Pas d'Internet sur les nœuds de calcul, pas de conteneurs sur les frontales.**
Les étapes qui téléchargent passent par `--partition=prepost`. Sur une frontale,
`singularity exec` échoue sur `failed to resolve session directory`.

**GPU obligatoire**, y compris pour un smoke test de deux itérations.

**Jupyter et TensorBoard** ne peuvent pas être lancés depuis un conteneur
Singularity. Si votre usage passe par JupyterHub, conservez en parallèle un petit
environnement conda.

**Les restes de conda.** Le conteneur rend conda inutile ici : `rm -rf
$WORK/.conda/pkgs` libère souvent plus de cent mille inodes.

**`sst` n'est pas la température brute** : l'entrée historique est transformée en
`log|∇T|`, une caractéristique de fronts. Pour une vraie cible de température de
surface, utiliser `sst_tgt`.

**`thetao_argo_*` et `thetao_*`** sont deux cibles distinctes — ARGO épars contre
GLORYS dense — et non la même variable dédoublée.

**Ordre des `head_group`** : les variables d'un même groupe doivent être
contiguës dans `multivar:`, sinon une erreur explicite est levée.

**Configurations dépréciées.** Les `unet_uv_full_integration_*` relèvent d'un
protocole hybride abandonné ; la référence est `osse3d_gs_multivar_unet`.

**mpi4py** n'est pas dans l'image. L'IDRIS documente l'exécution de codes MPI
depuis un conteneur avec `--mpi=pmix_v2` ou `--mpi=pmix_v3`. À traiter seulement
si le calcul distribué devient nécessaire.

---

# J. Mettre à jour l'image

L'image est figée : toute modification du YAML impose de reconstruire et de
réimporter.

```bash
# poste local
apptainer build --fakeroot nosc-2026-09.sif nosc.def
rsync -avP nosc-2026-09.sif jz:/lustre/fswork/projects/rech/yrf/uag47zl/
# Jean Zay
idrcontmgr cp $WORK/nosc-2026-09.sif
idrcontmgr ls          # 20 conteneurs maximum
```

Taguez les images par date et versionnez `nosc.def`, `nosc-jeanzay.yaml` et ce
README avec le code. C'est ce qui permet de retrouver, six mois plus tard, quel
environnement a produit quel résultat.
