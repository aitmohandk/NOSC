# NOSC sur Datarmor — guide pas à pas

De la construction de l'image au premier entraînement, dans l'ordre. Chaque
étape se valide avant de passer à la suivante. Ce document décrit l'utilisation
complète du projet NOSC (Neural Ocean Surface Currents) sur le calculateur
**Datarmor** (Ifremer, Pôle de Calcul et de Données Marines, Brest).

**Contexte.** Le dépôt est supposé cloné dans `$DATAWORK/NOSC`, le code
d'expérience dans `$DATAWORK/NOSC/code/multivar_drifter_`.

**Trois points de fonctionnement de Datarmor** qui conditionnent tout le
reste :

- **L'ordonnanceur est PBS Pro.** On soumet et on suit les jobs avec `qsub`,
  `qstat`, `qdel` ; les directives de job dans les scripts commencent par
  `#PBS`. La queue de calcul GPU s'appelle **`gpuq`**.
- **Les conteneurs s'exécutent directement.** L'image Singularity (`.sif`) se
  dépose sur `$DATAWORK` et se lance depuis là avec `singularity exec` : aucun
  import, aucune validation, aucun répertoire imposé.
- **Les nœuds de calcul n'ont pas d'accès Internet.** Une seule catégorie de
  nœuds accède au réseau extérieur : ceux de la **queue `ftp`**. Tout
  téléchargement (GLORYS via `copernicusmarine`, profils ARGO via `argopy`,
  `rsync` depuis un serveur distant) doit être soumis sur cette queue.

> **Sur les sources et les points à confirmer.** Ce guide s'appuie sur la
> documentation communautaire publique de Datarmor (dépôt
> `umr-marbec/datarmor-documentation`, doc MARS3D, doc xsar/LOPS, guide
> D. Kaplan). La documentation officielle complète est sur l'intranet Ifremer
> (Portail Domicile → « Documentation calculateur »). Les quelques points
> encore marqués **[À CONFIRMER]** concernent la partie GPU, peu documentée
> publiquement : leur valeur se lit en deux minutes une fois connecté, avec la
> commande indiquée. En cas de doute : `assistance@ifremer.fr`, domaine
> « Calcul et données scientifiques ».

---

# A. Construire l'image (sur votre poste Ubuntu)

L'image se construit **sur votre machine personnelle**, pas sur Datarmor : la
construction d'une image Apptainer/Singularity à partir d'un fichier de
définition exige les droits root, dont vous ne disposez pas sur un calculateur
mutualisé. On construit donc en local, puis on transfère le fichier `.sif`
résultant (étape B).

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

Le fichier d'environnement Conda est `env/nosc-jeanzay.yaml`. Malgré son nom,
rien dedans n'est propre à un calculateur particulier : c'est simplement la
liste des dépendances de NOSC, réutilisable telle quelle. Vous pouvez le
dupliquer en `nosc-datarmor.yaml` si vous souhaitez faire diverger vos deux
environnements, en adaptant alors la ligne `%files` de `nosc.def`.

```bash
cd ~/NOSC/env          # impératif : %files lit le YAML dans le répertoire courant
grep -n "oceanbench\|loguru" nosc-jeanzay.yaml
```

Vous devez voir `- git+https://github.com/jejjohnson/oceanbench.git` et
**aucune** ligne `- oceanbench` seule : le paquet homonyme de PyPI est celui de
Mercator Ocean, sans rapport avec ce projet. Et `- loguru`, dépendance rangée
dans un groupe Poetry qu'un `pip install git+…` ignore.

**Point d'attention CUDA.** L'image est construite avec
`CONDA_OVERRIDE_CUDA="12.4"`. Vérifiez que le pilote NVIDIA des nœuds `gpuq`
supporte CUDA 12.4 (`nvidia-smi` en session interactive GPU, étape D.2 — la
version CUDA en haut à droite doit être ≥ 12.4). Si le pilote est plus ancien
**[À CONFIRMER selon la génération des nœuds GPU]**, abaissez
`CONDA_OVERRIDE_CUDA` et la contrainte de build PyTorch dans le YAML, puis
reconstruisez.

### A.3 Construire et valider localement

```bash
apptainer build --fakeroot nosc-$(date +%Y-%m).sif nosc.def
apptainer exec nosc-*.sif \
  python -c "import ocean4dvarnet, oceanbench, ocn_tools, xrpatcher, jaxparrow; print('TOUT OK')"
```

Comptez 30 à 40 minutes. Apptainer ne construit pas de façon incrémentale :
toute modification du YAML relance l'intégralité du processus.

---

# B. Transférer vers Datarmor

### B.1 Accès

L'hôte d'accès est **`datarmor-access.ifremer.fr`**.

- **Depuis le réseau Ifremer** : connexion SSH directe.

  ```bash
  ssh -X <login-intranet>@datarmor-access.ifremer.fr
  ```

  `-X` (ou `-Y` s'il échoue) active le déport d'affichage, pratique pour les
  éditeurs graphiques (`gedit`).

- **Depuis l'extérieur** : activer d'abord le VPN **Pulse Secure** (client
  téléchargeable sur le cloud Ifremer, connexion avec vos identifiants
  **extranet**), puis la même commande SSH avec vos identifiants **intranet**.

**Connexion sans mot de passe (recommandée), via clé ED25519 :**

```bash
ls $HOME/.ssh/id_ed25519.pub    # existe déjà ?
ssh-keygen -t ed25519           # sinon, la créer
ssh-copy-id <login>@datarmor-access.ifremer.fr
```

Configuration `~/.ssh/config` pratique :

```
Host datarmor
    HostName datarmor-access.ifremer.fr
    User <votre-login-intranet>
    ForwardX11 yes
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

### B.2 Transfert de l'image (~5 à 10 Go)

Le canal de transfert dépend d'où vous êtes, car les deux hôtes dédiés ne sont
pas joignables depuis les mêmes réseaux. **Une erreur
`connect to host … port 22: Connection timed out` ne signale pas un mauvais
mot de passe : elle signifie que l'hôte visé n'est pas routé depuis votre
position** — typiquement `datacopy` depuis l'extérieur, même avec le VPN.

**Depuis un bâtiment Ifremer (ou un VPN routant tout le réseau interne).**
L'hôte dédié est **`datacopy.ifremer.fr`**, qui accepte **scp / sftp / rsync**
(transport SSH, port 22) avec vos identifiants **intranet**. Vérifiez d'abord
qu'il répond :

```bash
ssh <login>@datacopy.ifremer.fr    # doit demander le mot de passe, pas timeouter
```

S'il répond, transférez (dépôt par défaut dans votre `$HOME` Datarmor) :

```bash
rsync -avP nosc-2026-09.sif <login>@datacopy.ifremer.fr:
```

Le client graphique **FileZilla** (protocole SFTP) fait la même chose. Si
`datacopy` timeoute alors que `datarmor-access` répond, votre VPN ne route pas
cet hôte : passez à la méthode extérieure.

**Depuis l'extérieur (domicile).** La voie prévue passe par
**`eftp.ifremer.fr`**, qui parle **FTP** avec vos identifiants **extranet** et
expose le répertoire `$SCRATCH/eftp` de votre compte Datarmor.

> **Attention, le compte extranet est distinct du compte Datarmor.** Ce n'est
> ni le même mot de passe que votre login intranet/Datarmor (Ifremer interdit
> même de les faire identiques), ni forcément un compte déjà actif. Une erreur
> `530 Login incorrect` sur `eftp` vient presque toujours de là. Le compte
> extranet se crée depuis `teletravail.ifremer.fr` (identifiants envoyés sur
> votre mail Ifremer) ; le mot de passe se réinitialise sur
> `https://www.ifremer.fr/chpass/`. En cas de blocage, la solution de repli
> ci-dessous évite complètement ce compte.

En trois temps :

```bash
# 1. Créer le répertoire d'atterrissage, depuis une session SSH sur Datarmor
ssh datarmor
mkdir -p $SCRATCH/eftp
exit

# 2. Depuis votre poste, envoyer le .sif en FTP (identifiants EXTRANET).
#    lftp en ligne de commande, ou FileZilla en protocole FTP :
lftp -u <login-extranet> eftp.ifremer.fr
#   cd eftp
#   put nosc-2026-09.sif
#   bye
```

**Solution de repli, quelle que soit la position.** Vous atteignez déjà
`datarmor-access.ifremer.fr` en SSH (c'est votre hôte de connexion) : vous
pouvez y déposer le `.sif` directement. À réserver aux transferts ponctuels —
la frontale est un nœud partagé, à ne pas surcharger de gros transferts
répétés.

```bash
rsync -avP nosc-2026-09.sif <login>@datarmor-access.ifremer.fr:containers/
```

Dans tous les cas, rangez ensuite l'image à demeure sur `$DATAWORK` (le
`$SCRATCH` et le `$HOME` ne sont pas faits pour ça ; le `$SCRATCH` est de plus
purgé automatiquement, voir E) :

```bash
ssh datarmor
mkdir -p $DATAWORK/containers
mv $SCRATCH/eftp/nosc-2026-09.sif $DATAWORK/containers/   # voie extérieure
# ou :  mv ~/nosc-2026-09.sif $DATAWORK/containers/       # voie datacopy (dépôt dans $HOME)
```

`rsync -avP` affiche la progression et reprend un transfert interrompu, ce qui
est appréciable sur plusieurs gigaoctets ; `scp` reste silencieux jusqu'à la
fin.

---

# C. Rendre Singularity disponible

L'image s'exécute directement depuis `$DATAWORK`, sans étape préalable
d'import. Il suffit de charger le module qui fournit la commande `singularity`.

```bash
ssh datarmor
module avail singularity        # repérer le nom exact du module [À CONFIRMER]
module load singularity
singularity --version
```

Si aucun module `singularity`/`apptainer` n'apparaît dans `module avail`,
demandez-le à l'assistance : l'outil est utilisé par plusieurs équipes
(pipelines Nextflow de SeBiMER notamment), il est disponible sur le
calculateur, mais son mode de mise à disposition peut évoluer.

> **Le shell par défaut de Datarmor est csh/tcsh.** C'est un point structurant.
> Les commandes *interactives* de ce guide sont écrites en **bash** (avec
> `export`, boucles `for … do … done`, heredocs `<< EOF`) : tapez `bash` en
> début de session, ou faites changer votre shell de connexion par
> l'assistance. Les **scripts de job PBS**, eux, sont écrits en `#!/bin/csh`,
> qui est la convention locale illustrée par toute la documentation Datarmor
> (chargement des modules via `source /usr/share/Modules/…/init/csh`,
> `setenv`, `foreach … end`). Les deux styles cohabitent sans problème : le
> shell de votre session et le shebang d'un script soumis sont indépendants.
> Une variante bash est fournie en commentaire dans les scripts si vous
> préférez tout uniformiser.

Pour charger le module automatiquement à chaque connexion, ajoutez la ligne
`module load singularity` à votre `~/.cshrc` (ou `~/.bashrc` si vous êtes passé
en bash).

---

# D. Vérifier le conteneur

**Ne faites rien de lourd sur la frontale** (le nœud de login sur lequel vous
arrivez) : elle est partagée et réservée à la navigation, l'édition de
fichiers et la compilation. Aucun calcul, aucune manipulation de gros fichiers.
Toute exécution du conteneur passe par une session sur un nœud de calcul,
obtenue avec `qsub`.

> **Interactif ou batch ?** Deux façons de demander un nœud, à choisir selon la
> tâche :
>
> - **Interactif** (`qsub -I …`) : vous récupérez un shell sur le nœud une fois
>   la ressource allouée. Pratique pour les vérifications rapides, le debug, un
>   test de quelques minutes. Inconvénient : le terminal **reste figé en
>   attente** (`qsub: waiting for job … to start`) tant que la file n'a pas
>   accordé de nœud, et le job meurt si vous fermez la connexion.
> - **Batch** (`qsub script.pbs`) : le job part en arrière-plan, vous rendez la
>   main immédiatement, vous pouvez fermer le terminal, et vous suivez
>   l'avancement avec `qstat -u $USER` puis `tail -f` sur le fichier de sortie.
>   **À préférer pour toute étape longue** — préparation des données (F),
>   entraînement (G).
>
> Convertir un exemple interactif de ce guide en batch est mécanique : mettez
> les mêmes ressources en directives `#PBS` en tête d'un fichier `.pbs`, puis
> la même commande `singularity exec …` dans le corps (voir les modèles fournis
> en F.3, F.4 et G.3). Soumettez avec `qsub monscript.pbs`.
>
> **Piège du batch csh : `module: Command not found`.** Un job batch ne charge
> pas votre `~/.cshrc`, donc la fonction `module` n'y est pas définie d'office.
> Chaque script `.pbs` doit d'abord **sourcer l'init des modules** avant tout
> `module load` :
>
> ```csh
> source /usr/share/Modules/init/csh   # [À CONFIRMER : chemin exact de l'init]
> module load singularity
> ```
>
> Si ce chemin n'existe pas sur votre instance, `ls /usr/share/Modules/init/`
> (ou `ls /etc/profile.d/*module*`) donne le bon. Les modèles `.pbs` de ce guide
> incluent déjà cette ligne. (En session **interactive**, inutile : le shell de
> login a déjà chargé l'init.)
>
> **Piège de routage (attente anormalement longue).** La queue par défaut
> `sequentiel` est une *queue de routage* : elle ne fait tourner aucun job
> elle-même, elle redirige vers une queue d'exécution selon les ressources
> demandées. Une grosse demande de mémoire peut vous router vers des nœuds à
> très grosse RAM, peu nombreux et donc très disputés (p. ex. `ice_1t` = nœuds
> 1 To) — d'où une longue attente. Demandez **seulement ce dont vous avez
> besoin** : pour un test, `mem=16g` part généralement plus vite que `mem=64g`.
> Pour diagnostiquer une attente, depuis un second terminal :
> `qstat -f <jobid> | grep -i "job_state\|queue\|comment\|estimated"` (le champ
> `comment` donne souvent la raison), et `qstat -Q` liste l'état de toutes les
> queues.

### D.1 Session interactive GPU

```bash
qsub -I -q gpuq -l walltime=00:30:00 -l mem=64g
module load singularity
```

`-I` demande une session interactive ; vous êtes déposé sur le nœud une fois la
ressource allouée, et vous la libérez avec `exit`. Laissez la connexion ouverte
pendant toute la durée de la session interactive.

La façon exacte de réserver **le GPU et le nombre de cœurs** dans la queue
`gpuq` (via `ngpus=`/`ncpus=` dans un `-l select=…`, ou par attribution
implicite de la queue) est le point le moins documenté publiquement — **[À
CONFIRMER : voir G.2]**. Commencez avec la commande simple ci-dessus ; si le
GPU n'est pas visible en D.2, ajoutez la sélection explicite :

```bash
qsub -I -q gpuq -l select=1:ncpus=8:ngpus=1:mem=64g -l walltime=00:30:00
```

### D.2 Le GPU est-il visible ?

```bash
nvidia-smi          # note la version CUDA max supportée par le pilote

singularity exec --nv --bind $DATAWORK,$SCRATCH \
  $DATAWORK/containers/nosc-2026-09.sif \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Attendu : `True NVIDIA ...`. Si `False`, c'est l'option `--nv` manquante — ou
un pilote hôte trop ancien pour la build CUDA de l'image (voir A.2).

### D.3 Les imports du projet passent-ils ?

```bash
singularity exec --nv --bind $DATAWORK,$SCRATCH \
  $DATAWORK/containers/nosc-2026-09.sif \
  python -c "import ocean4dvarnet, oceanbench, ocn_tools, xrpatcher, jaxparrow; print('imports OK')"
```

Sortez avec `exit`.

---

# E. Organiser les espaces de stockage

Datarmor expose quatre espaces aux caractéristiques très différentes. Les
confondre est la première cause de perte de données ou de saturation de quota.

| Espace | Volume | Sauvegarde | Purge | Usage |
|---|---|---|---|---|
| `$HOME` | 50 Go | **oui** | non | codes, scripts, configuration, petits fichiers importants |
| `$DATAWORK` | 1 To | **non** | non | données, image `.sif`, copie maître des jeux préparés |
| `$SCRATCH` | 10 To | non | **fichiers de plus de 10 jours supprimés** | données de travail, sorties, checkpoints en cours de run |
| `/home/ref-<thématique>/…` | — | — | non | données de référence Ifremer, lecture seule (`ocean-reanalysis`, `argo`, `ecmwf`, `remote-sensing-public`, `cersat-public`…) |

Les emplacements exacts se lisent avec `echo $DATAWORK`, etc. Un fichier
supprimé par erreur sur `$HOME` peut être restauré via `assistance@ifremer.fr`.

```bash
export NOSC_DATA_ROOT=$SCRATCH/nosc/data
mkdir -p $NOSC_DATA_ROOT $SCRATCH/nosc_runs
# en bash, ajout à ~/.bashrc :
echo 'export NOSC_DATA_ROOT=$SCRATCH/nosc/data' >> ~/.bashrc
# en csh, ajouter plutôt à ~/.cshrc :  setenv NOSC_DATA_ROOT $SCRATCH/nosc/data
```

**Se prémunir de la purge du `$SCRATCH`.** Tout fichier de plus de **10 jours**
sur `$SCRATCH` est supprimé automatiquement. C'est l'espace le plus grand et le
plus rapide — donc celui où l'on calcule — mais il ne conserve rien à moyen
terme. Deux règles en découlent :

- gardez la **copie maître** des données préparées sur `$DATAWORK` (voir F.9) ;
- **rapatriez checkpoints et métriques** de `$SCRATCH/nosc_runs` vers
  `$DATAWORK` après chaque campagne de runs (voir G.5). Ne laissez jamais
  l'unique exemplaire d'un résultat sur `$SCRATCH` : dix jours sans y toucher
  et il disparaît.

La bonne pratique d'exécution locale en découle directement (voir F.3 et
G.3) : on copie code et données sur `$SCRATCH`, on lance le calcul depuis là,
puis on recopie les sorties vers `$DATAWORK`.

**L'espace « données de référence » (DATAREF) de Datarmor contient
vraisemblablement la plupart des données de NOSC — vérifiez-le avant de
télécharger quoi que ce soit.** DATAREF est un espace **en lecture seule**,
accessible directement depuis les nœuds de calcul et les scripts PBS (pas
besoin de la queue `ftp`, pas de téléchargement). Les données y sont rangées
par thématique, sous :

```
/home/ref-<thématique>/ifremer/<unité>/<projet>            # données privées (intranet)
/home/ref-<thématique>-public/ifremer/<unité>/<projet>     # données publiques
```

Le manuel du groupe Données d'Ifremer liste les thématiques hébergées. Sur
cette instance de Datarmor, les données de NOSC se trouvent aux chemins
suivants (confirmés par `ls /home/ref-*`) :

| Donnée NOSC | Chemin DATAREF confirmé |
|---|---|
| GLORYS12V1 (thetao, uo, vo, zos) | `/home/ref-ocean-reanalysis/global-reanalysis-phy-001-030-daily` (+ `.../glorys_native`) |
| Bathymétrie | `/home/ref-ocean-model-public/EMODNET_BATHY` (ou champ `static` GLORYS, à chercher dans `ocean-reanalysis`) |
| Profils ARGO | `/home/ref-argo/gdac` (GDAC natif), `/home/ref-coriolis-public/argo`, `/home/ref-ISAS/ANA_ISAS20_ARGO` |
| SST L4 (OSTIA) | `/home/ref-remote-sensing-public/METOFFICE-GLO-SST-L4-REP-OBS-SST` |
| SST (CERSAT) | `/home/ref-cersat-public/sea-surface-temperature` |
| SSH L4 (altimétrie) | `/home/ref-remote-sensing-public/altimetry`, `/home/ref-cersat-public/ocean-topography` |
| Chlorophylle L3 | `/home/ref-remote-sensing-public/GLOBCOLOUR_L3m_CHL1_MO` (+ `ESACCI-OC-L3S-CHLOR_A-MERGED-1M`) |
| Vent ERA5 | `/home/ref-ecmwf/ERA5` |

Attention : la thématique `cmems` n'expose qu'un dossier `tac` quasi vide sur
cette instance — GLORYS est bien sous `ocean-reanalysis`, pas sous `cmems`. La
convention générale reste `/home/ref-<thématique>/…` en lecture seule (souvent
avec des variantes `-public`/`-intranet`).

Il reste à descendre dans chaque dossier pour l'arborescence fine et à vérifier
la couverture, une fois connecté :

```bash
ls /home/ref-ocean-reanalysis/global-reanalysis-phy-001-030-daily/   # GLORYS
ls /home/ref-argo/gdac/                                              # ARGO GDAC natif
ls /home/ref-ecmwf/ERA5/                                             # ERA5
ls /home/ref-remote-sensing-public/                                  # SST/SSH/chl
```

Pour GLORYS, vérifiez que le produit couvre le domaine Gulf Stream (lat
32–44, lon −66 à −54), la période 2010–2020 et les 5 niveaux de profondeur
(`ncdump -h <un_fichier>.nc | head`).
Si GLORYS y est présent, l'étape F.3 se réduit à pointer le script de
préparation sur ce chemin (aucun téléchargement). **Nuance importante** :
DATAREF n'offre pas de garantie de sauvegarde absolue (le producteur reste
responsable de l'archivage), donc pour une expérience reproductible sur le long
terme, gardez sous `$DATAWORK` la copie sous-domainée que vous produisez à
partir de ces données — c'est de toute façon ce que fait F.3.

**Surveiller le nombre de fichiers autant que le volume.** Un jeu de données
découpé en un fichier NetCDF par jour et par capteur produit des dizaines de
milliers de petits fichiers qui dégradent le système de fichiers parallèle pour
tous les utilisateurs. Privilégiez l'agrégation, avec des chunks de quelques
dizaines à quelques centaines de mégaoctets alignés sur le motif de lecture —
pour du 4DVarNet, des patchs spatiaux parcourus dans le temps.

---

# F. Préparer les données

**C'est l'étape la plus longue, et elle doit précéder tout entraînement.** Le
pipeline reconstruit des champs océaniques denses à partir d'observations
éparses : il lui faut une vérité GLORYS, une bathymétrie, des ARGO virtuels,
et des masques d'observation.

Le protocole de référence est l'OSSE Gulf Stream, avec vérité GLORYS de bout
en bout. Domaine : latitude 32 à 44, longitude −66 à −54.

### F.1 Valider la composition de la configuration

À faire **en premier** : cela ne demande ni GPU ni données, et révèle
immédiatement les chemins à corriger. Une courte session interactive CPU
suffit (jamais sur la frontale) :

```bash
qsub -I -l walltime=00:30:00 -l mem=8g
module load singularity
cd $DATAWORK/NOSC/code/multivar_drifter_
export XP=osse3d_gs_multivar_unet        # config de référence du protocole OSSE

singularity exec --bind $DATAWORK,$SCRATCH \
  $DATAWORK/containers/nosc-2026-09.sif \
  python main.py xp=$XP --cfg job --resolve
```

Notez tous les chemins affichés : ce sont les fichiers que les étapes
suivantes doivent produire.

### F.2 Variables de session : `NOSC_DATA_ROOT` et `NOSC_TARGET_RES`

Deux variables d'environnement pilotent toute la préparation. À définir **une
fois par session** (ou dans votre `.bashrc`/`.cshrc`).

**`NOSC_DATA_ROOT` (obligatoire)** — où sont lus/écrits les fichiers préparés.
Depuis le patch de portabilité, les configs OSSE (`config/xp/osse3d_gs*.yaml`)
ne contiennent plus de chemin en dur : `paths.data_root` vaut
`${oc.env:NOSC_DATA_ROOT}`, et tous les chemins dérivés (GLORYS, bathy, masques,
pseudo-obs) s'en déduisent — plus aucun `paths.data_root=…` à passer en ligne de
commande. Si la variable n'est pas définie, le lancement échoue immédiatement
avec une erreur explicite plutôt que de pointer un chemin fantôme.

**`NOSC_TARGET_RES` (facultatif)** — la résolution cible en degrés (p. ex.
`0.25`). Si définie, `prepare_glorys_osse.py` interpole GLORYS sur une grille
régulière à ce pas ; sinon la résolution native 1/12° (~0,083°) est conservée.

```bash
export NOSC_DATA_ROOT=$SCRATCH/nosc/data       # obligatoire
export NOSC_TARGET_RES=0.25                     # facultatif (défaut : natif 1/12°)
# en csh :  setenv NOSC_DATA_ROOT $SCRATCH/nosc/data ; setenv NOSC_TARGET_RES 0.25
```

> **La règle d'or de la cohérence de grille.** Tous les jeux préparés — vérité
> GLORYS, ARGO virtuels, masques, pseudo-obs — doivent vivre sur **exactement la
> même grille**, sinon les entrées et cibles du réseau n'ont pas les mêmes
> dimensions. Le code garantit ça structurellement : **un seul fichier définit
> la grille** (le GLORYS préparé), et tous les autres producteurs la lisent via
> `--grid-from`/`--truth-path`. Concrètement : on regrille GLORYS une fois
> (F.3), puis on pointe tous les `--grid-from` des étapes suivantes (F.6, F.7)
> vers ce fichier. L'option `--expect-res` de ces étapes est un garde-fou qui
> refuse de tourner si la grille reçue n'a pas la résolution attendue — utilisez-la.

Vous pouvez vérifier la résolution de la config avec `--cfg job --resolve` (F.1).

### F.3 GLORYS — voie primaire : le miroir DATAREF

Depuis le patch d'accès aux données, la préparation de GLORYS passe par un seul
script, `download_data_/prepare_glorys_osse.py`, dont la **source est
paramétrable**. La même commande fonctionne que GLORYS vienne du miroir local
Datarmor ou d'un téléchargement — seul change le répertoire source. On privilégie
le miroir local.

**Étape 1 — localiser GLORYS dans DATAREF.** GLORYS12V1 est un produit CMEMS
central ; il est très probablement mirroré sous une des thématiques de
référence (voir E). Repérez le dossier exact :

```bash
ls /home/ref-ocean-reanalysis/
# GLORYS12V1 est ici (produit 001-030 utilisé par NOSC) :
ls /home/ref-ocean-reanalysis/global-reanalysis-phy-001-030-daily/
```

Vérifiez que le produit couvre le domaine (lat 32–44, lon −66 à −54), la période
(2010–2020) et contient bien les variables `thetao`, `uo`, `vo`, `zos` sur
plusieurs niveaux de profondeur (`ncdump -h <un_fichier>.nc | head -50`).

**Étape 2 — préparer, en pointant la source sur DATAREF.** Une session CPU avec
mémoire large suffit (pas de GPU, pas de réseau). En **interactif**, pour un
premier test sur un mois (démarre vite, valide la sortie) :

```bash
qsub -I -l walltime=00:30:00 -l mem=32g
module load singularity
cd $DATAWORK/NOSC/code/download_data_
export NOSC_DATA_ROOT=$SCRATCH/nosc/data      # où seront écrits les fichiers préparés
export NOSC_TARGET_RES=0.25                    # facultatif : résolution cible (défaut natif 1/12°)

singularity exec --bind $DATAWORK,$SCRATCH,/home/ref-ocean-reanalysis \
  $DATAWORK/containers/nosc-2026-09.sif \
  python -u prepare_glorys_osse.py \
    --src /home/ref-ocean-reanalysis/global-reanalysis-phy-001-030-daily \
    --start 2010-01-01 --end 2010-02-01
```

**Chronométrez ce mois** avant de lancer l'année entière ou l'array : il valide
la sortie *et* donne l'échelle de temps pour dimensionner le walltime batch. Le
`python -u` force une sortie non bufferisée, indispensable en batch pour voir la
progression dans le `.o` (sans lui, Python bufferise la sortie tant qu'il n'est
pas attaché à un terminal, et un job tué sur walltime laisse un log vide).

`--target-res` peut aussi être passé en option explicite
(`--target-res 0.25`) plutôt que par la variable d'environnement. Sans l'un ni
l'autre, GLORYS est préparé à sa résolution native 1/12°.

Pour les **11 ans complets** (long : préférez le **batch**, qui ne bloque pas le
terminal), on ne lance **pas** un seul job monolithique sur toute la période. Un
job unique enchaîne ~4000 ouvertures de fichiers de 15 Go sur un seul cœur sans
rien sauver en cours de route : au moindre dépassement de walltime tout est
perdu (c'est le piège qui faisait échouer l'ancienne version — voir l'encadré
plus bas). On découpe donc par année en **job array**, chaque sous-job traitant
une année indépendamment, puis on concatène. `jobs/prepare_glorys.pbs` :

```csh
#!/bin/csh
#PBS -N glorys_prep
#PBS -q omp
#PBS -l select=1:ncpus=8:mem=32g
#PBS -l walltime=03:00:00
#PBS -J 2010-2020

source /usr/share/Modules/init/csh   # indispensable en batch csh : définit `module`
module load singularity
cd $DATAWORK/NOSC/code/download_data_
setenv NOSC_DATA_ROOT $SCRATCH/nosc/data/by_year   # sorties par année (fusionnées ensuite)
setenv NOSC_TARGET_RES 0.25                        # facultatif ; retirer pour le natif 1/12°
setenv PYTHONUNBUFFERED 1

set YEAR = $PBS_ARRAY_INDEX
@ NEXT = $YEAR + 1

singularity exec --bind $DATAWORK,$SCRATCH,/home/ref-ocean-reanalysis \
  $DATAWORK/containers/nosc-2026-09.sif \
  python -u prepare_glorys_osse.py \
    --src /home/ref-ocean-reanalysis/global-reanalysis-phy-001-030-daily \
    --start ${YEAR}-01-01 --end ${NEXT}-01-01 --period-label $YEAR
```

Chaque sous-job écrit ses propres fichiers `glorys_gs_*_<année>.nc` sous
`$SCRATCH/nosc/data/by_year`. Une année qui dépasse son walltime se **relance
seule** (`qsub -J 2015-2015 …`), les autres restant acquises. On fusionne
ensuite les fichiers annuels en les deux fichiers consolidés attendus par la
config. Le travail de fusion vit dans un vrai fichier
`code/download_data_/concat_glorys.py` (voir plus bas pourquoi ce n'est pas du
Python en ligne), appelé par `jobs/concat_glorys.pbs` :

```csh
#!/bin/csh
#PBS -N glorys_concat
#PBS -q omp
#PBS -l select=1:ncpus=4:mem=32g
#PBS -l walltime=04:00:00

source /usr/share/Modules/init/csh
module load singularity
set BY_YEAR = $SCRATCH/nosc/data/by_year
set OUT     = $DATAWORK/nosc/data          # durable, non purgé
set LABEL   = 2010-2020

singularity exec --bind $DATAWORK,$SCRATCH \
  $DATAWORK/containers/nosc-2026-09.sif \
  python -u $DATAWORK/NOSC/code/download_data_/concat_glorys.py $BY_YEAR $OUT $LABEL
```

> **Pourquoi un fichier `.py` et pas un `python -c "…"` en ligne.** Les jobs
> tournent sous **csh**, qui ne sait pas porter une chaîne entre guillemets
> doubles sur plusieurs lignes : un programme Python multi-ligne dans
> `python -c "…"` casse à la soumission avec `Unmatched "`. Un fichier `.py`
> appelé avec ses arguments (`… concat_glorys.py $BY_YEAR $OUT $LABEL`) évite
> tout ce problème de quoting. `concat_glorys.py` crée son répertoire de sortie,
> fusionne les fichiers annuels `surface`/`multidepth` et écrit
> `glorys_gs_surface_2010-2020.nc` et `glorys_gs_multidepth_2010-2020.nc`.
>
> Deux détails de performance : la fusion **charge en mémoire** (`.load()`) avant
> d'écrire — imposer un `chunks={"time": 30}` refragmentait la lecture des
> fichiers annuels et faisait streamer dask à travers un graphe morcelé, ce qui
> poussait la fusion au-delà du walltime ; une boîte Gulf Stream sous-domainée et
> réduite en profondeur tient de toute façon en quelques Go de RAM. Et le script
> est **idempotent** : un fichier de sortie déjà présent est ignoré, donc après
> un kill sur walltime, relancer ne refait que ce qui manque (supprimez le
> fichier à la main pour forcer une reconstruction).

Soumission, suivi, et enchaînement automatique de la concaténation après
l'array (le `-W depend` ne lance `concat` que si toutes les années réussissent) :

```csh
set A = `qsub jobs/prepare_glorys.pbs`               # rend la main aussitôt ; garde le jobid
echo $A                                              # doit afficher p.ex. 1234567[].datarmor3
qsub -W "depend=afterokarray:$A" jobs/concat_glorys.pbs
qstat -tu $USER                                      # suivre l'array (Q puis R, sous-job par sous-job)
tail -f glorys_prep.o<jobid>.<index>                 # suivre une année une fois partie
```

> **Les guillemets autour de `"depend=afterokarray:$A"` sont indispensables sous
> csh** : l'ID d'un job array contient des crochets `[]` que csh interpréterait
> comme un motif de fichiers, ce qui corrompt l'argument et donne
> `qsub: illegal -W value`. Vérifiez toujours `echo $A` avant : si la variable
> est vide (nouveau shell, ou array non soumis dans CE shell), la dépendance est
> vide et rejetée pour la même raison. En cas de doute, **oubliez la dépendance**
> et lancez `concat` à la main une fois l'array terminé (`qstat -tu $USER` ne
> montre plus de sous-job `R`/`Q`) : `qsub jobs/concat_glorys.pbs`. Les fichiers
> annuels étant sur `$SCRATCH`, c'est sans risque.

> **File et mémoire (vérifié via `qstat -Qf`).** Ce job est mono-nœud
> multi-cœurs non-MPI : la file est **`omp`** (nœud partagé, jusqu'à 28 cœurs).
> On la nomme explicitement (`#PBS -q omp`). **Attention à la mémoire** : le
> nœud offre ~125 Go pour 28 cœurs (~4,5 Go/cœur), et les seules autres files
> ouvertes acceptant `ncpus=8` (`ftp`, `visuq`, `top`) plafonnent à 50–60 Go ;
> demander `mem=64g` sur 8 cœurs fait rejeter le job par **toutes** les
> destinations (`qsub: Job rejected by all possible destinations`). D'où
> `mem=32g` (4 Go/cœur). Pour vérifier sur votre instance :
>
> ```bash
> qstat -Qf | grep -iE 'Queue:|resources_max.(mem|ncpus)|walltime'
> pbsnodes -a | grep -m1 'resources_available.mem'
> ```
>
> Si `#PBS -J` (array) est refusé sur cette file, repliez-vous sur une boucle de
> `qsub` par année (année passée via `-v YEAR=2010`, lue avec
> `setenv YEAR $YEAR` dans le script).

> **Pourquoi l'ancienne version dépassait le walltime.** Le script d'origine
> regriddait les **50 niveaux natifs** avant de n'en garder que 5 : ~10× de
> lecture et de calcul inutiles. Le script corrigé sélectionne les 5 niveaux
> **dans le `preprocess`**, donc *avant* le regriddage et poussé jusqu'à la
> lecture NetCDF (seuls les 5 niveaux utiles sont décompressés), ouvre les
> fichiers en parallèle (`parallel=True`), et écrit sa progression sans
> bufferisation. Combiné au découpage par année sur 8 cœurs, une année passe
> largement sous les 3 h. Si un mois de test reste lent malgré tout, regardez le
> chunking disque des fichiers sources (`ncdump -hs <fichier>.nc | grep
> _ChunkSizes`) : des chunks couvrant toute la grille horizontale rendent la
> lecture d'une petite boîte coûteuse.

Le script explore l'arborescence sous `--src` (ici `année/mois/*.nc`, le layout
du miroir Datarmor), n'ouvre **que les années couvrant la période demandée** (pas
les 30+ ans de l'archive), sous-domaine chaque fichier à la boîte Gulf Stream à
l'ouverture, réduit aux 5 niveaux utiles et applique le regriddage si
`--target-res`/`NOSC_TARGET_RES` est défini, puis écrit une paire de fichiers
`glorys_gs_surface_<label>.nc` / `glorys_gs_multidepth_<label>.nc` sous
`$NOSC_DATA_ROOT`. Le `<label>` vient de `--period-label` (par défaut les années
de `--start`/`--end`) : avec le job array ci-dessus, c'est l'année, d'où les
fichiers annuels que `concat_glorys.pbs` fusionne ensuite en
`glorys_gs_surface_2010-2020.nc` et `glorys_gs_multidepth_2010-2020.nc`, les
noms exacts attendus par la config. **Ces fichiers définissent la grille de
référence de tout le pipeline** (voir la règle d'or en F.2). Le même script gère
aussi un répertoire plat (le `glorys_raw/` d'un download) ou un fichier unique,
et — pour un jeu assez petit — la période complète en une seule invocation
(`--start 2010-01-01 --end 2020-01-01`) si vous préférez éviter l'array.

Chaque fichier GLORYS global journalier pèse ~15 Go (grille 4320×2041, 50
niveaux, `float64`) — d'où trois réductions appliquées par défaut, sans quoi la
sortie serait ingérable : sous-domainage spatial à la boîte GS (fait à
l'ouverture), réduction aux **5 niveaux de profondeur** utiles (0.494/15/50/100/200
m, au lieu des 50 natifs), passage en **`float32`** et **compression** NetCDF.
Résultat : des fichiers de sortie de quelques gigaoctets. Un `--target-res`
(regriddage) réduit encore davantage sur l'horizontal (0,25° divise le nombre de
points par ~9 vs le 1/12°). Options pour revenir en arrière si besoin :
`--keep-all-depths`, `--no-float32`, `--depths …` (voir `--help`).

Les quatre leviers de données sont ainsi réunis dans ce script : **zone**
(`--lat-min/max`, `--lon-min/max`), **profondeur** (`--depths`,
`--keep-all-depths`), **période** (`--start/--end`) et **résolution**
(`--target-res` / `NOSC_TARGET_RES`). Élargir la zone ou changer la résolution
suppose de relancer cette préparation, puis d'ajuster en cohérence le bloc
`domain:` et les `patch_dims` de la config Hydra.

**Format de sortie NetCDF ou Zarr (`--format`).** Par défaut, chaque sortie est
un fichier NetCDF unique (`.nc`) — parfait jusqu'à quelques dizaines de Go. Pour
une zone étendue (grand bassin, global) où un `.nc` monolithique ne tient plus,
`--format zarr` écrit à la place un **store Zarr chunké** (`.zarr`), en blocs
temporels (`--zarr-time-chunk`, défaut 30 jours). Le data loader lit
**indifféremment** l'un ou l'autre, choisi d'après l'extension du chemin : il
suffit de pointer les `paths.*` de la config vers les `.zarr` au lieu des `.nc`.
Rien d'autre ne change dans la config ni le code.

```bash
# préparation en Zarr (ex. zone large, 1° pour commencer)
python prepare_glorys_osse.py --src /home/ref-ocean-reanalysis/global-reanalysis-phy-001-030-daily \
    --lat-min -80 --lat-max 90 --lon-min -180 --lon-max 180 \
    --target-res 1.0 --format zarr
# puis dans config/xp/…yaml :
#   glorys_multidepth: ${paths.data_root}/glorys_gs_multidepth_2010-2020.zarr
```

> **Rappel d'échelle et de stockage.** À résolution native 1/12° en 3D, une zone
> globale se compte en dizaines de téraoctets — bien au-delà du quota `$SCRATCH`
> (10 To). Zarr rend le jeu *lisible* par chunks, mais ne résout pas le
> stockage : pour du global 3D, commencez à basse résolution (`--target-res 1.0`)
> et/ou période courte, et prévoyez un espace projet dédié (assistance Ifremer)
> avant tout passage à l'échelle. Le `--target-res 1.0` réduit déjà le volume
> horizontal d'un facteur ~140 par rapport au 1/12°.

> **N'oubliez pas de monter le dossier DATAREF dans `--bind`.** C'est la cause
> n°1 de « fichier introuvable » : `/home/ref-ocean-reanalysis` doit figurer
> explicitement dans `--bind`, en plus de `$DATAWORK` et `$SCRATCH`.

Le domaine, la période et les noms de fichiers sont ajustables
(`python prepare_glorys_osse.py --help`) : `--lat-min/--lat-max/--lon-min/--lon-max`,
`--start/--end`, `--period-label`. Les valeurs par défaut correspondent au
protocole OSSE Gulf Stream de référence.

### F.4 GLORYS — voie de repli : téléchargement CMEMS

À n'utiliser **que si GLORYS n'est pas dans DATAREF** sur le domaine/période
voulus. Le téléchargement passe par le script `import_data_glorys_multidepth.py`
et **doit se faire sur la queue `ftp`** — la seule dont les nœuds accèdent à
Internet (frontale exclue, autres nœuds sans réseau). Un job type est fourni
sous `/appli/services/exemples/pbs/ftp.pbs`.

Depuis le patch d'accès aux données, le script écrit dans
`$NOSC_DATA_ROOT/glorys_raw` (plus de chemin en dur) — exactement le répertoire
que `prepare_glorys_osse.py` lit par défaut. Il prend trois arguments : année,
profondeur min, profondeur max.

```bash
qsub -I -q ftp -l walltime=10:00:00 -l mem=16g
module load singularity
cd $DATAWORK/NOSC/code/download_data_
export NOSC_DATA_ROOT=$SCRATCH/nosc/data

# identifiants Copernicus, une seule fois (persistés dans $HOME) :
singularity exec --bind $DATAWORK,$SCRATCH \
  $DATAWORK/containers/nosc-2026-09.sif copernicusmarine login

# téléchargement année par année (5 niveaux via l'intervalle 0.49–200) :
for y in $(seq 2010 2020); do
  echo "=== $y ==="
  singularity exec --bind $DATAWORK,$SCRATCH \
    $DATAWORK/containers/nosc-2026-09.sif \
    python import_data_glorys_multidepth.py $y 0.49 200
done
```

Puis on prépare les fichiers consolidés — même script qu'en F.3, mais en
laissant la source par défaut (`$NOSC_DATA_ROOT/glorys_raw`) :

```bash
singularity exec --bind $DATAWORK,$SCRATCH \
  $DATAWORK/containers/nosc-2026-09.sif \
  python prepare_glorys_osse.py
```

Version batch du téléchargement si la durée dépasse la session interactive
(`jobs/download_glorys.pbs`, sur le patron `ftp.pbs`) :

```csh
#!/bin/csh
#PBS -N glorys_dl
#PBS -q ftp
#PBS -l walltime=10:00:00
#PBS -l mem=16g

source /usr/share/Modules/init/csh   # indispensable en batch csh : définit `module`
module load singularity
cd $DATAWORK/NOSC/code/download_data_
setenv NOSC_DATA_ROOT $SCRATCH/nosc/data

foreach y (`seq 2010 2020`)
  echo "=== $y ==="
  singularity exec --bind $DATAWORK,$SCRATCH \
    $DATAWORK/containers/nosc-2026-09.sif \
    python import_data_glorys_multidepth.py $y 0.49 200
end
```

### F.5 Bathymétrie

**Absente du dépôt : aucun script ne produit `bathymetry_gs.nc`**, ni dans
`download_data_/` ni dans `contrib/`. La config attend une variable `deptho` sur
la même grille que la surface GLORYS.

Piste DATAREF d'abord. Le champ `deptho` statique de GLORYS n'est **pas** dans
`/home/ref-ocean-reanalysis/` (vérifié). Deux endroits à explorer :

```bash
# 1) la distribution "native" de GLORYS embarque souvent un masque/bathymétrie
ls /home/ref-ocean-reanalysis/glorys_native/ | grep -i "mask\|bathy\|mesh"
# 2) bathymétrie EMODNET dédiée (grille différente de GLORYS -> ré-interpolation)
ls /home/ref-ocean-model-public/EMODNET_BATHY/
```

L'idéal est un champ de profondeur sur exactement la même grille que la surface
GLORYS (variable attendue : `deptho`) — c'est ce que peut fournir un fichier
`mask_bathy`/`mesh` de `glorys_native` s'il existe. À défaut, EMODNET impose de
ré-interpoler sur la grille GLORYS. Sous-domainez sur la boîte Gulf Stream et
écrivez `$NOSC_DATA_ROOT/bathymetry_gs.nc`. En dernier recours, téléchargement
du static CMEMS sur la queue `ftp` :

```bash
singularity exec --bind $DATAWORK,$SCRATCH \
  $DATAWORK/containers/nosc-2026-09.sif python - << 'PYEOF'
import os, copernicusmarine
copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_static",
    variables=["deptho"],
    minimum_longitude=-66, maximum_longitude=-54,
    minimum_latitude=32, maximum_latitude=44,
    output_directory=os.environ["NOSC_DATA_ROOT"],
    output_filename="bathymetry_gs.nc",
)
PYEOF
```

**À vérifier** : que `bathymetry_gs.nc` contient bien `deptho` sur la même grille
(lat/lon) que `glorys_gs_surface_2010-2020.nc`, sans quoi le chargement du prior
statique échouera.
### F.6 ARGO virtuels

Géométrie réelle des profils, valeurs issues de GLORYS. Les ARGO virtuels sont
rasterisés sur la grille lue via `--grid-from` : pointez ce dernier vers le
GLORYS **préparé** (donc à la bonne résolution), et ajoutez `--expect-res` pour
que l'étape refuse de tourner si la grille reçue n'a pas la résolution attendue.

```bash
singularity exec --bind $DATAWORK,$SCRATCH \
  $DATAWORK/containers/nosc-2026-09.sif \
  python -m contrib.argo.virtual \
    --grid-from $NOSC_DATA_ROOT/glorys_gs_surface_2010-2020.nc \
    --truth-path $NOSC_DATA_ROOT/glorys_gs_multidepth_2010-2020.nc \
    --start-date 2010-01-01 --end-date 2020-01-01 \
    --lat-min 32 --lat-max 44 --lon-min -66 --lon-max -54 \
    --depths 0.49 15 50 100 200 \
    --expect-res 0.25 \
    --output-dir $NOSC_DATA_ROOT/argo_virtual/gridded
```

(Retirez `--expect-res` si vous avez préparé GLORYS à la résolution native, ou
mettez-y la valeur de `NOSC_TARGET_RES`. ARGO n'a pas de résolution propre — ce
qui compte est que sa rasterisation tombe sur la même grille que GLORYS.)

**Point de vigilance.** Cette étape appelle `argopy` avec son backend ERDDAP
distant par défaut — elle exige donc un accès Internet (queue `ftp`) et se
heurtera aux avertissements `argopy` de l'image : une incompatibilité entre
les versions d'`argopy` et d'`erddapy` désactive les *fetchers* ERDDAP,
ArgoVis et GDAC. Si l'étape échoue pour cette raison, il faut épingler des
versions compatibles dans le YAML et reconstruire l'image.

**Piste d'amélioration : le miroir Argo local est confirmé présent.** Ifremer
héberge le centre Coriolis, l'un des deux centres mondiaux (GDAC) du programme
Argo. Sur cette instance, le GDAC natif est directement accessible :

```bash
ls /home/ref-argo/gdac/                  # GDAC Argo natif
ls /home/ref-coriolis-public/argo/       # miroir Coriolis
ls /home/ref-ISAS/ANA_ISAS20_ARGO/       # analyse ISAS basée Argo (grillée)
```

S'en servir directement évite `argopy`/Internet — mais cela demande d'adapter
`contrib/argo/download.py`, qui appelle aujourd'hui `argopy.DataFetcher` en
dur, pour lire les fichiers GDAC natifs à la place. Un chantier à part entière,
pas un simple changement de chemin ; à faire seulement si le passage par
`argopy` devient un vrai goulot d'étranglement. (Note : `ANA_ISAS20_ARGO` est
une analyse déjà grillée, potentiellement plus simple à consommer qu'un GDAC de
profils bruts, si la géométrie éparse réelle n'est pas indispensable.)

### F.7 Masques d'observation

Ils sont produits par les `entrypoints:` de la configuration, au premier
lancement de `main.py`. C'est l'étape qui échoue si la grille de référence
produite en F.3 est absente. Pour les générer indépendamment, hors du
protocole OSSE :

```bash
singularity exec --bind $DATAWORK,$SCRATCH \
  $DATAWORK/containers/nosc-2026-09.sif \
  python -m contrib.synthetic_obs.build_masks \
    --grid-from $NOSC_DATA_ROOT/glorys_gs_surface_2010-2020.nc \
    --n-days 3653 \
    --missions jason3 sentinel3a sentinel3b saral hy2b swot \
    --expect-res 0.25 \
    --output $NOSC_DATA_ROOT/masks_synthetic.pickle
```

(Même logique qu'en F.6 : `--grid-from` pointe le GLORYS préparé, `--expect-res`
vérifie la grille. Retirez-le pour le natif, ou alignez sur `NOSC_TARGET_RES`.)

Contrairement à GLORYS, ARGO et la bathymétrie, cette étape ne dépend
d'**aucune donnée externe** : les trajectoires satellite sont calculées
analytiquement (modèle d'orbite en forme close, paramètres de mission réels
mais codés en dur) — aucun accès réseau, aucun jeu de données Datarmor à
chercher.

### F.8 Tests fournis

```bash
singularity exec --bind $DATAWORK,$SCRATCH \
  $DATAWORK/containers/nosc-2026-09.sif python tests/test_synthetic_obs.py

singularity exec --bind $DATAWORK,$SCRATCH \
  $DATAWORK/containers/nosc-2026-09.sif python tests/test_argo_and_pseudo_obs.py
```

### F.9 Archiver la copie maître

Sur `$DATAWORK`, qui n'est pas purgé, à l'inverse du `$SCRATCH` où les données
de travail viennent d'être préparées :

```bash
tar cf $DATAWORK/nosc_data_2026-09.tar -C $SCRATCH/nosc data/
```

# G. Lancer un entraînement

### G.1 Smoke test interactif

Un GPU est **obligatoire**, même pour un test de deux itérations : le
sélecteur de canaux multi-variable appelle `.cuda()` en dur.

```bash
qsub -I -q gpuq -l walltime=00:30:00 -l mem=64g
module load singularity
cd $DATAWORK/NOSC/code/multivar_drifter_
export XP=osse3d_gs_multivar_unet
export NOSC_DATA_ROOT=$SCRATCH/nosc/data

singularity exec --nv --bind $DATAWORK,$SCRATCH \
  $DATAWORK/containers/nosc-2026-09.sif \
  python -u main.py xp=$XP \
    hydra.run.dir=$SCRATCH/nosc_runs/$(date +%Y%m%d-%H%M%S) \
    ++trainer.max_epochs=2 ++trainer.limit_train_batches=5 ++trainer.limit_val_batches=2
```

**`hydra.run.dir` n'est pas optionnel.** Par défaut, Hydra écrit dans
`outputs/<date>/<heure>/` relativement au répertoire courant — donc, si vous
lancez depuis le code, sur `$DATAWORK`, où s'accumuleraient des milliers de
petits fichiers de logs. Pointez systématiquement le répertoire de sortie vers
`$SCRATCH`. L'horodatage est calculé par la commande shell `date` plutôt
qu'avec la syntaxe `${now:…}` de Hydra, que le shell tenterait d'interpréter.

En cas d'échec, `SINGULARITYENV_HYDRA_FULL_ERROR=1` donne la trace complète :
le préfixe `SINGULARITYENV_` est nécessaire pour qu'une variable d'environnement
traverse la barrière du conteneur.

### G.2 Dimensionner le job GPU

Sur Datarmor, tous les entraînements GPU passent par l'unique queue `gpuq`. Les
quelques valeurs à relever une fois sur place, toutes lisibles en une commande :

| À déterminer | Commande |
|---|---|
| Confirmer le nom de la queue GPU | `qstat -Q` (attendu : `gpuq`) |
| Walltime maximal de `gpuq` | `qstat -Qf gpuq \| grep -i walltime` |
| Syntaxe de réservation GPU (`ngpus=` explicite ou implicite via la queue) | doc intranet « Queues d'exécution PBS » ; à défaut, tester en interactif (D.1) |
| Nombre de GPU et de cœurs par nœud, mémoire GPU | `nvidia-smi` en session interactive, ou `pbsnodes` sur un nœud `gpuq` |

Règle de dimensionnement : demandez un nombre de cœurs proportionné au nombre
de GPU du nœud. Trop peu de cœurs étrangle les *dataloaders* et laisse le GPU
attendre les données ; en réserver plus que nécessaire immobilise inutilement
une ressource partagée et allonge votre temps d'attente en file.

### G.3 Script de soumission

`$DATAWORK/NOSC/jobs/train_gpu.pbs`, en csh (convention Datarmor). La variante
bash est en commentaire.

```csh
#!/bin/csh
#PBS -N nosc_train
#PBS -q gpuq
#PBS -l walltime=48:00:00
#PBS -l mem=100g
## Réservation GPU : décommenter/ajuster selon G.2 si le GPU n'est pas
## attribué par la seule queue :
##PBS -l select=1:ncpus=14:ngpus=1:mem=100g
#PBS -m ae
#PBS -M prenom.nom@ifremer.fr

source /usr/share/Modules/init/csh   # indispensable en batch csh : définit `module`
module load singularity

cd $PBS_O_WORKDIR                     # répertoire depuis lequel le job a été soumis
cd $DATAWORK/NOSC/code/multivar_drifter_

setenv NOSC_DATA_ROOT $SCRATCH/nosc/data

singularity exec --nv \
  --bind $DATAWORK,$SCRATCH \
  $DATAWORK/containers/nosc-2026-09.sif \
  python -u main.py xp=osse3d_gs_multivar_unet \
    hydra.run.dir=$SCRATCH/nosc_runs/`echo $PBS_JOBID | cut -d. -f1`

## --- variante bash : première ligne #!/bin/bash, puis
##   export NOSC_DATA_ROOT=$SCRATCH/nosc/data
##   ... hydra.run.dir=$SCRATCH/nosc_runs/${PBS_JOBID%%.*}
```

À ne pas modifier à la légère :

- `walltime=48:00:00` est un ordre de grandeur : ajustez-le sous le plafond
  réel de la queue `gpuq` (G.2). Si l'entraînement complet dépasse ce plafond,
  prévoyez la reprise sur checkpoint (G.5).
- `python -u` désactive la bufférisation de la sortie : sans cette option, les
  logs n'apparaissent qu'à la toute fin du job.
- `hydra.run.dir` pointe vers `$SCRATCH` avec l'identifiant numérique du job
  (`$PBS_JOBID` débarrassé de son suffixe `.datarmor0`), ce qui relie chaque
  répertoire de sortie au job qui l'a produit.
- Pas de module d'architecture à charger ni de compte d'imputation à préciser
  au moment de la soumission. Si votre projet dispose d'un compte
  d'allocation particulier, la doc intranet indique la directive
  correspondante.

### G.4 Soumettre et surveiller

```bash
cd $DATAWORK/NOSC
qsub jobs/train_gpu.pbs               # renvoie <jobid>.datarmor0

qstat -u $USER                        # état de vos jobs en file
qstat -f <jobid>                      # détail d'un job (nœud, ressources, raison d'attente)
qdel <jobid>.datarmor0                # annuler un job
```

Les sorties et erreurs sont écrites dans `nosc_train.o<jobid>` et
`nosc_train.e<jobid>` dans le répertoire de soumission (sauf redirection via
`#PBS -o`/`-e`). États affichés par `qstat` : `Q` en attente, `R` en cours,
`E` en fin d'exécution, `H` bloqué, `C` terminé. Si un job reste en `Q`,
`qstat -f <jobid>` (champ `comment`) en donne généralement la raison.

### G.5 Après le job

Si vous avez mis `#PBS -m ae`, PBS vous envoie un e-mail de fin contenant les
ressources réellement consommées :

```
resources_used.mem=...
resources_used.walltime=...
```

Ajustez `mem` et `walltime` en conséquence au prochain run : surdimensionner
allonge inutilement l'attente en file.

```bash
qstat -fx <jobid>                     # historique du job, dont Exit_status
ls $SCRATCH/nosc_runs/<jobid>/
```

`Exit_status = 0` signale une fin normale ; un job tué au walltime doit être
repris depuis son dernier checkpoint. Chaque run écrit ses logs CSV et ses
checkpoints — les trois meilleurs sur `val_total_mse` — dans le répertoire
Hydra. Pour comparer plusieurs runs, lisez les `metrics.csv` respectifs.

**Rapatriez checkpoints et métriques vers `$DATAWORK` sans attendre**, puisque
tout fichier de plus de dix jours sur `$SCRATCH` est supprimé :

```bash
cp -r $SCRATCH/nosc_runs/<jobid> $DATAWORK/nosc_results/
```

Pour enchaîner des jobs au-delà du walltime maximal (chaînage PBS) :

```bash
qsub -h -N nosc1 jobs/train_gpu.pbs
qsub -N nosc2 -W depend=afterok:'qselect -N nosc1 -u $USER' jobs/train_gpu.pbs
qrls 'qselect -N nosc1 -u $USER'
```

(ou, plus simplement, `qsub -W depend=afterok:<jobid> jobs/train_gpu.pbs` si
vous avez déjà l'identifiant du premier job).

---

# H. Inférence à partir d'un checkpoint

Pour reconstruire des champs sans réentraîner, ajoutez dans la configuration :

```yaml
entrypoints:
  - _target_: src.train.base_training
    trainer: ${trainer}
    lit_mod: ${model}
    dm: ${datamodule}
    ckpt: /chemin/vers/checkpoint.ckpt
    only_rec: True
```

Sortie : un `test_data_dimN.nc` par dimension de sortie, dans l'ordre des
entrées `full_output` du bloc `multivar:`, contenant les champs `inp`, `tgt`
et `out`. L'inférence tourne dans un job `gpuq` comme l'entraînement (script
G.3, en pointant la config vers ce checkpoint).

---

# I. Pièges connus

**Les montages `--bind`.** Seuls les répertoires explicitement liés par
`--bind` sont visibles depuis l'intérieur du conteneur. C'est la première cause
de « fichier introuvable » sur un chemin pourtant correct. `$DATAWORK`,
`$SCRATCH` et, si vous les utilisez, les dossiers de données de référence
(`/home/ref-ocean-reanalysis`, `/home/ref-argo`, `/home/ref-ecmwf`…) doivent
chacun figurer dans `--bind`.

**Le shell.** Datarmor est en csh/tcsh par défaut. Les commandes interactives
de ce guide sont en bash (`export`, `for … do … done`, heredocs) et échoueront
en csh ; les scripts PBS sont en csh (`setenv`, `foreach … end`,
`source …/init/csh`). Ne mélangez pas les deux syntaxes dans un même fichier.
Piège fréquent et discret : en csh, un `:` placé juste après une variable est
lu comme un *modificateur* (`$var:h`, `$var:t`…). Un montage écrit
`--bind $DATAWORK:$DATAWORK` déclenche donc `Bad : modifier in $ ($).` alors
que `$DATAWORK` est parfaitement définie. C'est pourquoi tous les montages de
ce guide sont écrits sans `:` — `--bind $DATAWORK,$SCRATCH`, qui monte chaque
chemin au même emplacement dans le conteneur et fonctionne dans les deux
shells.

**Rien de lourd sur la frontale.** Le nœud de login sert à naviguer, éditer et
compiler — pas à calculer ni à manipuler de gros fichiers. Tout passe par
`qsub`.

**La purge du `$SCRATCH` à 10 jours.** Données préparées, checkpoints et
métriques doivent tous avoir leur copie maître sur `$DATAWORK`. Un `tar` par
campagne suffit.

**Les sorties d'Hydra.** Sans `hydra.run.dir`, tout atterrit dans `outputs/`
sous le répertoire courant. Pointez systématiquement ce paramètre vers
`$SCRATCH`.

**Le HOME dans le conteneur.** Il est monté automatiquement, ce qui laisse
fuir caches et paquets Python locaux (`~/.local`) dans l'environnement du
conteneur. `PYTHONNOUSERSITE=1` est défini dans l'image pour neutraliser
`~/.local` ; en cas de comportement inexplicable, ajoutez `--no-home` à la
commande `singularity exec`. (Les variables `MPLCONFIGDIR` et `NUMBA_CACHE_DIR`
définies dans l'image pointent sous `$SCRATCH` si la variable existe, sinon
`/tmp` — rien à faire.)

**Pas d'Internet sur les nœuds de calcul.** Toute étape qui télécharge doit
être soumise sur la queue `ftp`. Un `copernicusmarine` ou un `pip install` qui
« pend » indéfiniment dans un job `gpuq`, c'est ça.

**GPU obligatoire**, y compris pour un smoke test de deux itérations.

**Jupyter.** Datarmor offre un JupyterHub
(`https://datarmor-jupyterhub.ifremer.fr/`, login intranet, réseau Ifremer ou
VPN) mais il ne lance pas les noyaux dans votre conteneur Singularity : il
fonctionne avec des environnements **conda**. Si votre usage passe par les
notebooks, gardez en parallèle un petit environnement conda dédié à
l'exploration — l'environnement de référence des runs reste l'image. Pour
activer conda : ajoutez `source /appli/anaconda/latest/etc/profile.d/conda.csh`
à votre `~/.cshrc`, et un `.condarc` déclarant vos `envs_dirs`. À la sélection
de ressources JupyterHub, **ne spécifiez pas d'environnement optionnel**.

**Exemples PBS locaux.** Datarmor fournit des patrons prêts à l'emploi sous
`/appli/services/exemples/` (sous-dossiers `pbs/`, `R/`…), dont le `ftp.pbs`
qui a inspiré le job de téléchargement en F.3.

**`sst` n'est pas la température brute** : l'entrée historique est transformée
en `log|∇T|`, une caractéristique de fronts. Pour une vraie cible de
température de surface, utiliser `sst_tgt`.

**`thetao_argo_*` et `thetao_*`** sont deux cibles distinctes — ARGO épars
contre GLORYS dense — et non une seule variable dédoublée.

**Ordre des `head_group`** : les variables d'un même groupe doivent être
contiguës dans `multivar:`, sinon une erreur explicite est levée.

**Configurations dépréciées.** Les `unet_uv_full_integration_*` relèvent d'un
protocole hybride abandonné ; la référence est `osse3d_gs_multivar_unet`.

**mpi4py** n'est pas inclus dans l'image. L'exécution MPI depuis un conteneur
sur Datarmor (avec les bibliothèques MPT/Intel MPI de l'hôte et `$MPI_LAUNCH`)
est un chantier à part entière — à traiter seulement si le calcul distribué
devient nécessaire, avec l'assistance.

---

# J. Mettre à jour l'image

L'image est figée une fois construite : toute modification du YAML impose de la
reconstruire sur votre poste (étape A) puis de la retransférer (étape B).

```bash
# poste local
apptainer build --fakeroot nosc-2026-10.sif nosc.def
rsync -avP nosc-2026-10.sif <login>@datacopy.ifremer.fr:   # voie de transfert : voir B.2
# Datarmor
mv ~/nosc-2026-10.sif $DATAWORK/containers/
```

Taguez les images par date et versionnez `nosc.def`, le YAML et ce README avec
le code. C'est ce qui permet de retrouver, six mois plus tard, quel
environnement a produit quel résultat.

---

# K. Remerciements Datarmor

À faire figurer dans toute publication utilisant Datarmor :

> *The authors acknowledge the Pôle de Calcul et de Données Marines (PCDM,
> http://www.ifremer.fr/pcdm) for providing DATARMOR storage, data access,
> computational resources, visualization and support services.*

---

# Annexe — aide-mémoire des commandes PBS

| Besoin | Commande |
|---|---|
| Session interactive CPU | `qsub -I -l walltime=HH:MM:SS -l mem=…g` |
| Session interactive GPU | `qsub -I -q gpuq -l walltime=HH:MM:SS -l mem=…g` |
| Session avec accès Internet (téléchargement) | `qsub -I -q ftp -l walltime=HH:MM:SS -l mem=…g` |
| Soumettre un job batch | `qsub script.pbs` |
| État de vos jobs | `qstat -u $USER` |
| Détail d'un job (en file ou en cours) | `qstat -f <jobid>` |
| Pourquoi un job attend | `qstat -f <jobid> \| grep -i "job_state\|queue\|comment\|estimated"` |
| Suivre la sortie d'un job batch | `tail -f <nom>.o<jobid>` (dans le répertoire de soumission) |
| Historique d'un job terminé | `qstat -fx <jobid>` |
| Annuler un job | `qdel <jobid>.datarmor0` |
| Lister les queues et leur charge | `qstat -Q` |
| Limites d'une queue | `qstat -Qf <queue>` |
| Détail d'un nœud | `pbsnodes <nom-du-nœud>` |
| Job dépendant d'un autre | `qsub -W depend=afterok:<jobid> script.pbs` |
| Variables utiles dans un script | `$PBS_O_WORKDIR` (répertoire de soumission), `$PBS_JOBID` (identifiant du job) |
