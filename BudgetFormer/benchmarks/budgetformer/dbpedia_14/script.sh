#!/bin/bash
#SBATCH --job-name=dbpedia_14  # Nom de votre job
#SBATCH --output=output.txt  # Fichier de sortie
#SBATCH --error=error.txt    # Fichier d'erreur
#SBATCH --partition=GPU-A100  # Spécifier la partition GPU-A100
#SBATCH --nodes=1            # Nombre de nœuds requis
#SBATCH --cpus-per-task=4    # Nombre de CPU par tâche
#SBATCH --gres=gpu:1         # Demande un seul GPU disponible
#SBATCH --mem=80G            # Mémoire requise
#SBATCH --time=infinite       # Temps d'exécution maximal (hh:mm:ss)

# Chargez les modules nécessaires, activez votre environnement virtuel, etc.
source ../../../magi-env/bin/activate

# Afficher le GPU attribué par Slurm
echo "GPU alloué par Slurm : $CUDA_VISIBLE_DEVICES"

# Exécutez votre commande ou script Python
CUDA_LAUNCH_BLOCKING=1 python3 main.py

