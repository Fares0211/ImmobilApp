
Immobil’App est une application Streamlit qui se constitue d’un ensemble de fonctionnalités complètes et interactives à la portée de tous.
L’application permet de mieux visualiser les données foncières en France sur l’année 2020 (métropole + DOM-TOM).

Une fois lancée, nous arrivons sur une page d’accueil, qui nous liste ses possibilités que l’application propose.
Immobil'App permet donc d’effectuer :
un aperçu rapide
des pré-traitements
un rapport Pandas Profiling
un rapport SweetViz Report
une exploration en profondeur

Le choix de l’ensemble des cinq fonctionnaliés, ainsi que l’accueil, s’effectue au niveau de la sidebar sur la gauche de la page.

L’utilsateur peut s’acclimater avec les données à l’aide d’un aperçu rapides des données (‘head’ + carte)

Les libraires Pandas Profiling et SweetViz Report fournissent des modèles de visualisation prédéfinis, tandis que notre approche, plus interactive, octroie une meilleure maniabilité de la donnée. 
Néanmoins, les deux procédés restent très complets et idéaux pour visualiser l’ensemble des données. 

Les pré-traitements permettent les opéations suivantes :
sample size (%)
Select fields
Convert Data Types
Missing values
Duplicate rows
Order values
Une fois fait, on puet afficher les colonnes qui en résultent, ou bien en créer un CSV.

Notre exploration détaillée affiche, selon le département et le type d’habitat sélectionné : 
la nature des mutations (graphe)
l’ensemble des statistiques 
l’ensemble des biens
mais surtout un graphe paramétré selon les colonnes choisies.

------------------------------

############Partie commune à tous les modules ########################

Avant tout, je définis une fonction décorateur qui permet de calculer le temps d'exécution d'une fonction
Je l'appliquerai principalement à la fonction du chargement des données qui est très consomatrice de temps dans notre cas, étant donné, la grande taille du dataset.

On applique deux décorateurs à la fonction load_data():
le décorateur @st.cache qui permet de garder en cache les données chargées, même quand l’application, donc le code, est mise à jour, suivi du décorateur @log_time défini précédemment

On affiche un texte comme quoi les données sont en train d'être chargées.

Afin d'alléger certains traitements, on génère un échantillon aléatoire.

Une fois les données chargées, on affiche le message associé.

###########Définitions de quelques transformations communes de bases supplémentaires statiques ############

#########Préparation des données : exploration rapide, plus des fonctionnalités de transformations#############

# Mettre un texte en sur-brillance
# Fonction de téléchargement en différents types de données
# Module d'exploration générale et rapide : apperçu sur les données
# Module de transformations approfondies des données

   # Colonnes / attributs

# Conversion des types de données
# Traitements des NULLS
# Traitements des duplicatats
# ORDER VALUES
# Bouton de téléchargement
##########Exploration personnalisée et détaillée des données##########################
# Application du décorateur suppression des Warnings"

    # Menu déroulant pour les 3 régions
# Pour chaque chois, choisir Maison ou appartement
# Puis pour chaque sous-choix, afficher :
# l'ensemble des biens, la moyenne de la valeur foncière ou la répartition des mutations

# Implémentation des fonction du sous-programme principal de l'exploration détaillée : map_info():
# Dataframe pour plot valeur fonciere en fonction de la surface des appart region 69
#69 appart slide

#########Le programme principal################################




