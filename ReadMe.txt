
Immobil�App est une application Streamlit qui se constitue d�un ensemble de fonctionnalit�s compl�tes et interactives � la port�e de tous.
L�application permet de mieux visualiser les donn�es fonci�res en France sur l�ann�e 2020 (m�tropole + DOM-TOM).

Une fois lanc�e, nous arrivons sur une page d�accueil, qui nous liste ses possibilit�s que l�application propose.
Immobil'App permet donc d�effectuer :
un aper�u rapide
des pr�-traitements
un rapport Pandas Profiling
un rapport SweetViz Report
une exploration en profondeur

Le choix de l�ensemble des cinq fonctionnali�s, ainsi que l�accueil, s�effectue au niveau de la sidebar sur la gauche de la page.

L�utilsateur peut s�acclimater avec les donn�es � l�aide d�un aper�u rapides des donn�es (�head� + carte)

Les libraires Pandas Profiling et SweetViz Report fournissent des mod�les de visualisation pr�d�finis, tandis que notre approche, plus interactive, octroie une meilleure maniabilit� de la donn�e. 
N�anmoins, les deux proc�d�s restent tr�s complets et id�aux pour visualiser l�ensemble des donn�es. 

Les pr�-traitements permettent les op�ations suivantes�:
sample size (%)
Select fields
Convert Data Types
Missing values
Duplicate rows
Order values
Une fois fait, on puet afficher les colonnes qui en r�sultent, ou bien en cr�er un CSV.

Notre exploration d�taill�e affiche, selon le d�partement et le type d�habitat s�lectionn��: 
la nature des mutations (graphe)
l�ensemble des statistiques 
l�ensemble des biens
mais surtout un graphe param�tr� selon les colonnes choisies.

------------------------------

############Partie commune � tous les modules ########################

Avant tout, je d�finis une fonction d�corateur qui permet de calculer le temps d'ex�cution d'une fonction
Je l'appliquerai principalement � la fonction du chargement des donn�es qui est tr�s consomatrice de temps dans notre cas, �tant donn�, la grande taille du dataset.

On applique deux d�corateurs � la fonction load_data():
le d�corateur @st.cache qui permet de garder en cache les donn�es charg�es, m�me quand l�application, donc le code, est mise � jour, suivi du d�corateur @log_time d�fini pr�c�demment

On affiche un texte comme quoi les donn�es sont en train d'�tre charg�es.

Afin d'all�ger certains traitements, on g�n�re un �chantillon al�atoire.

Une fois les donn�es charg�es, on affiche le message associ�.

###########D�finitions de quelques transformations communes de bases suppl�mentaires statiques ############

#########Pr�paration des donn�es : exploration rapide, plus des fonctionnalit�s de transformations#############

# Mettre un texte en sur-brillance
# Fonction de t�l�chargement en diff�rents types de donn�es
# Module d'exploration g�n�rale et rapide : apper�u sur les donn�es
# Module de transformations approfondies des donn�es

   # Colonnes / attributs

# Conversion des types de donn�es
# Traitements des NULLS
# Traitements des duplicatats
# ORDER VALUES
# Bouton de t�l�chargement
##########Exploration personnalis�e et d�taill�e des donn�es##########################
# Application du d�corateur suppression des Warnings"

    # Menu d�roulant pour les 3 r�gions
# Pour chaque chois, choisir Maison ou appartement
# Puis pour chaque sous-choix, afficher :
# l'ensemble des biens, la moyenne de la valeur fonci�re ou la r�partition des mutations

# Impl�mentation des fonction du sous-programme principal de l'exploration d�taill�e : map_info():
# Dataframe pour plot valeur fonciere en fonction de la surface des appart region 69
#69 appart slide

#########Le programme principal################################




