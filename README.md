						Immobil’App


#Aspects fonctionnels

INTRODUCTION : 
Application Streamlit qui se constitue d’un ensemble de fonctionnalités modules complets et interactifs à la portée de tous (qui se veut ergonomique) ; permet de mieux visualiser les données foncières en France sur l’année 2020 (métropole + DOM-TOM).

AVERTISSEMENT : 
A noter que les problèmes techniques rencontrés sont principalement à la taille du Dataset. Le GitHub ne supportant pas les jeux de données de cette taille, m'a obligé à me retourner vers d'autres solutions. J'ai réussi à faire tourner l'application grâce à GLFS (git large files), mais cette solution a été vite abandonnée, une fois dépassé la bande passante autorisée. D'autres solutions ont été mises en place, sur drive et sur Sharepoint, mais également abandonnées (problème de connexion et d'accès). Je me suis rabattu finalement sur le lien fourni.



FONCTIONNALITES ET MODULES MIS EN PLACE : 
L'application est composée de plusieurs modules, services et fonctionnalités répartis suivant les rubriques suivantes :


- Aperçu rapide

- Pré-traitements

- Rapport Pandas Profiling

- Rapport SweetViz Report

- Exploration personnalisée 



Le choix de l’ensemble des cinq fonctionnalités, ainsi que l’accueil, s’effectue au niveau de la sidebar sur la gauche de la page.



A travers ces rubriques l’utilisateur peut :

- S’acclimater avec les données à l’aide d’un aperçu rapides des données (‘head’ + carte)



- Exploiter les libraires Pandas Profiling et SweetViz fournissant des modèles de visualisation prédéfinis, contrairement à notre approche, plus interactive, octroie une meilleure maniabilité de la donnée. Néanmoins, les deux procédés restent très complets et idéaux pour visualiser l’ensemble des données. 



- Effectuer des pré-traitements grâce à des opérations telles que :

	* sample size (%) : pour générer un échantillon

	* Select fields : sélectionner les attributs à transformer

	* Convert Data Types : conversion des données

	* Missing values : traitements des Nulls

	* Duplicate rows : traitements des duplicatas 

	* Order values : ordre de tri

	* Export : Une fois fait, on peut afficher ou bien en créer un CSV ou un pickle.



- Effectuer une exploration détaillée des données :
	* par région et par type d’habitat

	* Par des graphiques à la demande selon les axes souhaités la nature des mutations (graphe)

	* Par des des statistiques en fonction du contexte de navigation

	* Par des cartes 
	* Par des interactions 
	* Etc. 



############Aspects techniques############



------Partie commune à tous les modules------



Avant tout, je définis une fonction décorateur qui permet de calculer le temps d'exécution d'une fonction.

Je l'appliquerai principalement à la fonction du chargement des données qui est très consommatrice de temps dans notre cas, étant donné, la grande taille du dataset.



J'applique deux décorateurs à la fonction load_data():

le décorateur @st.cache qui permet de garder en cache les données chargées, même quand l’application, donc le code, est mise à jour, suivi du décorateur @log_time défini précédemment



Lors du chargement, j'affiche un texte comme quoi les données sont en train d'être chargées.



Afin d'alléger certains traitements, je génère un échantillon aléatoire.



Une fois les données chargées, j'affiche un message l'annonçant.




------Définitions de quelques transformations communes------

Dans cette section, je définis et j'applique des transformations supplémentaires statiques de base.




------Préparation des données : exploration rapide, plus des fonctionnalités de transformations------



- Au niveau du Dataset :
	* Mettre un texte en sur-brillance

	* Fonction de téléchargement en différents types de données

	* Module d'exploration générale et rapide : aperçu sur les données

	* Module de transformations approfondies des données



- Au niveau des Colonnes / attributs



	* Conversion des types de données

	* Traitements des NULLS

	* Traitements des duplicatas

	* ORDER VALUES

	*Bouton de téléchargement


------Exploration personnalisée et détaillée des données------


- Application du décorateur suppression des Warnings"

- Menu déroulant pour les 3 régions

	* Pour chaque chois, choisir Maison ou appartement

	* Puis pour chaque sous-choix, afficher :

		* l'ensemble des biens, la moyenne de la valeur foncière ou la répartition des mutations
		* des cartes, statistiques, divers, ...
		* etc.

- Implémentation des fonctions du sous-programme principal de l'exploration détaillée : map_info(): 


------Le programme principal------

Le programme principal fait appel et assemble l'ensemble des fonctions et modules, librairies, etc. développés, initialisés, appelés, etc. dans les sections précédentes.









