
import pickle
import io
import base64
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import df_helper as helper  # custom script
import sweetviz as sv
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import altair as alt
import plotly.express as px
import plotly.graph_objs as go

############Partie commune à tous les modules ########################

st.title("Immobil'App")

# Avant tout, je défini une fonction décorateur qui permet de calculer le temps d'exécution d'une fonction
# Je l'appliquerai principalement à la fonction du chargement des données qui est très consomatrice de temps
# dans notre cas, étant donné, la grande taille du dataset.
def log_time(func):

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        f = open("log_dev.txt", 'a', encoding="utf8")
        time_res = end - start
        mes = "\n" + func.__name__ + " time = " + \
              str(time_res) + " secondes" + "... ça s'améliore ?"

        f.write(mes)
        return result

    return wrapper


# On applique deux décorateurs à la fonction load_data():
# le décorateur @st.cache qui permet de garder en cache les données chargées, même quand l’application, donc le code, est mise à jour
# Suivi du décorateur @log_time défini précédemment
@st.cache(allow_output_mutation=True)
@log_time
def load_data():
    data = pd.read_csv('https://jtellier.fr/DataViz/full_2020.csv', skiprows = lambda i: i % 20)
    columns = ['adresse_suffixe', 'ancien_code_commune', 'ancien_nom_commune', 'ancien_id_parcelle', 'numero_volume',
               'lot1_numero', 'lot1_surface_carrez', 'lot2_numero', 'lot2_surface_carrez', 'lot3_numero',
               'lot3_surface_carrez', 'lot4_numero', 'lot4_surface_carrez', 'lot5_numero', 'lot5_surface_carrez',
               'nombre_lots', 'code_nature_culture', 'code_nature_culture_speciale', 'nature_culture_speciale',
               'adresse_code_voie']
    data.drop(columns, inplace=True, axis=1)
    #data["code_commune"] = data["code_commune"].astype(str)
    #data["code_departement"] = data["code_departement"].astype(str)
    data.dropna()
    return data

# On affiche un texte comme quoi les données sont entrain d'être chargées.
# data_load_state = st.text('Chargement des données...')
data = load_data()
# Afin d'alléger certains traitements, on génère un échatntillon aléatoire.
data = data.sample(1000)
# Une fois les données chargées, on affiche le message associé.
# data_load_state.text("Chargement effectué! (en utilisant @st.cache(allow_output_mutation=True))")


###########Définitions de quelques transformations communes de bases supplémentaires statiques ############

def count_rows(rows):
    return len(rows)

def get_month(data):
    return data.month

def transformation(data):
    data['date_mutation'] = pd.to_datetime(data['date_mutation'])
    data['month'] = data['date_mutation'].map(get_month)


#########Préparation des données : exploration rapide, plus des fonctionnalités de transformations#############

# Mettre un texte en sur-brillance
def highlight(txt):
    return '<span style="color: #F04E4E">{}</span>'.format(txt)

# Fonction de téléchargement en différents types de données
def download_file(df, extension):
    if extension == 'csv':  # csv
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
    else:  # pickle
        b = io.BytesIO()
        pickle.dump(df, b)
        b64 = base64.b64encode(b.getvalue()).decode()

    href = f'<a href="data:file/csv;base64,{b64}" download="new_file.{extension}">Download {extension}</a>'
    st.write(href, unsafe_allow_html=True)

# Module d'exploration générale et rapide : apperçu sur les données
def explore(df):
    pr = ProfileReport(df, explorative=True)
    expander_df = st.expander('Data frame')
    expander_df.write(df)
    st_profile_report(pr)

# Module de transformations approfondies des données
def transform(df):
    # Echantillon
    expander_sample = st.expander('Sample size (%)')
    expander_sample.text('Select a random sample from this dataset')
    frac = expander_sample.slider('Random sample (%)', 1, 100, 100)
    if frac < 100:
        df = df.sample(frac=frac / 100)

    # Colonnes / attributs
    expander_fields = st.expander('Select fields')
    expander_fields.text('Select and order the fields')
    cols = expander_fields.multiselect('Columns', df.columns.tolist(), df.columns.tolist())
    df = df[cols]
    if len(cols) < 1:
        st.write('You must select at least one column.')
        return
    types = {'-': None, 'Boolean': '?', 'Byte': 'b', 'Integer': 'i',
             'Floating point': 'f', 'Date Time': 'M',
             'Time': 'm', 'Unicode String': 'U',
             'Object': 'O'}
    new_types = {}

    # Conversion des types de données
    expander_types = st.expander('Convert Data Types')

    for i, col in enumerate(df.columns):
        txt = f'Convert {col} from {highlight(df[col].dtypes)} to:'
        expander_types.markdown(txt, unsafe_allow_html=True)
        new_types[i] = expander_types.selectbox('Type:'
                                                , [*types]
                                                , index=0
                                                , key=i)

    # Traitements des NULLS
    expander_nulls = st.expander('Missing values')
    null_dict = {'-': None, 'Drop rows': 0, 'Replace with Note': 1,
                 'Replace with Average': 2, 'Replace with Median': 3, 'Replace with 0 (Zero)': 4}

    n_dict = {}
    cols_null = []
    for i, c in enumerate(df.columns):
        if df[c].isnull().values.any():
            cols_null.append(c)
            txt = f'{c} has {df[c].isnull().sum()} null values'
            expander_nulls.text(txt)
            n_dict[i] = expander_nulls.selectbox('What to do with Nulls:'
                                                 , [*null_dict]
                                                 , index=0
                                                 , key=i)

    # Traitements des duplicatats
    expander_duplicates = st.expander('Duplicate rows')
    duplicates_count = len(df[df.duplicated(keep=False)])
    if duplicates_count > 0:
        expander_duplicates.write(df[df.duplicated(keep=False)].sort_values(df.columns.tolist()))
        duplicates_dict = {'Keep': None, 'Remove all': False, 'Keep first': 'first', 'Keep last': 'last'}
        action = expander_duplicates.selectbox('Handle duplicates:', [*duplicates_dict])
    else:
        expander_duplicates.write('No duplicate rows')

    # ORDER VALUES
    expander_sort = st.expander('Order values')
    sort_by = expander_sort.multiselect('Sort by:', df.columns.values)
    order_dict = {'Ascending': True, 'Descending': False}
    ascending = []
    for i, col in enumerate(sort_by):
        order = expander_sort.radio(f'{col} order:', [*order_dict])
        ascending.append(order_dict[order])

    # Bouton de téléchargement
    st.text(" \n")  # break line
    #col1, col2, col3 = st.beta_columns([.3, .3, 1])
    col1, col2, col3 = st.columns([.3, .3, 1])
    with col1:
        btn1 = st.button('Show Data')
    with col2:
        btn2 = st.button('Get CSV')
    with col3:
        btn3 = st.button('Get Pickle')

    if btn1 or btn2 or btn3:
        st.spinner()
        with st.spinner(text='In progress'):
            # Transform
            df = helper.convert_dtypes(df, types, new_types)
            df = helper.handle_nulls(df, null_dict, n_dict)
            if duplicates_count > 0:
                df = helper.handle_duplicates(df, duplicates_dict, action)
            if sort_by:
                df = df.sort_values(sort_by, ascending=ascending)
            # Display/ download
            if btn1:
                st.write(df)
            if btn2:
                download_file(df, "csv")
            if btn3:
                download_file(df, "pickle")

    # return df


##########Exploration personnalisée et détaillée des données##########################

# Application du décorateur suppression des Warnings"
@st.cache(suppress_st_warning=True)
def map_info():

    batiment69 = get_batiment69()
    batiment75 = get_batiment75()
    batiment13 = get_batiment13()
    batiment_info69 = get_info69()

    Valeur_moyenne69 = get_mean69()
    Valeur_moyenne75 = get_mean75()
    Valeur_moyenne13 = get_mean13()
    g = get_nomCommune75()
    oo = get_nomCommune69()
    d = get_nomCommune13()

    data.dropna(subset=["longitude"], inplace=True)
    data.dropna(subset=["latitude"], inplace=True)

    df_map69 = data.mask(data["code_departement"] != 69)
    df_map69 = df_map69.mask(df_map69["type_local"] != 'Maison')
    df_map69.dropna(subset=["longitude"], inplace=True)
    df_map69.dropna(subset=["latitude"], inplace=True)

    df_mapp69 = data.mask(data["code_departement"] != 69)
    df_mapp69 = df_mapp69.mask(df_mapp69["type_local"] != 'Appartement')
    df_mapp69.dropna(subset=["longitude"], inplace=True)
    df_mapp69.dropna(subset=["latitude"], inplace=True)

    df_mapp75 = data.mask(data["code_departement"] != 75)
    df_mapp75 = df_mapp75.mask(df_mapp75["type_local"] != 'Maison')
    df_mapp75.dropna(subset=["longitude"], inplace=True)
    df_mapp75.dropna(subset=["latitude"], inplace=True)

    df_mappA75 = data.mask(data["code_departement"] != 75)
    df_mappA75 = df_mappA75.mask(df_mappA75["type_local"] != 'Appartement')
    df_mappA75.dropna(subset=["longitude"], inplace=True)
    df_mappA75.dropna(subset=["latitude"], inplace=True)

    df_mapM13 = data.mask(data["code_departement"] != 13)
    df_mapM13 = df_mapM13.mask(df_mapM13["type_local"] != 'Maison')
    df_mapM13.dropna(subset=["longitude"], inplace=True)
    df_mapM13.dropna(subset=["latitude"], inplace=True)

    df_map13 = data.mask(data["code_departement"] != 13)
    df_map13 = df_map13.mask(df_map13["type_local"] != 'Appartement')
    df_map13.dropna(subset=["longitude"], inplace=True)
    df_map13.dropna(subset=["latitude"], inplace=True)

    if st.checkbox("Afficher / masquer toutes les maisons "):
        st.map(data)

    # Menu déroulant pour les 3 régions
    # Pour chaque chois, choisir Maison ou appartement
    # Puis pour chaque sous-choix, afficher :
    # l'ensemble des biens, la moyenne de la valeur foncière ou la répartition des mutations
    option = st.sidebar.selectbox('Choisir une region', ('Rhône', 'Paris', 'Bouches-du-Rhône'))
    if option == 'Rhône':
        st.subheader("Région Rhône-Alpes")

        if st.checkbox("Appuyer pour consulter l'ensemble des biens"):
            st.write(batiment69)

        if st.checkbox("Appuyer pour consulter la moyenne de la valeure foncière en Rhône-Alpes"):
            st.title(round(Valeur_moyenne69,2))
            st.write(batiment_info69)

        if st.checkbox('Afficher la répartition des natures de mutation'):
            ax = sns.countplot(y="nature_mutation", hue="code_departement", data=oo)
            st.pyplot()
            st.write(
                "D' après ce graphique nous pouvons en deduire que les majeures transactions sont des ventes immobilières ")

        option1 = st.sidebar.radio('Choisissez votre type de local', ('Maison (Département : 69)', 'Appartement (Département : 69)'))

        if option1 == 'Maison (Département : 69)':
            st.header("Maison")
            option2 = st.radio('Que voulez vous savoir sur les maisons ?',
                               ('Emplacement', 'Valeurs foncière', 'Autres informations'))
            if option2 == 'Emplacement':
                st.map(df_map69)
            if option2 == 'Valeurs foncière':
                st.write(batiment69)
                ag = st.slider('Choissisez le max pour la valeur foncière', 0, 290000)
                if ag < 95000:
                    st.map(map69maisonprix)
                if 275706 < ag:
                    st.map(map69appartprix3)
            if option2 == 'Autres informations':
                st.subheader('Barchart')
                chart = alt.Chart(df_map69).mark_bar().encode(
                    x='nom_commune',
                    y='valeur_fonciere'
                )
                chart = chart.properties(
                    width=alt.Step(80)
                )
                st.write(chart)

                st.subheader('Répartition des appartements selon leur valeur foncière')
                regionlyonM.plot.scatter(x='code_commune', y='valeur_fonciere', c='DarkBlue', marker='o')
                st.pyplot()

                st.subheader('Nombre de ventes par mois')
                month1 = lap('month', df_map69)
                st.bar_chart(month1)

                st.subheader('Répartition des type de locals sur toute la France')
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True,
                        startangle=65)
                ax1.axis('equal')
                st.pyplot(fig1)
                st.text(
                    "Nous pouvons voir que à l'échelle de la France, il y a environ le même nombre de maison et d'appartement")
                st.text("Cependant un grand nombre de biens ne sont pas identifiés")

        if option1 == 'Appartement (Département : 69)':
            st.header("Appartement")
            option8 = st.radio('Que voulez vous savoir sur les maisons ?',
                               ('Emplacements', 'Valeurs foncières', 'Autres informations'))
            if option8 == 'Emplacements':
                st.map(df_mapp69)
            if option8 == 'Valeurs foncières':
                st.write(batiment69)
                a1 = st.slider('Choissisez le maximum pour la valeur foncière', 0, 290000)
                if a1 < 65000:
                    st.map(map69appartprix)
                if 275706 < a1:
                    st.map(map69appartprix3)
            if option8 == 'Autres informations':
                st.subheader('Barchart')
                chart = alt.Chart(df_mapp69).mark_bar().encode(
                    x='nom_commune',
                    y='valeur_fonciere'
                )
                chart = chart.properties(
                    width=alt.Step(80)
                )
                st.write(chart)

                st.subheader('Répartition des appartements selon leur valeur foncière')
                regionlyon.plot.scatter(x='code_commune', y='valeur_fonciere', c='DarkBlue', marker='o')
                st.pyplot()

                st.subheader('Nombre de ventes par mois')
                month2 = lap('month', df_mapp69)
                st.bar_chart(month2)

                st.subheader('Répartition des type de locals sur toute la France')
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True,
                        startangle=65)
                ax1.axis('equal')
                st.pyplot(fig1)
                st.text(
                    "Nous pouvons voir que à l'échelle de la France, il y a environ le même nombre de maison et d'appartement")
                st.text("Cependant un grand nombre de biens ne sont pas identifiés")

    if option == 'Paris':
        st.subheader("Région Ile-de-France")

        if st.checkbox("Appuyer pour consulter l'ensemble des biens"):
            st.write(batiment75)

        if st.checkbox('Afficher la répartition des natures de mutation'):
            ax = sns.countplot(y="nature_mutation", hue="code_departement", data=g)
            st.pyplot()
            st.write(
                "D' après ce graphique nous pouvons en deduire que les majeures transactions sont des ventes immobilières ")

        if st.checkbox("Appuyer pour consulter la moyenne de la valeure foncière en Iles-de-France"):
            st.title("Moyenne de la valeur foncière en €")
            st.title(round(Valeur_moyenne75, 2))

        option6 = st.sidebar.radio('Choisissez votre type de local', ('Maison (Département : 75)', 'Appartement (Département : 75)'))

        if option6 == 'Maison (Département : 75)':
            st.header("Maison")
            option4 = st.radio('Que voulez vous savoir sur les maisons ?',
                               ('Emplacements', 'Valeurs foncières', 'Autres informations'))
            if option4 == 'Emplacements':
                st.map(df_mapp75)
            if option4 == 'Valeurs foncières':
                st.write(batiment75)
                a2 = st.slider('Choissisez le maximum pour la valeur foncière', 0, 290000)
                if a2 < 95000:
                    st.map(map75maisonprix)
                if 255706 < a2:
                    st.map(map75maisonprix3)
            if option4 == 'Autres informations':
                st.subheader('Barchart')
                chart = alt.Chart(df_mapp75).mark_bar().encode(
                    x='nom_commune',
                    y='valeur_fonciere'
                )
                chart = chart.properties(
                    width=alt.Step(80)
                )
                st.write(chart)

                st.subheader('Répartition des appartements selon leur valeur foncière')
                region75M.plot.scatter(x='code_commune', y='valeur_fonciere', c='DarkBlue', marker='o')
                st.pyplot()
                st.set_option('deprecation.showPyplotGlobalUse', False)

                st.subheader('Nombre de ventes par mois')
                month3 = lap('month', df_mapp75)
                st.bar_chart(month3)

                st.subheader('Répartition des type de locals sur toute la France')
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True,
                        startangle=65)
                ax1.axis('equal')
                st.pyplot(fig1)
                st.text(
                    "Nous pouvons voir que à l'échelle de la France, il y a environ le même nombre de maison et d'appartement")
                st.text("Cependant un grand nombre de biens ne sont pas identifiés")

        if option6 == 'Appartement (Département : 75)':
            st.header("Appartement")
            option8 = st.radio('Que voulez vous savoir sur les maisons ?',
                               ('Emplacements', 'Valeurs foncières', 'Autres informations'))
            if option8 == 'Emplacements':
                st.map(df_mappA75)
            if option8 == 'Valeurs foncières':
                st.write(batiment75)
                a6 = st.slider('Choissisez le maximum pour la valeur foncière', 0, 290000)
                if a6 < 95000:
                    st.map(map75appartprix)
                if 255706 < a6:
                    st.map(map75appartprix3)

            if option8 == 'Autres informations':
                st.subheader('Barchart')
                chart = alt.Chart(df_mappA75).mark_bar().encode(
                    x='nom_commune',
                    y='valeur_fonciere'
                )
                chart = chart.properties(
                    width=alt.Step(80)
                )
                st.write(chart)

                st.subheader('Répartition des appartements selon leur valeur foncière')
                region75.plot.scatter(x='code_commune', y='valeur_fonciere', c='DarkBlue', marker='o')
                st.pyplot()

                st.subheader('Nombre de ventes par mois')
                month6 = lap('month', df_mappA75)
                st.bar_chart(month6)

                st.subheader('Répartition des type de locals sur toute la France')
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True,
                        startangle=65)
                ax1.axis('equal')
                st.pyplot(fig1)
                st.text(
                    "Nous pouvons voir que à l'échelle de la France, il y a environ le même nombre de maison et d'appartement")
                st.text("Cependant un grand nombre de biens ne sont pas identifiés")

    if option == 'Bouches-du-Rhône':
        st.subheader("Région Bouche-du-Rhône")

        if st.checkbox("Appuyer pour consulter l'ensemble des biens"):
            st.write(batiment13)
        if st.checkbox("Appuyer pour consulter la moyenne de la valeure foncière au Bouche-du-Rhônes"):
            st.title(round(Valeur_moyenne13,2))

        if st.checkbox('Afficher la répartition des natures de mutation'):
            ax = sns.countplot(y="nature_mutation", hue="code_departement", data=oo)
            st.pyplot()
            st.write(
                "D' après ce graphique nous pouvons en deduire que les majeures transactions sont des ventes immobilières ")

        option3 = st.sidebar.radio('Choisissez votre type de local', ('Maison (Département : 13)', 'Appartement (Département : 13)'))

        if option3 == 'Maison (Département : 13)':
            st.header("Maison")
            option4 = st.radio('Que voulez vous savoir sur les maisons ?',
                               ('Emplacements', 'Valeurs foncières', 'Autres informations'))
            if option4 == 'Emplacements':
                st.map(df_mapM13)
            if option4 == 'Valeurs foncières':
                st.write(batiment13)
                a3 = st.slider('Choissisez le maximum pour la valeur foncière', 0, 290000)
                if a3 < 95000:
                    st.map(map13maisonprix)
                if 265706 < a3:
                    st.map(map13maisonprix3)
            if option4 == 'Autres informations':
                st.subheader('Barchart')
                chart = alt.Chart(df_mapM13).mark_bar().encode(
                    x='nom_commune',
                    y='valeur_fonciere'
                )
                chart = chart.properties(
                    width=alt.Step(80)
                )

                st.write(chart)

                st.subheader('Répartition des appartements selon leur valeur foncière')
                region13M.plot.scatter(x='code_commune', y='valeur_fonciere', c='DarkBlue', marker='o')
                st.pyplot()

                st.subheader('Nombre de ventes par mois')
                month5 = lap('month', df_mapM13)
                st.bar_chart(month5)

                st.subheader('Répartition des type de locals sur toute la France')
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True,
                        startangle=65)
                ax1.axis('equal')
                st.pyplot(fig1)
                st.text(
                    "Nous pouvons voir que à l'échelle de la France, il y a environ le même nombre de maison et d'appartement")
                st.text("Cependant un grand nombre de biens ne sont pas identifiés")

        if option3 == 'Appartement (Département : 13)':
            st.header("Appartement")
            option9 = st.radio('Que voulez vous savoir sur les maisons ?',
                               ('Emplacements', 'Valeurs foncières', 'Autres informations'))
            if option9 == 'Emplacements':
                st.map(df_map13)
            if option9 == 'Valeurs foncières':
                st.write(batiment13)
                a5 = st.slider('Choissisez le maximum pour la valeur foncière', 0, 290000)
                if a5 < 95000:
                    st.map(map13appartprix)
                if 255706 < a5:
                    st.map(map13appartprix3)
            if option9 == 'Autres informations':
                st.subheader('Barchart')
                chart = alt.Chart(df_map13).mark_bar().encode(
                    x='nom_commune',
                    y='valeur_fonciere'
                )
                chart = chart.properties(
                    width=alt.Step(80)
                )

                st.subheader('Répartition des appartements selon leur valeur foncière')
                region13.plot.scatter(x='code_commune', y='valeur_fonciere', c='DarkBlue', marker='o')
                st.pyplot()

                st.subheader('Nombre de ventes par mois')
                month4 = lap('month', df_map13)
                st.bar_chart(month4)

                st.subheader('Répartition des types de locals sur toute la France')
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True,
                        startangle=65)
                ax1.axis('equal')
                st.pyplot(fig1)
                st.text(
                    "Nous pouvons voir que à l'échelle de la France, il y a environ le même nombre de maison et d'appartement")
                st.text("Cependant un grand nombre de biens ne sont pas identifiés")

# Implémentation des fonction du sous-programme principal de l'exploration détaillée : map_info():
@st.cache()
def get_mean69():
    batiments = data[
        ['code_departement', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'code_type_local',
         'type_local']].drop_duplicates()
    batiments = batiments.mask(batiments["code_departement"] != 69)
    batiments = batiments.dropna()
    batiment_mean = batiments['valeur_fonciere'].mean()

    batiments = batiments.sort_values(by=['code_type_local'])

    return batiment_mean


@st.cache()
def get_prix_bati69():
    batiments = data[
        ['code_departement', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'code_type_local',
         'type_local']].drop_duplicates()
    batiments = batiments.mask(batiments["code_departement"] != 69)
    batiment_prix69 = batiments['valeur_fonciere']
    batiments = batiments.sort_values(by=['code_type_local'])
    return batiment_prix69


@st.cache()
def get_info69():
    batiments = data[
        ['code_departement', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'code_type_local',
         'type_local']].drop_duplicates()
    batiments = batiments.mask(batiments["code_departement"] != 69)
    batiments = batiments.dropna()
    batiment_info = batiments.describe()
    return batiment_info


@st.cache()
def get_nomCommune69():
    nom_commune69 = data[['nom_commune', 'code_departement', 'nature_mutation', ]].drop_duplicates()
    nom_commune_69 = nom_commune69.mask(nom_commune69['code_departement'] != 69)
    nom_commune_69_ = nom_commune_69.dropna()

    return nom_commune_69_


@st.cache()
def get_nomCommune75():
    nom_commune75 = data[['nom_commune', 'code_departement', 'nature_mutation']].drop_duplicates()
    nom_commune_75 = nom_commune75.mask(nom_commune75['code_departement'] != 75)
    nom_commune_75_ = nom_commune_75.dropna()

    return nom_commune_75_


@st.cache()
def get_nomCommune13():
    nom_commune75 = data[['nom_commune', 'code_departement', 'nature_mutation']].drop_duplicates()
    nom_commune_75 = nom_commune75.mask(nom_commune75['code_departement'] != 13)
    nom_commune_75_ = nom_commune_75.dropna()

    return nom_commune_75_


@st.cache()
def get_mean75():
    batiments = data[
        ['code_departement', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'code_type_local',
         'type_local']].drop_duplicates()
    batiments = batiments.mask(batiments["code_departement"] != 75)
    batiments = batiments.dropna()
    batiments = batiments.sort_values(by=['code_type_local'])
    batiment_mean = batiments['valeur_fonciere'].mean()
    return batiment_mean


@st.cache()
def get_mean13():
    batiments = data[
        ['code_departement', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'code_type_local',
         'type_local']].drop_duplicates()
    batiments = batiments.mask(batiments["code_departement"] != 13)
    batiments = batiments.dropna()
    batiments = batiments.sort_values(by=['code_type_local'])
    batiment_mean = batiments['valeur_fonciere'].mean()
    return batiment_mean


@st.cache()
def get_batiment69():
    batiments = data[
        ['code_departement', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'code_type_local',
         'type_local']].drop_duplicates()
    batiments = batiments.mask(batiments["code_departement"] != 69)
    batiments = batiments.sort_values(by=['code_type_local'])
    batiments = batiments.dropna()
    return batiments


@st.cache()
def get_batiment75():
    batiments = data[
        ['code_departement', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'code_type_local',
         'type_local']].drop_duplicates()
    batiments = batiments.mask(batiments["code_departement"] != 75)
    batiments = batiments.dropna()

    batiments = batiments.sort_values(by=['code_type_local'])

    return batiments


@st.cache()
def get_batiment13():
    batiments = data[
        ['code_departement', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'code_type_local',
         'type_local']].drop_duplicates()
    batiments = batiments.mask(batiments["code_departement"] != 13)
    batiments = batiments.dropna()
    batiments = batiments.sort_values(by=['code_type_local'])

    return batiments


# Dataframe pour plot valeur fonciere en fonction de la surface des appart region 69
def df4():
    df = data[
        ['code_departement', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'code_type_local',
         'type_local']].drop_duplicates()
    df1 = df.mask(df["code_departement"] != 69)
    df1 = df1.mask(df1["surface_reelle_bati"] > 107)
    df1 = df1.mask(df1["valeur_fonciere"] > 25706)  # 3ème quartile
    df1 = df1.mask(df1["type_local"] != 'Appartement')
    df1 = df1.dropna()
    df1 = df1.sort_values(by=['surface_reelle_bati'])
    return df1


df4 = df4()


#69 appart slide
def map69appartprix():
    data.dropna(subset=["longitude"], inplace=True)
    data.dropna(subset=["latitude"], inplace=True)
    df_mapprix69 = data.mask(data["code_departement"] != 69)
    df_mapappart69 = df_mapprix69.mask(df_mapprix69["type_local"] != 'Appartement')
    df_mapprix69 = df_mapappart69.mask(df_mapappart69["valeur_fonciere"] >= 65000)
    df_mapprix69.dropna(subset=["longitude"], inplace=True)
    df_mapprix69.dropna(subset=["latitude"], inplace=True)
    return df_mapprix69


map69appartprix = map69appartprix()


def map69appartprix3():
    df_mapprix69_3 = data.mask(data["code_departement"] != 69)
    df_mapappart69_3_ = df_mapprix69_3.mask(df_mapprix69_3["type_local"] != 'Appartement')
    df_mapprix69_3_ = df_mapappart69_3_.mask(275706 < df_mapappart69_3_["valeur_fonciere"])
    df_mapprix69_3_.dropna(subset=["longitude"], inplace=True)
    df_mapprix69_3_.dropna(subset=["latitude"], inplace=True)
    return df_mapprix69_3_


map69appartprix3 = map69appartprix3()


#69 maison slide

def map69maisonprix():
    df_mapprix69_3 = data.mask(data["code_departement"] != 69)
    df_mapmaison69 = df_mapprix69_3.mask(df_mapprix69_3["type_local"] != 'Maison')
    df_mapmaison69 = df_mapmaison69.mask(df_mapmaison69["valeur_fonciere"] >= 95000)
    df_mapmaison69.dropna(subset=["longitude"], inplace=True)
    df_mapmaison69.dropna(subset=["latitude"], inplace=True)
    return df_mapmaison69


map69maisonprix = map69maisonprix()


def map69maisonprix3():
    df_mapprix69_3 = data.mask(data["code_departement"] != 69)
    df_mapmaison69 = df_mapprix69_3.mask(df_mapprix69_3["type_local"] != 'Maison')
    df_mapmaison69 = df_mapmaison69.mask(275706 < df_mapmaison69["valeur_fonciere"])
    df_mapmaison69.dropna(subset=["longitude"], inplace=True)
    df_mapmaison69.dropna(subset=["latitude"], inplace=True)
    return df_mapmaison69


map69maisonprix3 = map69maisonprix3()


#silde 75 maison

def map75maisonprix():
    df_mapmaison75_3 = data.mask(data["code_departement"] != 75)
    df_mapmaison75 = df_mapmaison75_3.mask(df_mapmaison75_3["type_local"] != 'Maison')
    df_mapmaison75 = df_mapmaison75.mask(df_mapmaison75["valeur_fonciere"] >= 95000)
    df_mapmaison75.dropna(subset=["longitude"], inplace=True)
    df_mapmaison75.dropna(subset=["latitude"], inplace=True)
    return df_mapmaison75


map75maisonprix = map75maisonprix()


def map75maisonprix3():
    df_mapprix69_3 = data.mask(data["code_departement"] != 75)
    df_mapmaison75 = df_mapprix69_3.mask(df_mapprix69_3["type_local"] != 'Maison')
    df_mapmaison75 = df_mapmaison75.mask(275706 < df_mapmaison75["valeur_fonciere"])
    df_mapmaison75.dropna(subset=["longitude"], inplace=True)
    df_mapmaison75.dropna(subset=["latitude"], inplace=True)
    return df_mapmaison75


map75maisonprix3 = map75maisonprix3()


#slide 75 appart

def map75appartprix():
    data.dropna(subset=["longitude"], inplace=True)
    data.dropna(subset=["latitude"], inplace=True)
    df_mapprix75 = data.mask(data["code_departement"] != 75)
    df_mapappart75 = df_mapprix75.mask(df_mapprix75["type_local"] != 'Appartement')
    df_mapprix75 = df_mapappart75.mask(df_mapappart75["valeur_fonciere"] >= 65000)
    df_mapprix75.dropna(subset=["longitude"], inplace=True)
    df_mapprix75.dropna(subset=["latitude"], inplace=True)
    return df_mapprix75


map75appartprix = map75appartprix()


def map75appartprix3():
    df_mapprix75_3 = data.mask(data["code_departement"] != 75)
    df_mapappart75_3_ = df_mapprix75_3.mask(df_mapprix75_3["type_local"] != 'Appartement')
    df_mapprix75_3_ = df_mapappart75_3_.mask(275706 < df_mapappart75_3_["valeur_fonciere"])
    df_mapprix75_3_.dropna(subset=["longitude"], inplace=True)
    df_mapprix75_3_.dropna(subset=["latitude"], inplace=True)
    return df_mapprix75_3_


map75appartprix3 = map75appartprix3()


#slide 13 appart

def map13appartprix():
    df_mapprix13 = data.mask(data["code_departement"] != 13)
    df_mapappart13 = df_mapprix13.mask(df_mapprix13["type_local"] != 'Appartement')
    df_mapprix13 = df_mapappart13.mask(df_mapappart13["valeur_fonciere"] >= 65000)
    df_mapprix13.dropna(subset=["longitude"], inplace=True)
    df_mapprix13.dropna(subset=["latitude"], inplace=True)
    return df_mapprix13


map13appartprix = map13appartprix()


def map13appartprix3():
    df_mapprix13_3 = data.mask(data["code_departement"] != 13)
    df_mapappart13_3_ = df_mapprix13_3.mask(df_mapprix13_3["type_local"] != 'Appartement')
    df_mapprix13_3_ = df_mapappart13_3_.mask(275706 < df_mapappart13_3_["valeur_fonciere"])
    df_mapprix13_3_.dropna(subset=["longitude"], inplace=True)
    df_mapprix13_3_.dropna(subset=["latitude"], inplace=True)
    return df_mapprix13_3_


map13appartprix3 = map13appartprix3()


#slide 13 maison
def map13maisonprix():
    df_mapprix13_3 = data.mask(data["code_departement"] != 13)
    df_mapmaison13 = df_mapprix13_3.mask(df_mapprix13_3["type_local"] != 'Maison')
    df_mapmaison13 = df_mapmaison13.mask(df_mapmaison13["valeur_fonciere"] >= 95000)
    df_mapmaison13.dropna(subset=["longitude"], inplace=True)
    df_mapmaison13.dropna(subset=["latitude"], inplace=True)
    return df_mapmaison13


map13maisonprix = map13maisonprix()


def map13maisonprix3():
    df_mapprix13_3 = data.mask(data["code_departement"] != 13)
    df_mapmaison13 = df_mapprix13_3.mask(df_mapprix13_3["type_local"] != 'Maison')
    df_mapmaison13 = df_mapmaison13.mask(275706 < df_mapmaison13["valeur_fonciere"])
    df_mapmaison13.dropna(subset=["longitude"], inplace=True)
    df_mapmaison13.dropna(subset=["latitude"], inplace=True)
    return df_mapmaison13


map13maisonprix3 = map13maisonprix3()

labels = 'Donées Manquantes', 'Maison', 'Appartement', 'Dépendance', 'Local industriel, commercial ou assimilé'
colors = '#7D43B0', '#D5FBE4', '#0EC736', '#FEBBDC', '#4400FF'
sizes = [41, 21.4, 18, 12.9, 0.37]
explode = (0.05, 0.05, 0.1, 0.1, 0.1)


def regionlyon():
    df = data[
        ['code_departement', 'code_commune', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales',
         'code_type_local', 'type_local']].drop_duplicates()
    df1 = df.mask(df["code_departement"] != 69)
    df1 = df1.mask(df1["surface_reelle_bati"] > 100)
    df1 = df1.mask(df1["valeur_fonciere"] > 275606)
    df1 = df1.mask(df1["type_local"] != 'Appartement')
    df1 = df1.dropna()
    df1 = df1.sort_values(by=['code_commune'])
    return df1


regionlyon = regionlyon()


def regionlyonM():
    df = data[
        ['code_departement', 'code_commune', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales',
         'code_type_local', 'type_local']].drop_duplicates()
    df1 = df.mask(df["code_departement"] != 69)
    df1 = df1.mask(df1["surface_reelle_bati"] > 100)
    df1 = df1.mask(df1["valeur_fonciere"] > 275606)
    df1M = df1.mask(df1["type_local"] != 'Maison')
    dfM = df1.dropna()
    dfM = df1.sort_values(by=['code_commune'])
    return dfM


regionlyonM = regionlyonM()


#75
def region75():
    df = data[
        ['code_departement', 'code_commune', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales',
         'code_type_local', 'type_local']].drop_duplicates()
    df1 = df.mask(df["code_departement"] != 75)
    df1 = df1.mask(df1["surface_reelle_bati"] > 100)
    df1 = df1.mask(df1["valeur_fonciere"] > 275606)
    df1 = df1.mask(df1["type_local"] != 'Appartement')
    df1 = df1.dropna()
    df1 = df1.sort_values(by=['code_commune'])
    return df1


region75 = region75()


def region75M():
    df = data[
        ['code_departement', 'code_commune', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales',
         'code_type_local', 'type_local']].drop_duplicates()
    df1 = df.mask(df["code_departement"] != 75)
    df1 = df1.mask(df1["surface_reelle_bati"] > 100)
    df1 = df1.mask(df1["valeur_fonciere"] > 275606)
    df1M = df1.mask(df1["type_local"] != 'Maison')
    dfM = df1.dropna()
    dfM = df1.sort_values(by=['code_commune'])
    return dfM


region75M = region75M()


#13

def region13():
    df = data[
        ['code_departement', 'code_commune', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales',
         'code_type_local', 'type_local']].drop_duplicates()
    df1 = df.mask(df["code_departement"] != 13)
    df1 = df1.mask(df1["surface_reelle_bati"] > 100)
    df1 = df1.mask(df1["valeur_fonciere"] > 275606)
    df1 = df1.mask(df1["type_local"] != 'Appartement')
    df1 = df1.dropna()
    df1 = df1.sort_values(by=['code_commune'])
    return df1


region13 = region13()


def region13M():
    df = data[
        ['code_departement', 'code_commune', 'valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales',
         'code_type_local', 'type_local']].drop_duplicates()
    df1 = df.mask(df["code_departement"] != 13)
    df1 = df1.mask(df1["surface_reelle_bati"] > 100)
    df1 = df1.mask(df1["valeur_fonciere"] > 275606)
    df1M = df1.mask(df1["type_local"] != 'Maison')
    dfM = df1.dropna()
    dfM = df1.sort_values(by=['code_commune'])
    return dfM


region13M = region13M()


def get_month(dt):
    return dt.month


def count_rows(rows):
    return len(rows)


@st.cache(suppress_st_warning=True)
def lap(laptime, data):
    return data.groupby(laptime).apply(count_rows)

#########Le programme principal################################
st.sidebar.image('https://www.efrei.fr/wp-content/uploads/2019/06/Logo-Efrei-2017-Fr-Web.png',
                 width=150, caption='Farès FADILI, M1-APP-BD Projet Data Vizualisation')
#st.sidebar.caption("Farès FADILI")

def main():
    
    task = st.sidebar.radio('Fonctionnalités', ['Accueil','Aperçu rapide',
                                                'Pré-traitements', 'Pandas profiling',
                                                'SweetViz report', 'Exploration détaillée des données'], 0)
    
    transformation(data)
    df = data
    
    if task == 'Accueil' :
        st.image('https://media.istockphoto.com/vectors/luxury-real-estate-agent-key-logo-vector-id1127282505?k=20&m=1127282505&s='
                         '170667a&w=0&h=qTnpsJnUv_wn3imUXzNKu8NPJFJCioHtx_9JK_7anRA=')
        st.subheader('Bienvenue dans notre application')
        st.write("Voici l'application qui vous permettra "
                 "de mieux visualiser et comprendre l'ensemble des données immobilières de l'année 2020...")
 
    elif task == 'Aperçu rapide':
        st.header('Faites-vous une idée globale')
        st.write("Parcourez les données grâce à un aperçu dataset...")
        df["code_commune"] = df["code_commune"].astype(str)
        df["code_departement"] = df["code_departement"].astype(str)
        st.write(df.head(15))
        st.write('Carte')
        st.map(df)
        
    elif task == 'Pandas profiling':
        st.title('Ayez une autre vision des donnees')
        st.write("Un rapport Pandas Profiling")
        pr = df.profile_report()
        st_profile_report(pr)
        
    elif task == 'SweetViz report':
        st.write('Le rapport SweetViz arrive')
        mask = df.code_commune.apply(lambda x: isinstance(x, str))
        df = df[~mask]
        sweet_report = sv.analyze(df)
        sweet_report.show_html("rapportSweetVis.html")
        import streamlit.components.v1 as components
        HtmlFile = open("rapportSweetVis.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, width=1300, height=800, scrolling=True)
        
    elif task == 'Pré-traitements':
        st.write('Effectuer des transformations générales sur les données')
        transform(df)
        
    else:
        st.write('Exploration détaillée des données')
        st.write("Quelques graphiques paramétrés : séléctionner les valurs par lesquelles vous voulez cummuler, grouper et trier les valeurs foncières")
        #df["code_commune"] = df["code_commune"].astype(str)
        #df["code_departement"] = df["code_departement"].astype(str)
        df["month"] = df["month"].astype(str)
        #st.write(list(df.columns))
        #columns = df.columns.tolist()
        columns = ["nature_mutation","valeur_fonciere"
                    ,"nom_commune","code_departement"
                    ,"type_local","nature_culture"
                    ,"month"]
        selected_columns = st.multiselect("select column", columns, default="code_departement")
        s = df[selected_columns[0]].str.strip().value_counts()
        trace = go.Bar(x=s.index, y=s.values, showlegend=True)
        df1 = [trace]
        df1.sort()
        layout = go.Layout(title="Valeur sélectionnée par rapport au cummul des valeurs foncières")
        fig = go.Figure(data=df1, layout=layout)
        st.plotly_chart(fig)

        oo = map_info()





if __name__ == "__main__":
    main()
    
    
