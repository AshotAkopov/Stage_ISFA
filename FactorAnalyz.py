import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# Mes fonctions

# Fonction pour détecter les colonnes fortement corrélées
def corr_over_threshold(df, threshold=0.95, method: str = 'pearson'):
    """
    Identifie les paires de colonnes dans un DataFrame dont la corrélation 
    (absolue) dépasse un certain seuil.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        threshold (float): Seuil de corrélation en valeur absolue.
        method (str): Méthode de corrélation ('pearson', 'spearman', 'kendall').
                      Par défaut 'pearson'.

    Returns:
        pd.Series: Paires de colonnes et leur coefficient de corrélation.
    """
    # 1) Matrice de corrélation absolue selon la méthode choisie
    R = df.corr(method=method).abs()
    
    # 2) Masque du triangle supérieur (k=1 enlève la diagonale)
    mask = np.triu(np.ones(R.shape, dtype=bool), k=1)
    
    # 3) On garde seulement ce triangle
    R_triu = R.where(mask)
    
    # 4) On transforme en série, triée par valeur
    correlations = (
        R_triu.stack()
              .rename('r')
              .sort_values(ascending=False)
    )
    
    # 5) Filtre sur le seuil
    suspects = correlations[correlations > threshold]
    
    return suspects



# Fonction pour extraire les résultats d'une ACP sklearn
def pca_results(pca, X_pca, X_original, index_individus=None):
    """
    Calcule les principaux indicateurs associés à une ACP (PCA sklearn).

    Paramètres
    ----------
    pca : sklearn.decomposition.PCA
        Objet PCA déjà ajusté (fit).
    X_pca : np.ndarray
        Coordonnées des individus dans l’espace des axes principaux (pca.transform(X)).
    X_original : pd.DataFrame
        Données originales utilisées pour la PCA (avant transform).
    index_individus : array-like, optionnel
        Index des individus (ex: noms, identifiants). 
        Si None, utilise un RangeIndex.

    Retour
    ------
    dict de DataFrames :
        - val_propres
        - prop_var_expl
        - coord_individus
        - cos2_individus
        - contrib_indiv
        - vec_propres
        - coord_var
        - cos2_var
        - contrib_var
    """
    # Index des individus
    if index_individus is None:
        index_individus = range(X_pca.shape[0])

    # Valeurs propres
    val_propres = pd.DataFrame(
        pca.explained_variance_,
        columns=['Valeur_propre'],
        index=[f'PC{i+1}' for i in range(len(pca.explained_variance_))]
    )

    # Proportion de variance expliquée
    prop_var_expl = pd.DataFrame(
        pca.explained_variance_ratio_,
        columns=['Proportion_variance_expliquée'],
        index=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))]
    )

    # Coordonnées individus
    coord_individus = pd.DataFrame(
        X_pca,
        index=index_individus,
        columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
    )

    # cos² individus
    cos2_individus = coord_individus ** 2

    # Contributions individus
    n = X_pca.shape[0]
    contrib_indiv = (coord_individus ** 2) / (n * pca.explained_variance_)

    # Vecteurs propres
    vec_propres = pd.DataFrame(
        pca.components_,
        columns=X_original.columns,
        index=[f'PC{i+1}' for i in range(len(pca.components_))]
    )

    # Coordonnées variables
    coord_var = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        columns=[f'PC{i+1}' for i in range(len(pca.explained_variance_))],
        index=X_original.columns
    )

    # cos² variables
    cos2_var = coord_var ** 2

    # Contributions variables
    contrib_var = (coord_var ** 2) / pca.explained_variance_

    return {
        "val_propres": val_propres,
        "prop_var_expl": prop_var_expl,
        "coord_individus": coord_individus,
        "cos2_individus": cos2_individus,
        "contrib_indiv": contrib_indiv,
        "vec_propres": vec_propres,
        "coord_var": coord_var,
        "cos2_var": cos2_var,
        "contrib_var": contrib_var
    }


# Fonction pour tracer la variance expliquée et la variance expliquée cumulée
def double_axis_plot(explained_variance):
    plt.bar(range(1,len(explained_variance)+1),explained_variance)
    plt.ylabel('Explained variance')
    plt.xlabel('Components')
    plt.plot(range(1,len(explained_variance)+1),
         np.cumsum(explained_variance),
         c='red',
         label="Cumulative Explained Variance",
         marker='o')
    plt.legend(loc='upper left')                



# Kaiser plot
def kaiser_plot(val_propres):
    """
    Affiche un scree plot avec le critère de Kaiser et
    imprime combien de valeurs propres dépassent 1.
    
    Paramètres
    ----------
    val_propres : array-like
        Tableau des valeurs propres (déjà ordonnées).
    """

    # 1) Compter les valeurs > 1
    k = int(np.sum(val_propres > 1.0))
    print(f"{k} première(s) valeur(s) propre(s) dépassent 1 (critère de Kaiser).")

    # 2) Graphique
    plt.plot(val_propres, marker='o')
    plt.axhline(y=1.0, linestyle='--', color='red')
    plt.title("Scree plot (critère de Kaiser)")
    plt.xlabel("Composante principale")
    plt.ylabel("Valeur propre")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.show()

# Horn's Parallel Analysis
def parallel_analysis(X, n_iter=500, quantile=0.95, random_state=None):
    """
    Horn's Parallel Analysis pour ACP
    ----------------------------------
    X           : array (n, p) - Données (pas encore standardisées)
    n_iter      : int - nombre de jeux aléatoires simulés
    quantile    : float - quantile utilisé (0.95 = Glorfeld, 0.50 = Horn original)
    random_state: int ou None - graine pour reproductibilité
    
    Retourne :
      k         : nombre de composantes à retenir
      evals_obs : valeurs propres observées
      evals_thr : seuils simulés (quantiles)
    """
    rng = np.random.default_rng(random_state)
    n, p = X.shape

    # ----- Standardiser les données (moyenne 0, écart-type 1)

    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    # ----- Matrice de corrélation + valeurs propres observées

    R = np.corrcoef(X_std,            # matrice de corrélation
                    rowvar=False)     # Indique à NumPy que les variables sont en colonnes et les observations en lignes (format habituel)
    evals_obs = np.linalg.eigvalsh(R)[::-1]  # On calcule les valeurs propres de la matrice de corrélation.
                                             # np.linalg.eigvalsh est une fonction qui calcule les valeurs propres d'une matrice symétrique
                                             # ar défaut, cette fonction les renvoie dans l’ordre croissant.
                                             # En Python, [::-1] est une opération de slicing qui lit la liste ou le tableau à l’envers.       



    # ----- Simulations

    evals_sim = np.empty((n_iter, p)) # On prépare un tableau vide pour stocker les résultats des simulations.
                                      # n_iter = nombre de jeux simulés
                                      # p = nombre de variables (composantes principales)
    

    for i in range(n_iter):
        Z = rng.normal(size=(n, p))                       # On crée un jeu de données aléatoires suivant une loi normale standard (moyenne 0, écart-type 1).
                                                          # Chaque case est un tirage aléatoire de la loi normale standard.
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)  # Standardisation pour que les données simulées aient la même échelle que les données observées.
        Rb = np.corrcoef(Z, rowvar=False)                 # Matrice de corrélation des données simulées.
        evals_sim[i] = np.linalg.eigvalsh(Rb)[::-1]       # On calcule les valeurs propres de la matrice de corrélation des données simulées
                                                          # et on les stocke dans evals_sim par ordre décroissant.

    # ----- Calcul des quantiles (seuils)

    
    #La prochaine étape est de comparer les valeurs propres observées avec les valeurs des données simulées.
    #Nous avons plusieurs simulations et dans la méthode originale de Horn, on prend la moyenne par colonne des valeurs propres simulées.
    #Cependant, Glorfeld 1995 recommande de prendre le quantile 0.95 des valeurs propres simulée car plus conservateur.

    # Rappel : le 95 percentile est la valeur en dessous de laquelle 95% des observations se situent.
   

    evals_thr = np.quantile(evals_sim, quantile, axis=0) # calcul des quantiles (seuils) des valeurs propres simulées par colonne


    # ----- Décision : nombre de composantes où obs > seuil

    k = np.sum(evals_obs > evals_thr) # On compte le nombre de composantes principales où la valeur propre observée est supérieure au seuil simulé.
    print(f"Nombre de composantes retenues : {k}")

    
    #Comme les composantes principales sont ordonnées par ordre décroissant de leur valeur propre, en faisant la somme des composantes où la valeur propre observée est supérieure au seuil simulé,
    # nous obtenons le nombre de composantes principales à retenir.
  

    # ----- Graphique

    plt.plot(range(1, p+1), evals_obs, marker='o', label="Observé")
    plt.plot(range(1, p+1), evals_thr, marker='x', label=f"Seuil {quantile*100:.0f}% simulé")
    plt.axvline(k, color='r', linestyle='--', label=f"{k} composantes retenues")
    plt.xlabel("Composante principale")
    plt.ylabel("Valeur propre")
    plt.title("Horn's Parallel Analysis")
    plt.legend()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Moyenne des corrélations au carré hors diagonale
def average_squared_off_diagonals(R):       # La fonction prend en entrée une matrice de corrélation R
    """Moyenne des corrélations au carré hors diagonale."""
    p = R.shape[0]                          # R shape donne les dimensions de la matrice R. Comme la matrice est carrée, R.shape[0] = R.shape[1]
    mask = ~np.eye(p, dtype=bool)

    # np.eye(N, M=None, k=0, dtype=float) est une fonction qui crée une matrice identité elle a plusieurs arguments :
    # N : nombre de lignes
    # M : nombre de colonnes (si pas précisé, M = N, on a une matrice carrée)
    # k : décalage de la diagonale (0 = on remplit la diagonale principale avec les 1, > 0 = au remplit la diagonale au-dessus de la principale, au remplit la diagonale < 0 = en-dessous) 
    # dtype : type de données (par défaut float)
    # np.eye(p, dtype=bool) crée une matrice identité de taille p x p avec des valeurs booléennes (True sur la diagonale, False ailleurs).

    # ~ est l’opérateur “bitwise NOT” (NON bit à bit). Il a des compretements différents selon le type de données. 
    # Mais avec les tableaux numPy booléens, il inverse les valeurs (True devient False et vice versa).

    # Cela nous donne un masque qui est True pour les éléments hors diagonale et False pour les éléments sur la diagonale.

    R_masked = R[mask]  # On applique le masque pour ne garder que les éléments hors diagonale de la matrice de corrélation R.

    R_masked_2 = R_masked ** 2  # On élève au carré les éléments hors diagonale.

    return np.mean(R_masked_2)  # On calcule la moyenne des éléments hors diagonale au carré et on la retourne.

# Test MAP de Velicer
def velicer_map(X, max_components=None, plot=True, show_values=True, return_avg_sqrs=False, retrn_k_map=False):
    """
    Test MAP de Velicer avec affichage optionnel du graphique.

    Paramètres
    ----------
    X : array (n_samples, n_features)
        Matrice des données.
    max_components : int ou None
        Nombre max de composantes à tester (par défaut p-1).
    plot : bool
        Si True, affiche la courbe des moyennes carrées.
    show_values : bool
        Si True, affiche les valeurs numériques sur le graphique.

    Retour
    ------
    k_map : int
        Nombre optimal de composantes.
    avg_sqrs : list
        Liste des moyennes carrées par étape.
    """
    X = np.asarray(X, dtype=float) # Convertit X en tableau NumPy de type float pour éviter les erreurs de type.
    n, p = X.shape # On récupère le nombre d'observations (n) et le nombre de variables (p) dans la matrice X. mais nous n'allons utilisera que p
    if max_components is None: # Nous avons un paramètre max_components qui permet de limiter le nombre de composantes à tester.                         
        max_components = p - 1 # Si max_components n'est pas spécifié, on le fixe à p - 1 (toutes les composantes sauf la dernière).

    # Standardisation  ----------
    Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    # Étape 0  ----------
    R0 = np.corrcoef(Xs, rowvar=False) # On calcule la matrice de corrélation des données standardisées. 
                                       # rowvar=False indique que les variables sont en colonnes et les observations en lignes.
    
    avg_sqrs = [average_squared_off_diagonals(R0)] # avec average_squared_off_diagonals définie plus haut, on calcule la moyenne des corrélations au carré hors diagonale
                                                   # on utilise [] pour stocker la moyenne dans une liste car nous allons ajouter les moyennes des étapes suivantes à cette liste.
                                                   # pour l'instant cette liste ne contient que la moyenne de l'étape 0.
    # Étapes suivantes  ----------
    for k in range(1, max_components + 1):                   # On applique les étapes du test MAP de Velicer.
        pca = PCA(n_components=k)                            # On définit une PCA avec k composantes principales.
        scores = pca.fit_transform(Xs)                       # On calcule les scores (coordonnées des observations dans le nouvel espace).       
        X_rec = pca.inverse_transform(scores)                # On reconstruit les données à partir des scores pour obtenir les résidus.
        X_resid = Xs - X_rec                                 # On calcule les résidus en soustrayant les données reconstruites des données standardisées.
        Rk = np.corrcoef(X_resid, rowvar=False)              # On calcule la matrice de corrélation des résidus.
        avg_sqrs.append(average_squared_off_diagonals(Rk))   # On ajoute la moyenne des corrélations au carré hors diagonale des résidus à la liste avg_sqrs.

        

    # Étape avec minimum  ----------
    k_map = int(np.argmin(avg_sqrs))                         # On cherche l'indice le plus bas

    # Graphique  ----------
    if plot:
        steps = np.arange(len(avg_sqrs))
        plt.figure()
        plt.plot(steps, avg_sqrs, marker='o')
        plt.axvline(k_map, linestyle='--', color='red', label=f"Minimum à l'étape {k_map}")
        if show_values:
            for i, val in enumerate(avg_sqrs):
                plt.text(i, val, f"{val:.3f}", ha='center', va='bottom', fontsize=8)
        plt.xlabel("Étape (k composantes retirées)")
        plt.ylabel("Moyenne des corrélations² hors diagonale")
        plt.title("Velicer's MAP — courbe des moyennes")
        plt.legend()
        plt.show()

    if return_avg_sqrs:
        return avg_sqrs
    
    if retrn_k_map:
        return k_map
    

# Fonction pour afficher les top et bottom variables/individus selon un indicateur donné
def pca_top(results: dict, indicator='coord_var', dim='PC1', top=10, *, return_frames=False):
    """
    Pour une dimension et un indicateur donnés, affiche les top et bottom 'top' valeurs
    (triées par valeur absolue mais affichées avec leur vrai signe),
    avec les autres indicateurs associés.

    Paramètres
    ----------
    results : dict
        Dictionnaire renvoyé par la fonction pca_results(...).
        Doit contenir les clés :
        'coord_var','cos2_var','contrib_var',
        'coord_individus','cos2_individus','contrib_indiv'.
    indicator : str, par défaut 'coord_var'
        Indicateur à afficher ('coord_var', 'cos2_var', 'contrib_var',
                               'coord_individus', 'cos2_individus', 'contrib_indiv').
    dim : str, par défaut 'PC1'
        Dimension à analyser (ex: 'PC1', 'PC2', etc.).
    top : int, par défaut 10
        Nombre de valeurs à afficher (positives et négatives).
    return_frames : bool, par défaut False
        - False : affiche uniquement les résultats (print) et retourne None.
        - True  : affiche les résultats ET retourne (top_positive_df, top_negative_df).

    Retour
    ------
    None ou tuple(pd.DataFrame, pd.DataFrame)
    """
    indicators_var_names   = ['coord_var', 'cos2_var', 'contrib_var']
    indicators_indiv_names = ['coord_individus', 'cos2_individus', 'contrib_indiv']

    # dictionnaires construits uniquement à partir de results
    indicators_var = {
        "coord_var": results["coord_var"],
        "cos2_var": results["cos2_var"],
        "contrib_var": results["contrib_var"]
    }
    indicators_indiv = {
        "coord_individus": results["coord_individus"],
        "cos2_individus": results["cos2_individus"],
        "contrib_indiv": results["contrib_indiv"]
    }

    if indicator in indicators_var:
        df = indicators_var[indicator]
        indicators = indicators_var.copy()
        del indicators[indicator]
    else:
        df = indicators_indiv[indicator]
        indicators = indicators_indiv.copy()
        del indicators[indicator]

    # Tri par valeur absolue mais récupération des vraies valeurs signées
    principal_sorted_idx = df[dim].abs().sort_values(ascending=False).index
    top_positive = df.loc[principal_sorted_idx].head(top)[dim]
    top_negative = df.loc[principal_sorted_idx].tail(top)[dim]

    # Identifier les deux autres indicateurs associés
    if indicator in indicators_var_names:
        indicators_names = [x for x in indicators_var_names if x != indicator]
    else:
        indicators_names = [x for x in indicators_indiv_names if x != indicator]

    indic_1_pos = indicators[indicators_names[0]].loc[top_positive.index, dim]
    indic_1_neg = indicators[indicators_names[0]].loc[top_negative.index, dim]
    indic_2_pos = indicators[indicators_names[1]].loc[top_positive.index, dim]
    indic_2_neg = indicators[indicators_names[1]].loc[top_negative.index, dim]

    # Construction des DataFrames
    top_positive_df = pd.concat([top_positive, indic_1_pos, indic_2_pos], axis=1)
    top_positive_df.columns = [indicator, indicators_names[0], indicators_names[1]]

    top_negative_df = pd.concat([top_negative, indic_1_neg, indic_2_neg], axis=1)
    top_negative_df.columns = [indicator, indicators_names[0], indicators_names[1]]

    label = "Variable" if indicator in indicators_var_names else "Observation"

    top_positive_df = top_positive_df.reset_index()
    top_positive_df.rename(columns={top_positive_df.columns[0]: label}, inplace=True)

    top_negative_df = top_negative_df.reset_index()
    top_negative_df.rename(columns={top_negative_df.columns[0]: label}, inplace=True)

    # Affichage identique à avant
    print(f"Les {top} plus grandes valeurs pour l'indicateur '{indicator}' sur la dimension '{dim}'")
    print(top_positive_df)
    print()
    print(f"Les {top} plus petites valeurs pour l'indicateur '{indicator}' sur la dimension '{dim}'")
    print(top_negative_df)

    return (top_positive_df, top_negative_df) if return_frames else None


# Fonction pour tracer le cercle de corrélation d'une ACP
def plot_correlation_circle(
    coord_var, feature_names, ax1=1, ax2=2, seuil=None, figsize=(6,6),
    show_unselected=False, unselected_color="#bbbbbb", unselected_alpha=0.4,
    use_cos2=False):

    """
    Trace le cercle de corrélation d’une ACP (Analyse en Composantes Principales).

    Chaque variable est représentée par un vecteur allant de l’origine (0,0) 
    vers ses coordonnées factorielles sur les axes principaux PC1, PC2, etc.
    Ce graphique permet de visualiser la qualité de représentation des variables
    et leurs contributions aux composantes principales.

    Paramètres
    ----------
    coord_var : pandas.DataFrame ou ndarray
        Coordonnées des variables dans le nouvel espace factoriel
        (généralement `results["coord_var"]` renvoyé par pca_results).
    feature_names : list of str
        Noms des variables à afficher.
    ax1 : int, par défaut 1
        Numéro du premier axe à représenter (PC1 = 1).  
        Attentin : numérotation 1-based (donc ax1=1 → première composante).
    ax2 : int, par défaut 2
        Numéro du second axe à représenter (PC2 = 2).
    seuil : None, float ou tuple(float, float), par défaut None
        Filtrage des variables affichées :
          - None → aucun filtrage, toutes les flèches sont affichées en bleu.
          - scalaire (float) → même seuil appliqué aux deux axes.
          - tuple (sx, sy) → seuil distinct par axe.
        Interprétation du seuil :
          - si use_cos2=False → seuil appliqué sur |coordonnée|.
          - si use_cos2=True  → seuil appliqué sur cos² (coordonnée²).
    figsize : tuple, par défaut (6,6)
        Taille de la figure matplotlib.
    show_unselected : bool, par défaut False
        Si True, affiche aussi les variables non retenues par le seuil 
        (en gris translucide).
    unselected_color : str, par défaut "#bbbbbb"
        Couleur des vecteurs non sélectionnés.
    unselected_alpha : float, par défaut 0.4
        Transparence des vecteurs non sélectionnés.
    use_cos2 : bool, par défaut False
        - False → le filtrage utilise les coordonnées factorielles (|coord|).
        - True  → le filtrage utilise la qualité de représentation (cos²).

    Retour
    ------
    matplotlib.axes.Axes
        L’objet Axes de la figure matplotlib (permet de personnaliser ensuite).

    Notes
    -----
    - Les couleurs codent le type de dépassement du seuil :
        * rouge : dépassement uniquement sur PC{ax1}
        * vert  : dépassement uniquement sur PC{ax2}
        * violet : dépassement sur les deux axes
    - Le cercle unité (rayon = 1) est tracé en pointillé pour repère.
    - Une légende explicite est ajoutée si un filtrage est actif.
    """

    # Axes (1-based -> 0-based)
    ax1 -= 1
    ax2 -= 1

    # Numpy array
    coord_array = coord_var.values if hasattr(coord_var, "values") else np.asarray(coord_var)

    # Normalisation du seuil (jamais d'infini dans les libellés)
    mode = "nofilter"
    sx = sy = None
    if seuil is None:
        mode = "nofilter"
    elif np.isscalar(seuil):
        if float(seuil) <= 0:
            mode = "nofilter"
        else:
            sx = sy = float(seuil)
            mode = "both_nonzero"
    else:
        s = list(seuil)
        if len(s) != 2:
            raise ValueError("`seuil` doit être None, un scalaire, ou un tuple (s_PC1, s_PC2).")
        sx, sy = float(s[0]), float(s[1])
        nzx, nzy = sx > 0, sy > 0
        if nzx and nzy:
            mode = "both_nonzero"
        elif nzx or nzy:
            mode = "one_nonzero"
        else:
            mode = "nofilter"

    fig, ax = plt.subplots(figsize=figsize)
    handles, labels = [], []

    for i in range(len(feature_names)):
        x, y = coord_array[i, ax1], coord_array[i, ax2]

        if mode == "nofilter":
            # tout afficher, pas de légende
            color = "blue"
            ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, fc=color, ec=color, alpha=0.8)
            ax.text(x*1.1, y*1.1, feature_names[i], color=color, ha='center', va='center')
            continue

        # Au moins un seuil actif
        if use_cos2:
            # seuils sur cos² (dans [0,1])
            cx, cy = x*x, y*y
            cond_x = (sx is not None and sx > 0 and cx > sx)
            cond_y = (sy is not None and sy > 0 and cy > sy)
        else:
            # seuils sur |coord|
            cond_x = (sx is not None and sx > 0 and abs(x) > sx)
            cond_y = (sy is not None and sy > 0 and abs(y) > sy)

        if cond_x or cond_y:
            if cond_x and cond_y:
                color = "purple"
                label = (f"cos² > ({sx:.3g},{sy:.3g}) sur PC{ax1+1} & PC{ax2+1}"
                         if use_cos2 else
                         f"|coord| > ({sx:.3g},{sy:.3g}) sur PC{ax1+1} & PC{ax2+1}")
            elif cond_x:
                color = "red"
                if mode == "one_nonzero" and not (sy and sy > 0):
                    label = (f"cos² > {sx:.3g} sur PC{ax1+1} (pas de filtre sur PC{ax2+1})"
                             if use_cos2 else
                             f"|coord| > {sx:.3g} sur PC{ax1+1} (pas de filtre sur PC{ax2+1})")
                else:
                    label = (f"cos² > {sx:.3g} sur PC{ax1+1}"
                             if use_cos2 else
                             f"|coord| > {sx:.3g} sur PC{ax1+1}")
            else:  # cond_y
                color = "green"
                if mode == "one_nonzero" and not (sx and sx > 0):
                    label = (f"cos² > {sy:.3g} sur PC{ax2+1} (pas de filtre sur PC{ax1+1})"
                             if use_cos2 else
                             f"|coord| > {sy:.3g} sur PC{ax2+1} (pas de filtre sur PC{ax1+1})")
                else:
                    label = (f"cos² > {sy:.3g} sur PC{ax2+1}"
                             if use_cos2 else
                             f"|coord| > {sy:.3g} sur PC{ax2+1}")

            ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, fc=color, ec=color, alpha=0.8)
            ax.text(x*1.1, y*1.1, feature_names[i], color=color, ha='center', va='center')

            if label not in labels:
                handles.append(plt.Line2D([0], [0], color=color, lw=2))
                labels.append(label)
        else:
            # non retenu par le seuil
            if show_unselected:
                ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.03,
                         fc=unselected_color, ec=unselected_color, alpha=unselected_alpha)
                ax.text(x*1.1, y*1.1, feature_names[i],
                        color=unselected_color, alpha=unselected_alpha,
                        ha='center', va='center')

    # Cercle unité + axes
    ax.add_patch(plt.Circle((0, 0), 1, facecolor='none', edgecolor='gray', linestyle='--'))
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(f"PC{ax1+1}")
    ax.set_ylabel(f"PC{ax2+1}")

    # Titre & légende (en bas à gauche)
    if mode == "nofilter":
        ax.set_title("Cercle de corrélation (toutes variables)")
    else:
        s1 = f"{sx:.3g}" if (sx is not None and sx > 0) else "aucun"
        s2 = f"{sy:.3g}" if (sy is not None and sy > 0) else "aucun"
        metric_label = "cos²" if use_cos2 else "|coord|"
        ax.set_title(f"Cercle de corrélation (seuils {metric_label} PC{ax1+1}={s1}, PC{ax2+1}={s2})")
        if handles:
            ax.legend(handles, labels, loc="lower left")


def calculate_vif(X, order_by_vif=True):
    """
    Calcule le facteur d'inflation de la variance (VIF) pour chaque variable dans un DataFrame.

    Paramètres
    ----------
    X : pd.DataFrame
        DataFrame contenant les variables explicatives.

    Retour
    ------
    vif_data : pd.DataFrame
        DataFrame avec les variables et leurs VIF correspondants.
    """
    # Ajouter une constante (intercept) pour le calcul
    X_with_const = add_constant(X)

    # Calculer le VIF pour chaque variable
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_with_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_with_const.values, i)
        for i in range(X_with_const.shape[1])
    ]

    if order_by_vif:
        vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)

    return vif_data