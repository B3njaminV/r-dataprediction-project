# Projet Prédiction de Défauts de Paiement
# Auteurs : VALLEIX Benjamin et GIRAUDIER Augustin

# Installation des bibliothèques nécessaires
install.packages(c("ggplot2", "C50", "cluster", "rpart", "rpart.plot", "tree", "tsne", "caret"))

# Chargement des bibliothèques nécessaires
library(ggplot2)
library(C50)
library(cluster)
library(rpart)
library(rpart.plot)
library(tree)
library(tsne)
library(caret)

##########################################
# 1. Chargement et exploration des données
##########################################

donnees <- read.csv("dataset/projet.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = T)
donnees_prediction <- read.csv("dataset/projet_new.csv", sep = ",", dec = ".", stringsAsFactors = T)

# Vérification des données
str(donnees)
summary(donnees)

##############################
# 2. Prétraitement des données
##############################

# Gestion des valeurs manquantes
donnees$age[donnees$age == 999] <- NA
donnees$adresse[donnees$adresse == 999] <- NA
donnees_prediction$age[donnees_prediction$age == 999] <- NA
donnees_prediction$adresse[donnees_prediction$adresse == 999] <- NA

# Imputation des valeurs manquantes
donnees$age <- ifelse(is.na(donnees$age), median(donnees$age, na.rm = TRUE), donnees$age)
donnees$adresse <- ifelse(is.na(donnees$adresse), median(donnees$adresse, na.rm = TRUE), donnees$adresse)
donnees_prediction$age <- ifelse(is.na(donnees_prediction$age), median(donnees$age, na.rm = TRUE), donnees_prediction$age)
donnees_prediction$adresse <- ifelse(is.na(donnees_prediction$adresse), median(donnees$adresse, na.rm = TRUE), donnees_prediction$adresse)

# Suppression des variables non pertinentes
donnees <- subset(donnees, select = -c(client, categorie))
donnees_prediction <- subset(donnees_prediction, select = -c(client, categorie))

# Création de catégories pour les variables continues (pour une meilleure analyse des proportions)
donnees$age_cat <- cut(donnees$age,
                       breaks = c(0, 25, 35, 45, 55, 100),
                       labels = c("18-25", "26-35", "36-45", "46-55", "56+"))
donnees_prediction$age_cat <- cut(donnees_prediction$age,
                                 breaks = c(0, 25, 35, 45, 55, 100),
                                 labels = c("18-25", "26-35", "36-45", "46-55", "56+"))

donnees$revenus_cat <- cut(donnees$revenus,
                           breaks = c(0, 25, 50, 100, 200, max(donnees$revenus)),
                           labels = c("0-25k", "25-50k", "50-100k", "100-200k", ">200k"))
donnees_prediction$revenus_cat <- cut(donnees_prediction$revenus,
                                     breaks = c(0, 25, 50, 100, 200, max(donnees$revenus)),
                                     labels = c("0-25k", "25-50k", "50-100k", "100-200k", ">200k"))

donnees$debcred_cat <- cut(donnees$debcred,
                           breaks = c(0, 5, 10, 15, 20, max(donnees$debcred)),
                           labels = c("0-5", "5-10", "10-15", "15-20", ">20"))
donnees_prediction$debcred_cat <- cut(donnees_prediction$debcred,
                                     breaks = c(0, 5, 10, 15, 20, max(donnees$debcred)),
                                     labels = c("0-5", "5-10", "10-15", "15-20", ">20"))

donnees$debcarte_cat <- cut(donnees$debcarte,
                            breaks = c(0, 1, 2, 3, 4, 6, 10, 15, 20, max(donnees$debcarte)),
                            labels = c("0", "1", "2", "3", "4", "5", "6", "7", "8+"))
donnees_prediction$debcarte_cat <- cut(donnees_prediction$debcarte,
                                      breaks = c(0, 1, 2, 3, 4, 6, 10, 15, 20, max(donnees$debcarte)),
                                      labels = c("0", "1", "2", "3", "4", "5", "6", "7", "8+"))

donnees$autres_cat <- cut(donnees$autres,
                          breaks = c(0, 1, 2, 3, 4, 6, 10, 15, 20, max(donnees$autres)),
                          labels = c("0", "1", "2", "3", "4", "5", "6", "7", "8+"))
donnees_prediction$autres_cat <- cut(donnees_prediction$autres,
                                    breaks = c(0, 1, 2, 3, 4, 6, 10, 15, 20, max(donnees$autres)),
                                    labels = c("0", "1", "2", "3", "4", "5", "6", "7", "8+"))

#####################################
# 3. Analyse exploratoire des données
#####################################

# Visualisation des données
qplot(age, data = donnees, fill = defaut)
qplot(education, data = donnees, fill = defaut)
qplot(emploi, data = donnees, fill = defaut)
qplot(adresse, data = donnees, fill = defaut)
qplot(revenus, data = donnees, fill = defaut)
qplot(debcred, data = donnees, fill = defaut)
qplot(debcarte, data = donnees, fill = defaut)
qplot(autres, data = donnees, fill = defaut)

# Analyse des proportions de défauts par catégorie
prop.table(table(donnees$age_cat, donnees$defaut)) * 100
prop.table(table(donnees$education, donnees$defaut)) * 100
prop.table(table(donnees$emploi, donnees$defaut)) * 100
prop.table(table(donnees$adresse, donnees$defaut)) * 100
prop.table(table(donnees$revenus_cat, donnees$defaut)) * 100
prop.table(table(donnees$debcred_cat, donnees$defaut)) * 100
prop.table(table(donnees$debcarte_cat, donnees$defaut)) * 100
prop.table(table(donnees$autres_cat, donnees$defaut)) * 100

################################################
# 4. Préparation des données pour classification
################################################

# Séparation train/test
set.seed(123)
index_train <- createDataPartition(donnees$defaut, p = 0.7, list = FALSE)
train_data <- donnees[index_train,]
test_data <- donnees[-index_train,]

#########################################
# 5. Définition des matrices de confusion
#########################################

# C5.0
model_c50 <- C5.0(defaut ~ ., data = train_data, control = C5.0Control(minCases = 10), trials = 10)

# RPART
model_rpart <- rpart(defaut ~ ., data = train_data)

# TREE
model_tree <- tree(defaut ~ ., data = train_data)

# Fonctions d'évaluation des classifieurs
evaluation_classifieur <- function(matrice_confusion) {
  # Rappel = TP / (TP + FN)
  rappel <- matrice_confusion[2, 2] / (matrice_confusion[2, 2] + matrice_confusion[2, 1])
  # Précision = TP / (TP + FP)
  precision <- matrice_confusion[2, 2] / (matrice_confusion[2, 2] + matrice_confusion[1, 2])
  # Spécificité = TN / (TN + FP)
  specificite <- matrice_confusion[1, 1] / (matrice_confusion[1, 1] + matrice_confusion[1, 2])
  # TVN = TN / (TN + FN)
  tvn <- matrice_confusion[1, 1] / (matrice_confusion[1, 1] + matrice_confusion[2, 1])
  # Score = (Rappel * 0.5) + (Précision * 0.3) + (Spécificité * 0.2)
  score <- (rappel * 0.5) +
    (precision * 0.3) +
    (specificite * 0.2)

  return(list(
    "RAPPEL" = rappel,
    "PRECISION" = precision,
    "SPECIFICITE" = specificite,
    "TVN" = tvn,
    "SCORE" = score
  ))
}

# Matrice de confusion
print(table(test_data$defaut, predict(model_c50, test_data, type = "class")))
print(table(test_data$defaut, predict(model_rpart, test_data, type = "class")))
print(table(test_data$defaut, predict(model_tree, test_data, type = "class")))

# Évaluation des modèles
print(evaluation_classifieur(table(test_data$defaut, predict(model_c50, test_data, type = "class"))))
print(evaluation_classifieur(table(test_data$defaut, predict(model_rpart, test_data, type = "class"))))
print(evaluation_classifieur(table(test_data$defaut, predict(model_tree, test_data, type = "class"))))

############################
# 6. APPROCHE PAR CLUSTERING
############################

# Préparation des données
matrix <- daisy(donnees, metric = "gower")
summary(matrix)

### K-means ###
kmeans <- kmeans(matrix, centers = 4)

# Visualisation des clusters
table(kmeans$cluster, donnees$defaut)
qplot(kmeans$cluster, data = donnees, fill = defaut, binwidth = 0.5)

# Analyse des clusters
qplot(age, debcred, data = donnees, color = kmeans$cluster)
qplot(revenus, debcred, data = donnees, color = kmeans$cluster)
qplot(emploi, debcred, data = donnees, color = kmeans$cluster)

donnees_clustered <- data.frame(donnees)
donnees_clustered$Cluster <- kmeans$cluster

### CLUSTERING HIERARCHIQUE AGGLOMERATIF ###
hierarchique <- agnes(matrix, method = "ward")
plot(hierarchique)
rect.hclust(hierarchique, k = 4)
hierarchique_agg <- cutree(hierarchique, k = 4)

### CLUSTERING HIERARCHIQUE DIVISIF ###
hierarchique_div <- diana(matrix, diss = TRUE)
plot(hierarchique_div)
rect.hclust(hierarchique_div, k = 4)
hierarchique_divisif <- cutree(hierarchique_div, k = 4)

### CLUSTERING PAR DENDROGRAMME ###
dendrogramme <- as.dendrogram(hierarchique)
coupage <- cut(dendrogramme, h = 4)
plot(coupage)
groupes <- cutree(coupage, k = 4)

# Comparaison des clusters
table(kmeans$cluster, hierarchique_agg)
table(kmeans$cluster, hierarchique_divisif)
table(kmeans$cluster, groupes)

################
# 7. PREDICTIONS
################

## K-means ##
var_predictives <- c("age", "adresse", "revenus", "debcred", "debcarte", "autres")
donnees_KM <- donnees[, var_predictives]
for(col in vars_num) {
  median_val <- median(donnees_KM[,col], na.rm = TRUE)
  donnees_KM[is.na(donnees_KM[,col]), col] <- median_val
}

# Prédiction
km <- kmeans(donnees_KM, centers=5)
donnees$cluster_km <- km$cluster

