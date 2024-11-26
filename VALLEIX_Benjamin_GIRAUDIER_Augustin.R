# Projet Prédiction de Défauts de Paiement
# Auteurs : VALLEIX Benjamin et GIRAUDIER Augustin

# Installation des bibliothèques nécessaires
install.packages(c("tidyverse", "caret", "cluster", "pROC", "randomForest", "e1071", "kernlab"))

# Chargement des bibliothèques nécessaires
library(tidyverse)   # Pour manipulation de données
library(caret)       # Pour preprocessing et machine learning
library(cluster)     # Pour clustering
library(pROC)        # Pour courbes ROC
library(randomForest)# Pour Random Forest
library(e1071)       # Pour SVM
library(kernlab)     # Pour SVM

# 1. Chargement et exploration des données
donnees <- read.csv("dataset/projet.csv", sep = ",", header = TRUE)
donnees_prediction <- read.csv("dataset/projet_new.csv", sep = ",", header = TRUE)

# Vérification des données
str(donnees)
summary(donnees)

# 2. Prétraitement des données
# Gestion des valeurs manquantes
donnees$age[donnees$age == 999] <- NA
donnees$adresse[donnees$adresse == 999] <- NA

# Imputation des valeurs manquantes
donnees$age <- ifelse(is.na(donnees$age), median(donnees$age, na.rm = TRUE), donnees$age)
donnees$adresse <- ifelse(is.na(donnees$adresse), median(donnees$adresse, na.rm = TRUE), donnees$adresse)

# Conversion de variables catégorielles
donnees$education <- factor(donnees$education)
donnees$defaut <- factor(donnees$defaut)

# 3. Clustering
# Sélection des variables pour clustering
variables_clustering <- c("age", "revenus", "debcred", "debcarte", "autres")
donnees_clustering <- scale(donnees[, variables_clustering])

# Détermination du nombre optimal de clusters
set.seed(123)
wss <- sapply(1:10, function(k) {
  kmeans(donnees_clustering, k, nstart = 10)$tot.withinss
})
plot(1:10, wss, type = "b", xlab = "Nombre de clusters", ylab = "WSS")

# Clustering K-means
nb_clusters <- 3  # À ajuster selon le graphique
clusters <- kmeans(donnees_clustering, nb_clusters)
donnees$cluster <- clusters$cluster

# Analyse des clusters
cluster_summary <- aggregate(. ~ cluster, data = donnees[, c(variables_clustering, "cluster", "defaut")],
                             FUN = function(x) {
                               if(is.numeric(x)) mean(x, na.rm = TRUE)
                               else names(sort(table(x), decreasing = TRUE)[1])
                             })
print(cluster_summary)

# 4. Préparation des données pour classification
# Séparation train/test
set.seed(123)
index_train <- createDataPartition(donnees$defaut, p = 0.7, list = FALSE)
train_data <- donnees[index_train, ]
test_data <- donnees[-index_train, ]

# 5. Définition des méthodes d'évaluation
control <- trainControl(method = "cv",
                        number = 10,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)

# 6. Création de plusieurs classifieurs
# Régression logistique
model_logistic <- train(defaut ~ age + revenus + debcred + debcarte + autres,
                        data = train_data,
                        method = "glm",
                        trControl = control)

# Random Forest
model_rf <- train(defaut ~ age + revenus + debcred + debcarte + autres,
                  data = train_data,
                  method = "rf",
                  trControl = control)

# SVM
model_svm <- train(defaut ~ age + revenus + debcred + debcarte + autres,
                   data = train_data,
                   method = "svmRadial",
                   trControl = control)

# 7. Évaluation des modèles
models <- list(Logistic = model_logistic,
               RandomForest = model_rf,
               SVM = model_svm)

resultats <- lapply(models, function(model) {
  predictions <- predict(model, test_data)
  confusionMatrix(predictions, test_data$defaut)
})

# Sélection du meilleur modèle (ici, on choisit selon la précision)
meilleur_modele <- models[[which.max(sapply(resultats, function(x) x$overall['Accuracy']))]]

# 8. Prédiction sur le nouveau jeu de données
predictions_finales <- predict(meilleur_modele, donnees_prediction, type = "prob")
predictions_classe <- predict(meilleur_modele, donnees_prediction)

# Préparation du fichier de résultats
resultats_csv <- data.frame(
  client = donnees_prediction$client,
  classe_predite = predictions_classe,
  probabilite_defaut = predictions_finales$Oui
)

# Écriture du fichier CSV de résultats
write.csv(resultats_csv, file = "VALLEIX_Benjamin_GIRAUDIER_Augustin.csv", row.names = FALSE)