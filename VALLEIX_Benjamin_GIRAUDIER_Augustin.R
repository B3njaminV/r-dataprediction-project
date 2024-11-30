# Projet Prédiction de Défauts de Paiement
# Auteurs : VALLEIX Benjamin et GIRAUDIER Augustin

# Installation des bibliothèques nécessaires
install.packages(c("ggplot2", "C50", "cluster", "rpart", "rpart.plot", "tree","tsne"))

# Chargement des bibliothèques nécessaires
library(ggplot2)
library(C50)
library(cluster)
library(rpart)
library(rpart.plot)
library(tree)
library(tsne)

# 1. Chargement et exploration des données
donnees <- read.csv("dataset/projet.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = T)
donnees_prediction <- read.csv("dataset/projet_new.csv", sep = ",", dec = ".", stringsAsFactors = TRUE)

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

# Suppression des variables non pertinentes
donnees <- subset(donnees, select = -c(client, categorie))

# Création de catégories pour les variables continues (pour une meilleure analyse des proportions)
donnees$age_cat <- cut(donnees$age,
                                   breaks=c(0, 25, 35, 45, 55, 100),
                                   labels=c("18-25", "26-35", "36-45", "46-55", "56+"))

donnees$revenus_cat <- cut(donnees$revenus,
                                       breaks=c(0, 25, 50, 100, 200, max(donnees$revenus)),
                                       labels=c("0-25k", "25-50k", "50-100k", "100-200k", ">200k"))

donnees$debcred_cat <- cut(donnees$debcred,
                                       breaks=c(0, 5, 10, 15, 20, max(donnees$debcred)),
                                       labels=c("0-5", "5-10", "10-15", "15-20", ">20"))

donnees$debcarte_cat <- cut(donnees$debcarte,
                                        breaks=c(0,1,2,3,4,6,10,15,20,max(donnees$debcarte)),
                                        labels=c("0","1","2","3","4","5","6","7","8+"))

donnees$autres_cat <- cut(donnees$autres,
                                      breaks=c(0,1,2,3,4,6,10,15,20,max(donnees$autres)),
                                      labels=c("0","1","2","3","4","5","6","7","8+"))

# Analyse exploratoire des données
qplot(age, data=donnees, fill=defaut)
qplot(education, data=donnees, fill=defaut)
qplot(emploi, data=donnees, fill=defaut)
qplot(adresse, data=donnees, fill=defaut)
qplot(revenus, data=donnees, fill=defaut)
qplot(debcred, data=donnees, fill=defaut)
qplot(debcarte, data=donnees, fill=defaut)
qplot(autres, data=donnees, fill=defaut)

# Analyse des proportions de défauts par catégorie
prop.table(table(donnees$age_cat, donnees$defaut))*100
prop.table(table(donnees$education, donnees$defaut))*100
prop.table(table(donnees$emploi, donnees$defaut))*100
prop.table(table(donnees$adresse, donnees$defaut))*100
prop.table(table(donnees$revenus_cat, donnees$defaut))*100
prop.table(table(donnees$debcred_cat, donnees$defaut))*100
prop.table(table(donnees$debcarte_cat, donnees$defaut))*100
prop.table(table(donnees$autres_cat, donnees$defaut))*100

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