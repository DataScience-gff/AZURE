library(Information)
library(tidyverse)
library(openxlsx)
library(readxl)

Data <- read_excel("/Users/juanisaulamejia/Downloads/01 ProvidenceClean.xlsx") %>% 
  select(-c(REMODEL)) %>% as.data.frame()


resultados <- create_infotables(Data, y = "TOTAL VALUE", parallel = T) 

library(glmnet)
X <- as.matrix(Data[, -which(names(Data) == "TOTAL_VALUE")])
Y <- Data$`TOTAL VALUE`

modelo_lasso <- cv.glmnet(X, Y, alpha = 1)
coef(modelo_lasso, s = "lambda.min")  # Variables seleccionadas por LASSO



library(randomForest)
rf_model <- randomForest(`TOTAL VALUE` ~ ., data = Data, importance = TRUE)
importance(rf_model)

library(randomForest)
names(Data) <- trimws(names(Data))
names(Data) <- gsub(" ", "_", names(Data))
rf_model <- randomForest(TOTAL_VALUE ~ ., data = Data, importance = TRUE)
importance(rf_model)  # Ver qué otras variables son relevantes

Data_selected <- Data[, c("TOTAL_VALUE", "TAX", "LIVING_AREA", "LOT_SQFT", "GROSS_AREA", "FLOORS")]

modelo_final <- lm(TOTAL_VALUE ~ ., data = Data_selected)
summary(modelo_final)  # Ver R² y significancia de variables

library(xgboost)
X <- as.matrix(Data_selected[, -1])
Y <- Data_selected$TOTAL_VALUE
dtrain <- xgb.DMatrix(data = X, label = Y)
modelo_xgb <- xgboost(data = dtrain, nrounds = 100, objective = "reg:squarederror")


set.seed(123)  # Para reproducibilidad
train_index <- sample(1:nrow(Data), 0.8 * nrow(Data))  # 80% train, 20% test
train_data <- Data[train_index, ]
test_data  <- Data[-train_index, ]

# Convertir en formato XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(train_data[, -which(names(Data) == "TOTAL_VALUE")]), 
                      label = train_data$TOTAL_VALUE)
dtest  <- xgb.DMatrix(data = as.matrix(test_data[, -which(names(Data) == "TOTAL_VALUE")]), 
                      label = test_data$TOTAL_VALUE)

# Entrenar modelo con datos de entrenamiento
modelo_xgb <- xgboost(data = dtrain, nrounds = 100, objective = "reg:squarederror")

# Evaluar en datos de prueba
pred_test <- predict(modelo_xgb, dtest)
rmse_test <- sqrt(mean((pred_test - test_data$TOTAL_VALUE)^2))
print(rmse_test)  # Comparar RMSE en test vs train


importance_matrix <- xgb.importance(feature_names = colnames(Data[, -which(names(Data) == "TOTAL_VALUE")]), 
                                    model = modelo_xgb)
xgb.plot.importance(importance_matrix)




library(xgboost)

# Filtrar solo TAX como predictor
X_train <- as.matrix(train_data[, "TAX", drop = FALSE])
X_test <- as.matrix(test_data[, "TAX", drop = FALSE])

# Crear matrices para XGBoost
dtrain <- xgb.DMatrix(data = X_train, label = train_data$TOTAL_VALUE)
dtest  <- xgb.DMatrix(data = X_test, label = test_data$TOTAL_VALUE)

# Entrenar modelo XGBoost con solo TAX
modelo_xgb_tax <- xgboost(
  data = dtrain,
  nrounds = 200,
  max_depth = 5,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  objective = "reg:squarederror"
)


# Hacer predicciones en los datos de prueba
predicciones <- predict(modelo_xgb_tax, dtest)

# Agregar predicciones al dataset de prueba
test_data$PRED_TOTAL_VALUE <- predicciones

# Mostrar primeras filas con valores reales vs predichos
head(test_data[, c("TAX", "TOTAL_VALUE", "PRED_TOTAL_VALUE")])

# Calcular RMSE en test
rmse_test <- sqrt(mean((test_data$TOTAL_VALUE - test_data$PRED_TOTAL_VALUE)^2))

# Calcular MAE (error absoluto medio)
mae_test <- mean(abs(test_data$TOTAL_VALUE - test_data$PRED_TOTAL_VALUE))

# Calcular R^2 (coeficiente de determinación)
r2_test <- 1 - sum((test_data$TOTAL_VALUE - test_data$PRED_TOTAL_VALUE)^2) / 
  sum((mean(test_data$TOTAL_VALUE) - test_data$TOTAL_VALUE)^2)

# Mostrar métricas
cat("🔹 RMSE:", rmse_test, "\n🔹 MAE:", mae_test, "\n🔹 R^2:", r2_test)





