## =============================================================================
## ================= INICIO ====================================================
## =============================================================================
# Autor:     Clebson Moura
# Data:      Fevereiro/2020
# Descrição: Projeto - DSA - Data Science Academy
#            Prever se o Usuario fara o Donwload depois de Clicar em um Anuncio
#            Se o Clique he Fraudulento?
#
# Definicao do Problema de Negocio: Prver se o Clique he Fraudulento
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/overview
#
# Configurando o diretorio de trabalho
setwd("C:/FCD/01_projeto01A")
getwd()
# ---------------
# Carrega pacotes
# ---------------
# Se necessario instalar os pacotes que serao utilizados, apenas descomentar
# as linhas de install.packages
#
#install.packages(caret)
#install.packages(class)
#install.packages(DMwR)
#install.packages(e1071)
#install.packages(rpart)
#install.packages(corrplot)
#install.packages(gridExtra)
#
#library(data.table)
#library(lubridate)
#library(ggplot2)
library(caret)
library(class)
library(DMwR)
library(e1071)
library(rpart)
library(corrplot)
library(gridExtra)
#
## =============================================================================
## =============================================================================
## Etapa 1 - Coletando os Dados
## =============================================================================
## =============================================================================
dados <- read.csv("train_sample.csv", stringsAsFactors = FALSE)
#
# Obs: stringsAsFactors = FALSE
#      Nao considere nenhuma variavel numrica como factor(categorica), ou seja,
#      mantanha as variaveis numericas como numerica.
#
# Verificando os tipos de dados e os dados em tabela
str(dados)
head(dados)
View(dados)
## =============================================================================
## =============================================================================
## Etapa 2 - Pre-Processamento
## =============================================================================
## =============================================================================
# Verificando a quantidade de ocorrencia das variaveis, Utilizando tabelas 
# de contingencia
View(table(dados$ip))
View(table(dados$app))
View(table(dados$device))
View(table(dados$os))
View(table(dados$channel))
#
# Correlação entre variaveis numericas
numeric.var <- sapply(dados, is.numeric)
corr.matrix <- cor(dados[,numeric.var])
corr.matrix <- cor(dados[2:5])           # variaveis posição 2 a 5
corrplot(corr.matrix, main="\n\nGrafico de Correlacao para Variaveis Numericas", method="number")
#
# Tratando os campos data
#
# Data no formato YYY-MM-DD
#dados$click_time <- as.POSIXct(dados$click_time,format='%Y-%m-%d')
#dados$attributed_time <- as.POSIXct(dados$attributed_time,format='%Y-%m-%d')
# Dia da semana
#dados$click_time_day_week <- wday(dados$click_time)
#dados$attributed_time_day_week <- wday(dados$attributed_time)
# Ano da data
#dados$click_time_year <- year(dados$click_time)
#dados$attributed_time_year <- year(dados$attributed_time)
# Mes da data
#dados$click_time_month <- month(dados$click_time)
#dados$attributed_time_month <- month(dados$attributed_time)
# Dia da data
#dados$click_time_day <- day(dados$click_time)
#dados$attributed_time_day <- day(dados$attributed_time)
#
# Excluindo variaveis
dados$click_time = NULL
dados$attributed_time = NULL
dados$ip = NULL
#
# Ajustando o label da variavel alvo
dados$is_attributed = sapply(dados$is_attributed, function(x){ifelse(x==0, 'N', 'S')})
#
# Muitos classificadores requerem que as variaveis sejam do tipo Fator
# Passando a variavel alvo categorica para fator
dados$is_attributed <- factor(dados$is_attributed, levels = c("N", "S"), labels = c("N", "S"))
#
# Selecao de variaveis
#
# Separando os dados com as classes que quero prever
treino_rfe <- dados[-5]
#
# Definindo o objeto de controle para o modelo de seleção de variáveis
controle <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
#
# Rodando o algoritmo RFE (Eliminação recursiva de atributos)
resultados <- rfe(treino_rfe, dados$is_attributed, sizes = c(1:4), rfeControl = controle)
# Obs: Considerando apenas as variaveis de 1 a 4(exceto a variavel posicao 5 que alvo)
#
# Resumo dos resultados, onde a coluna Accuracy representa a relevancia da vaiavel
print(resultados)
#
# Lista com as variáveis escolhidas
predictors(resultados)
#
# Plotando os resultados
plot(resultados, type=c("g", "o"))
#
# Passando variaveis categoricas para fator
dados$device <- factor(dados$device)
dados$os <- factor(dados$os)
dados$channel <- factor(dados$channel)
dados$app <- factor(dados$app)
#
# Analisando as variaveis categoricas
# Graficos de barra de variaveis categricas
#
p1 <- ggplot(dados, aes(x=app)) + ggtitle("app") + xlab("APP") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_minimal()
p2 <- ggplot(dados, aes(x=device)) + ggtitle("device") + xlab("device") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_minimal()
p3 <- ggplot(dados, aes(x=os)) + ggtitle("os") + xlab("os") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_minimal()
p4 <- ggplot(dados, aes(x=channel)) + ggtitle("channel") + xlab("channel") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_minimal()
#
# Analise dos resultados de todas as variaveis categoricas juntas
grid.arrange(p1, p2, p3, p4, ncol=2)
# Analise da variavel app
grid.arrange(p1, ncol=2)
# Analise da variavel device
grid.arrange(p2, ncol=2)
# Analise da variavel os
grid.arrange(p3, ncol=2)
# Analise da variavel channel
grid.arrange(p4, ncol=2)
#
# Verificando a proporcao de cada variavel categorica
View(round(prop.table(table(dados$is_attributed)) * 100, digits = 1))
View(round(prop.table(table(dados$device)) * 100, digits = 1)) # Evidenciado que 94% he correspondente a dispositivo=1
#
# Analisando o tipo das variaveis
str(dados)
#
## =============================================================================
## =============================================================================
## Etapa 3: Treinando o modelo com SVM
## =============================================================================
## =============================================================================
#
# Criando dados de treino e dados de teste
# Utilizando separação por range, selecionando para treino(primeiros registros)
# e teste(restantes registros)
dados_treino <- dados[1:3887, ]
dados_teste <- dados[3888:5500, ]
#
# Criando os labels para os dados de treino e de teste
dados_treino_labels <- dados[1:3887, 5]
dados_teste_labels <- dados[3888:5550, 5]
#
# length , verificando o comprimento
length(dados_treino_labels)
length(dados_teste_labels)
#
# Visualizando os dados da separação de treino e teste
View(dados_treino)
View(dados_teste)
#
# Verificando o balanceamento da variavel target
table(dados_treino$is_attributed)
#
# Utilizando a função SMOTE do pacote DMwR p/ realizar o balanceamento da variavel target
# Balanceando dados de treino
dados_treino <- SMOTE(is_attributed ~ ., as.data.frame(dados_treino), k = 3, perc.over = 400, perc.under = 150)
table(dados_treino$is_attributed)
# Balanceando dados de teste
dados_teste <- SMOTE(is_attributed ~ ., as.data.frame(dados_teste), k = 3, perc.over = 400, perc.under = 150)
table(dados_teste$is_attributed)
#
typeColNum <- grep('is_attributed',names(dados))
#
# Criando o modelo
modelo_svm_v1 <- svm(is_attributed ~ .,         # variavel target
                     data = dados_treino,       # dados de treino
                     type = 'C-classification', # tipo de modelo
                     kernel = 'radial')         # kernel
#
# Previsoes
#
# Previsoes nos dados de treino
pred_train <- predict(modelo_svm_v1, dados_treino)
#
# Percentual de previsoes corretas com dataset de treino
# funcao mean no campo target p/ mostrar acuracia
mean(pred_train == dados_treino$is_attributed)
#
# Previsoes nos dados de teste
pred_test <- predict(modelo_svm_v1, dados_teste)
#
# Percentual de previsoes corretas com dataset de teste
# funcao mean no campo target p/ mostrar acuracia
mean(pred_test == dados_teste$is_attributed)
#
# Confusion Matrix
table(pred_test, dados_teste$is_attributed)
#
#
## =============================================================================
## =============================================================================
# Etapa 7: Construindo um Modelo com Algoritmo Random Forest
## =============================================================================
## =============================================================================
# Criando o modelo
modelo_rf_v1 = rpart(is_attributed ~ ., data = dados_treino, control = rpart.control(cp = .0005))
#
# PrevisÃµes nos dados de teste
tree_pred = predict(modelo_rf_v1, dados_teste, type='class')
#
# Percentual de previsoes corretas com dataset de teste
mean(tree_pred==dados_teste$is_attributed)
#
# Confusion Matrix
table(tree_pred, dados_teste$is_attributed)
#
## =============================================================================
## ================= FIM =======================================================
## =============================================================================