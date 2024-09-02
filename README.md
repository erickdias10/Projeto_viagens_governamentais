# Projeto - Viagens Governamentais

Este projeto foi desenvolvido para realizar a análise e o tratamento de dados de viagens utilizando técnicas de ETL, análise de clusterização e modelagem de séries temporais com ARIMA. Os dados utilizados foram extraídos do Portal da Transparência.

## Tecnologias Utilizadas

- Python
- Pandas
- SQLAlchemy
- scikit-learn
- Statsmodels
- Matplotlib
- Requests

## Estrutura do Projeto

### 1. Carregamento das Bibliotecas
Iniciamos o projeto importando as bibliotecas necessárias para manipulação de dados, conexão com o banco de dados, modelagem e visualização.

### 2. Função de Download e Extração de Arquivos ZIP
Uma função foi criada para baixar e extrair o conteúdo de arquivos ZIP provenientes do Portal da Transparência.

### 3. ETL
As funções de ETL foram desenvolvidas para carregar, padronizar e concatenar os dados de diferentes anos em um único DataFrame. Adicionalmente, foi realizada a limpeza dos dados, removendo duplicatas e preenchendo valores nulos.

### 4. Conexão ao SQL Server
Os dados processados foram carregados em um banco de dados SQL Server para posterior consulta e análise.

### 5. Análise de Clusterização
Foi aplicada a técnica de clusterização K-Means para identificar padrões nos dados de viagens, considerando valores monetários como `valor diárias`, `valor passagens`, entre outros.

### 6. Análise com ARIMA
Utilizando a série temporal `valor diárias`, foi ajustado um modelo ARIMA para prever valores futuros. Esta análise permitiu a identificação de tendências e padrões sazonais.

## Visualizações

### Clusterização
Um gráfico de dispersão foi gerado para visualizar os clusters formados com base nos valores das diárias e passagens.

![Clusterização de Viagens](cluster_plot.png)

### Previsão com ARIMA
Foi criado um gráfico para visualizar as previsões do modelo ARIMA comparadas com os dados históricos.

![Previsão com ARIMA](arima_forecast.png)

### Visualização Dashboard no PowerBI
![PowerBI]([arima_forecast.png](https://app.powerbi.com/view?r=eyJrIjoiOTliZDY1MTAtZGZmNy00MjUwLWIzYTctN2YwYTIxMDI2ZTg3IiwidCI6IjdmZTBkZDY5LTlhYjctNGJjYS05YTg2LThlMjA0ZGFjNWE5MSJ9))


## Como Executar

1. Clone o repositório.
2. Instale as dependências necessárias listadas no arquivo `requirements.txt`.
3. Execute o script Jupyter Notebook ou Python para realizar as análises.

## Melhorias Futuras

- Incluir mais anos de dados para análise.
- Experimentar outras técnicas de modelagem de séries temporais.
- Refinar o processo de ETL para lidar com possíveis outliers.

## Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue ou envie um pull request.

## Licença

Este projeto está licenciado sob a Licença MIT. Consulte o arquivo LICENSE para mais informações.
