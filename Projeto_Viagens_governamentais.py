#!/usr/bin/env python
# coding: utf-8

# # Projeto - Viagens

# In[1]:


# Carregando Bibliotecas
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import requests
import zipfile
import io


# In[2]:


# Função para baixar e extrair o conteúdo de um arquivo ZIP
def baixar_e_extrair_zip(url, pasta_destino):
    """
    Baixa um arquivo ZIP da URL fornecida e extrai seu conteúdo para o diretório especificado.
    Args:
    url (str): URL do arquivo ZIP a ser baixado.
    pasta_destino (str): Diretório onde o conteúdo do ZIP será extraído.
    Returns:
    List: Lista dos arquivos extraídos.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Verifica se ocorreu um erro na requisição
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(pasta_destino)
            arquivos = z.namelist()
            print(f"Arquivos extraídos: {arquivos}")
            return arquivos
    except Exception as e:
        print(f"Erro ao baixar ou extrair o ZIP da URL {url}: {e}")
        return []


# # ETL

# In[3]:


# Função para carregar dados de um arquivo CSV
def carregar_dados_csv(caminho_arquivo):
    """
    Carrega um arquivo CSV em um DataFrame, padronizando os nomes das colunas.
    Args:
    caminho_arquivo (str): Caminho do arquivo CSV.
    Returns:
    pd.DataFrame: DataFrame com os dados carregados.
    """
    try:
        df = pd.read_csv(caminho_arquivo, sep=';', encoding="latin-1")
        df.columns = [col.strip().lower() for col in df.columns]  # Padronizando os nomes das colunas
        return df
    except Exception as e:
        print(f"Erro ao carregar dados do arquivo CSV {caminho_arquivo}: {e}")
        return pd.DataFrame()  # Retornar um DataFrame vazio em caso de erro


# In[4]:


# URLs dos arquivos ZIP
url_2024 = 'https://portaldatransparencia.gov.br/download-de-dados/viagens/2024'
url_2023 = 'https://portaldatransparencia.gov.br/download-de-dados/viagens/2023'
url_2022 = 'https://portaldatransparencia.gov.br/download-de-dados/viagens/2022'


# In[5]:


# Diretório onde os arquivos serão extraídos
pasta_destino = r'C:\Users\erick\OneDrive\Área de Trabalho\Jupyter Notebook\Case - Bem Promotora'


# In[6]:


# Baixar e extrair arquivos ZIP para cada ano
arquivos_2024 = baixar_e_extrair_zip(url_2024, pasta_destino)
arquivos_2023 = baixar_e_extrair_zip(url_2023, pasta_destino)
arquivos_2022 = baixar_e_extrair_zip(url_2022, pasta_destino)


# In[7]:


# Função para carregar e padronizar dados
def carregar_dados(nome_arquivo):
    """
    Carrega um arquivo CSV em um DataFrame, padronizando os nomes das colunas.
    Args:
    nome_arquivo (str): Nome do arquivo CSV.
    Returns:
    pd.DataFrame: DataFrame com os dados carregados.
    """
    if not os.path.isfile(nome_arquivo):
        raise FileNotFoundError(f"O arquivo {nome_arquivo} não foi encontrado.")
    df = pd.read_csv(nome_arquivo, sep=';', encoding="latin-1")
    df.columns = [col.strip().lower() for col in df.columns]  # Padronizando os nomes das colunas
    return df


# In[8]:


# Função para carregar e concatenar dados de diferentes anos
def carregar_e_concatenar(anos, base):
    """
    Carrega e concatena dados de diferentes anos em um único DataFrame.
    Args:
    anos (list): Lista de anos para carregar dados.
    base (str): Nome base dos arquivos CSV.
    Returns:
    pd.DataFrame: DataFrame concatenado com dados de todos os anos.
    """
    dfs = []
    for ano in anos:
        nome_arquivo = f'{ano}_{base}.csv'
        df = carregar_dados(nome_arquivo)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# In[9]:


# Anos a serem carregados
anos = [2022, 2023, 2024]

# Carregando e concatenando as bases de dados
df_pagamento_unificado = carregar_e_concatenar(anos, 'Pagamento')
df_trecho_unificado = carregar_e_concatenar(anos, 'Trecho')
df_passagem_unificado = carregar_e_concatenar(anos, 'Passagem')
df_viagem_unificado = carregar_e_concatenar(anos, 'Viagem')


# In[10]:


# Função para tratar dados unificados
def tratar_dados_unificados(df, nome):
    """
    Remove duplicatas e substitui valores nulos por 0 em um DataFrame.
    Args:
    df (pd.DataFrame): DataFrame a ser tratado.
    nome (str): Nome do DataFrame para mensagens de log.
    """
    df.drop_duplicates(inplace=True)
    if df.isnull().values.any():
        print(f"Valores nulos encontrados em {nome}, substituindo por 0.")
        df.fillna(0, inplace=True)


# In[11]:


# Tratamento de todas as bases
tratar_dados_unificados(df_pagamento_unificado, "Pagamento")
tratar_dados_unificados(df_trecho_unificado, "Trecho")
tratar_dados_unificados(df_passagem_unificado, "Passagem")
tratar_dados_unificados(df_viagem_unificado, "Viagem")


# In[12]:


# Verificar os tipos de Dados de todas as bases
for nome, df in [('Pagamento', df_pagamento_unificado), 
                 ('Trecho', df_trecho_unificado), 
                 ('Passagem', df_passagem_unificado), 
                 ('Viagem', df_viagem_unificado)]:
    print(f"Tipos de dados em {nome}:\n", df.dtypes, "\n")


# # Conectando ao SQL Server

# In[13]:


# Crie a URL de conexão para o SQL Server
conn_str = (
    'mssql+pyodbc://'
    'ERICK_DIAS/CASE_BEM_PROMOVIDA?'
    'driver=ODBC+Driver+17+for+SQL+Server'
)


# In[14]:


# Crie o engine com a URL de conexão
engine = create_engine(conn_str)


# In[15]:


# Carregar as tabelas no banco de dados com chunksize para evitar problemas de memória
for nome, df in [('viagem', df_viagem_unificado), 
                 ('passagem', df_passagem_unificado), 
                 ('trecho', df_trecho_unificado), 
                 ('pagamento', df_pagamento_unificado)]:
    try:
        df.to_sql(nome, con=engine, if_exists='replace', index=False, chunksize=1000)
    except Exception as e:
        print(f"Erro ao carregar a tabela {nome}: {e}")


# # Cluster

# In[16]:


# Função para garantir que a coluna é tratada como string e substituir vírgulas por pontos
def convert_to_numeric(df, column_name):
    """
    Converte uma coluna para tipo numérico, substituindo vírgulas por pontos.
    Args:
    df (pd.DataFrame): DataFrame contendo a coluna.
    column_name (str): Nome da coluna a ser convertida.
    """
    if df[column_name].dtype == 'object':
        df[column_name] = pd.to_numeric(df[column_name].str.replace(',', '.'), errors='coerce')
    else:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')


# In[17]:


# Conversão de colunas de valor para numérico
convert_to_numeric(df_pagamento_unificado, 'valor')
convert_to_numeric(df_passagem_unificado, 'valor da passagem')
convert_to_numeric(df_passagem_unificado, 'taxa de serviço')
convert_to_numeric(df_viagem_unificado, 'valor diárias')
convert_to_numeric(df_viagem_unificado, 'valor passagens')
convert_to_numeric(df_viagem_unificado, 'valor devolução')
convert_to_numeric(df_viagem_unificado, 'valor outros gastos')


# In[18]:


# Função para converter colunas de data para datetime
def convert_to_datetime(df, column_name):
    """
    Converte uma coluna para tipo datetime.
    Args:
    df (pd.DataFrame): DataFrame contendo a coluna.
    column_name (str): Nome da coluna a ser convertida.
    """
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce', dayfirst=True)


# In[19]:


# Conversão de colunas de data para datetime
convert_to_datetime(df_trecho_unificado, 'origem - data')
convert_to_datetime(df_trecho_unificado, 'destino - data')
convert_to_datetime(df_passagem_unificado, 'data da emissão/compra')
convert_to_datetime(df_viagem_unificado, 'período - data de início')
convert_to_datetime(df_viagem_unificado, 'período - data de fim')


# In[20]:


# Remover duplicatas específicas
df_pagamento_unificado = df_pagamento_unificado.drop_duplicates(subset='identificador do processo de viagem')


# In[21]:


# Merge com df_viagem_unificado
df_combinado = df_viagem_unificado.merge(
    df_pagamento_unificado[['identificador do processo de viagem', 'valor']],
    on='identificador do processo de viagem',
    how='left'
)


# In[22]:


# Escolha as colunas relevantes para clusterização
colunas = ['valor', 'valor diárias', 'valor passagens', 'valor devolução', 'valor outros gastos']
df_cluster = df_combinado[colunas].copy()


# In[23]:


# Remover linhas com NaN para o modelo
df_cluster.dropna(inplace=True)


# In[24]:


# Normalização dos dados
scaler = StandardScaler()
df_cluster_scaled = scaler.fit_transform(df_cluster)


# In[25]:


# Aplicação do K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_cluster_scaled)


# In[26]:


# Certifique-se de que df_combinado e clusters têm o mesmo tamanho
df_combinado = df_combinado.loc[df_cluster.index].copy()  # Manter apenas as linhas utilizadas
df_combinado['cluster'] = clusters


# In[27]:


# Visualização dos clusters
plt.scatter(df_combinado['valor diárias'], df_combinado['valor passagens'], c=df_combinado['cluster'])
plt.xlabel('Valor Diárias')
plt.ylabel('Valor Passagens')
plt.title('Clusterização de Viagens')
plt.colorbar(label='Cluster')
plt.show()


# # Análise com ARIMA

# In[28]:


# Suponha que estamos interessados na coluna 'valor diárias' como série temporal
df_viagem_unificado['data'] = pd.to_datetime(df_viagem_unificado['período - data de início'])
df_viagem_unificado.set_index('data', inplace=True)
df_viagem_unificado.sort_index(inplace=True)


# In[29]:


# Verifique se a série é estacionária
sample_df = df_viagem_unificado['valor diárias'].dropna().sample(n=10000)  # Ajuste o tamanho conforme necessário
result = adfuller(sample_df)
print('p-valor do teste ADF:', result[1])


# # Ajuste o modelo ARIMA

# In[30]:


# Certifique-se de que não há espaços extras ou caracteres especiais nos nomes das colunas
df_viagem_unificado.columns = df_viagem_unificado.columns.str.strip()


# In[31]:


# Verifique os nomes das colunas para confirmar
print("Nomes das colunas:", df_viagem_unificado.columns)


# In[32]:


# Ajuste o nome da coluna conforme necessário
data_col = 'período - data de fim'  # Nome exato da coluna


# In[33]:


# Convertendo a coluna 'período - data de início' para o tipo datetime e definindo-a como índice do DataFrame
df_viagem_unificado['data'] = pd.to_datetime(df_viagem_unificado['período - data de início'])
df_viagem_unificado.set_index('data', inplace=True)
df_viagem_unificado.sort_index(inplace=True)  # Ordena os dados por data

# Remover índices duplicados para garantir a integridade dos dados
df_viagem_unificado = df_viagem_unificado[~df_viagem_unificado.index.duplicated(keep='first')]

# Verifica e define a frequência do índice se necessário
if df_viagem_unificado.index.freq is None:
    # Define a frequência como diária, ajuste conforme a granularidade dos seus dados
    df_viagem_unificado = df_viagem_unificado.asfreq('D', fill_value=0)  # Preenche valores ausentes com 0

# Seleciona a coluna 'valor diárias' e ajusta o modelo ARIMA
# A ordem (5, 1, 0) é um exemplo e pode precisar ser ajustada com base na análise dos dados
model = ARIMA(df_viagem_unificado['valor diárias'].dropna(), order=(5, 1, 0))
model_fit = model.fit()

# Exibe um resumo do modelo ajustado, que inclui métricas e parâmetros do ARIMA
print(model_fit.summary())

# Faz previsões para os próximos 12 períodos
# O número de períodos a prever pode ser ajustado conforme necessário
forecast = model_fit.forecast(steps=12)

# Cria um índice de datas para as previsões, começando um dia após o último índice disponível
forecast_index = pd.date_range(start=df_viagem_unificado.index[-1] + pd.DateOffset(1), periods=12, freq='D')

# Cria uma série temporal com as previsões e o índice correspondente
forecast_series = pd.Series(forecast, index=forecast_index)

# Visualiza os resultados
plt.figure(figsize=(10, 6))  # Define o tamanho da figura
plt.plot(df_viagem_unificado.index, df_viagem_unificado['valor diárias'], label='Dados Históricos')  # Plota os dados históricos
plt.plot(forecast_series.index, forecast_series, color='red', label='Previsão')  # Plota as previsões em vermelho
plt.xlabel('Data')  # Rotula o eixo X
plt.ylabel('Valor Diárias')  # Rotula o eixo Y
plt.title('Previsão com ARIMA')  # Define o título do gráfico
plt.legend()  # Adiciona a legenda
plt.show()  # Exibe o gráfico


# In[ ]:




