# Projeto de exemplo de pipeline no Azure Machine Learning
Projeto de exemplo que mostra como criar um pipeline no Azure Machine Learning pensando em reprodutibilidade de resultados em treino de modelos de machine learning.
Neste exemplo, é apresentado como treinar um modelo para classificar flores usando o Iris Dataset.
Ao final, teremos um pipeline automático que permitirá retreino com grande facilidade de troca de dados ou parâmetros.
Além disso, todos os hiperparâmetros, dataset, modelo e artefatos são registrados durante o experimento.

Como requerimentos para usar o notebook principal do projeto:
- obrigatório: ter conta no Portal do Azure
    https://azure.microsoft.com/pt-br/features/azure-portal/    
- obrigatório: ter um workspace AzureMl
- obrigatório: adicionar o arquivo de config.json do workspace em (azureml_files -> configuration)
- obrigatório: ter compute target


- opcional: ter datastore criado (pode ser usado o default)
- opcional: ter um yaml de um ambiente virtual (conda env export --name azureml > environment.yml) ou arquivo de requirements.txt

------------------------------

### Passos dentro do notebook:
- Importando pacotes
- Definindo variáveis
- Acessando o workspace
------------------------------
- Criando um datastore
- Criando um ambiente
- Fazendo o upload de dataset no datastore
- Registrando o dataset
- Registrando o ambiente
------------------------------
- Definindo os steps do pipeline
- Criando o pipeline
- Criando o experimento com o pipeline

------------------------------

### Vídeo explicativo: [em andamento]
