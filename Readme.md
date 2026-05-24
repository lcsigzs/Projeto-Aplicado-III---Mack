# 🎬 Projeto Aplicado III — Análise e Recomendação de Filmes Culturais

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Universidade Presbiteriana Mackenzie**  
*Curso de Ciência de Dados — 2026/2*

</div>

---

## 📌 Sobre o Projeto

Este projeto foi desenvolvido como parte da disciplina **Projeto Aplicado III** do curso de Ciência de Dados da **Universidade Presbiteriana Mackenzie**. O objetivo é aplicar técnicas de **análise exploratória de dados (EDA)**, **classificação** e **sistemas de recomendação** sobre uma base de filmes, com foco em **conteúdo cultural**.

A solução combina:
- 📊 Análises estatísticas e visualizações sobre o catálogo de filmes;
- 🤖 Um classificador para identificar filmes com relevância cultural;
- 🎯 Um sistema de **recomendação avançada** baseado em similaridade;
- 🖥️ Uma aplicação web interativa construída em **Streamlit**.

---

## 🎯 Objetivos

- **Geral:** Construir um sistema capaz de analisar e recomendar filmes culturais com base em atributos relevantes do catálogo.
- **Específicos:**
  - Realizar análise exploratória da base de dados;
  - Tratar e enriquecer os dados (feature engineering);
  - Implementar um classificador de filmes culturais;
  - Desenvolver um motor de recomendação;
  - Disponibilizar a solução em uma interface web acessível.

---

## 👥 Integrantes

| Nome | RA |
|---|---|
| Lucas Iglezias dos Anjos | 10433522 |
| Thaís Cristine de Andrade Gomes | 10721642 |
| Paulo Ricardo de Oliveira Ramos | 10721464 | 

**Professora orientadora: CAROLINA TOLEDO FERRAZ**

---

## 🛠️ Tecnologias Utilizadas

| Categoria | Ferramentas |
| **Linguagem** | Python 3.10+ |
| **Análise de Dados** | Pandas, NumPy |
| **Visualização** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Aplicação Web** | Streamlit |
| **Notebooks** | Jupyter |
| **Manipulação de Arquivos** | OpenPyXL |
| **Controle de Versão** | Git & GitHub |

---

## 📂 Estrutura do Repositório

```
projeto-aplicado-iii-mack/
│
├── data/                          # Bases de dados
│   ├── raw/                       # Dados originais
│   └── processed/                 # Dados tratados
│       └── base_recomendacao_avancada.xlsx
│
├── notebooks/                     # Análises em Jupyter
│   ├── 01_exploracao_dados.ipynb
│   └── 02_graficos_analise.ipynb
│
├── src/                           # Código-fonte
│   ├── app.py                     
│   └── classificador_filmes_culturais.py
│
├── docs/                          # Documentação e relatórios
│   └── imagens/
│
├── requirements.txt               # Dependências do projeto
├── .gitignore
└── README.md
```

---

## 🚀 Como Executar o Projeto

### 📋 Pré-requisitos

- Python **3.10** ou superior
- `pip` 
- Git

### 1️⃣ Clonar o repositório

```bash
git clone https://github.com/lcsigzs/Projeto-Aplicado-III---Mack.git
cd Projeto-Aplicado-III---Mack
```

### 2️⃣ Criar e ativar um ambiente virtual (recomendado)

**Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4️⃣ Executar a aplicação Streamlit

```bash
streamlit run src/app.py
```

A aplicação abrirá automaticamente em `http://localhost:8501`.

### 5️⃣ Executar os notebooks de análise

```bash
jupyter notebook
```

E abra os arquivos dentro da pasta `notebooks/`.

---

## 📚 Referências

- Documentação Pandas — https://pandas.pydata.org/docs/
- Documentação Streamlit — https://docs.streamlit.io/
- Scikit-learn User Guide — https://scikit-learn.org/stable/user_guide.html

---

## 📄 Licença

Este projeto é de cunho **acadêmico**, desenvolvido para fins educacionais na disciplina de Projeto Aplicado III — Mackenzie.

---

<div align="center">

⭐ Se este projeto foi útil, deixe uma estrela no repositório!

</div>
