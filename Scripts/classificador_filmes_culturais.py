# ==========================================================
# CLASSIFICADOR CULTURAL E MOTOR DE RECOMENDAÇÃO OFFLINE v3
# Alta Escala (1500 Registros) + Sistema de Checkpoint (Backup)
# ==========================================================

import pandas as pd
import ast
import sys
import warnings
from tqdm import tqdm
from transformers import pipeline

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

CAMINHO_ORIGINAL = r"C:\Users\paulo\Desktop\base dados projeto 3\movies_metadata.csv"
CAMINHO_RESULTADO = r"C:\Users\paulo\Desktop\base_recomendacao_avancada.xlsx"
CAMINHO_BACKUP = r"C:\Users\paulo\Desktop\base_recomendacao_BACKUP.xlsx"

# ==========================================================
# 🧠 1. INICIALIZANDO A IA
# ==========================================================
print("⏳ Carregando o cérebro da IA (BART-Large)...")
classificador_ia = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

# ==========================================================
# 🛠️ 2. PREPARANDO BASE (1.500 FILMES)
# ==========================================================
def preparar_dados(path, limite=1500): # <--- LIMITE AJUSTADO PARA 1500
    print(f"🧹 Preparando os {limite} filmes mais populares...")
    df = pd.read_csv(path, low_memory=False)
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
    df = df.sort_values(by='vote_count', ascending=False).head(limite).copy()
    df['Ano'] = df['release_date'].astype(str).str[:4]
    df['overview'] = df['overview'].fillna("No plot available.")
    
    def extrair_generos(x):
        try:
            if pd.isna(x) or x == "": return ""
            return ", ".join([i.get("name", "") for i in ast.literal_eval(x) if "name" in i])
        except: return ""
            
    df['Generos'] = df['genres'].apply(extrair_generos)
    return df[['id', 'title', 'Ano', 'Generos', 'overview', 'vote_count', 'vote_average']]

# ==========================================================
# 🎯 3. DICIONÁRIOS DE TRADUÇÃO E CATEGORIAS
# ==========================================================
TRADUCAO_CULTURAL = {
    "historical event or biography": "Fatos Históricos ou Biografia",
    "social issues and inequality": "Questões Sociais e Desigualdade",
    "political movement or revolution": "Movimentos Políticos ou Revolução",
    "cultural and ethnic identity": "Identidade Cultural e Étnica",
    "philosophical or existential journey": "Jornada Filosófica/Existencial",
    "generic blockbuster action": "Ação Comercial Genérica",
    "pure entertainment and escapism": "Puro Entretenimento/Escapismo",
    "standard romantic comedy": "Comédia Romântica Padrão"
}

LABELS_CULTURAIS = list(TRADUCAO_CULTURAL.keys())[:5]
LABELS_COMERCIAIS = list(TRADUCAO_CULTURAL.keys())[5:]
TODOS_LABELS_NATUREZA = LABELS_CULTURAIS + LABELS_COMERCIAIS

TEMAS_RECOMENDACAO = [
    "revenge and justice", "artificial intelligence", "organized crime", 
    "survival and nature", "war and conflict", "love and romance", 
    "family dynamics", "space exploration", "superheroes", "systemic corruption",
    "coming of age", "horror and paranormal"
]

CLIMAS_RECOMENDACAO = [
    "dark and gritty", "uplifting and inspiring", 
    "tense and suspenseful", "melancholic and dramatic",
    "fun and energetic", "mind-bending and complex"
]

# ==========================================================
# 🕵️‍♂️ 4. O MOTOR ANALÍTICO
# ==========================================================
def analisar_filme(titulo, generos, sinopse):
    texto = f"Title: {titulo}. Genres: {generos}. Plot: {sinopse}"
    
    try:
        res_natureza = classificador_ia(texto, TODOS_LABELS_NATUREZA)
        vencedor_natureza = res_natureza['labels'][0]
        confianca_natureza = res_natureza['scores'][0]
        
        res_temas = classificador_ia(texto, TEMAS_RECOMENDACAO)
        tema_1, tema_2 = res_temas['labels'][0], res_temas['labels'][1]
        
        res_clima = classificador_ia(texto, CLIMAS_RECOMENDACAO)
        clima = res_clima['labels'][0]
        
        if vencedor_natureza in LABELS_CULTURAIS:
            is_cultural = 1
            justificativa = f"Relevância Cultural: O roteiro foca fortemente em '{TRADUCAO_CULTURAL[vencedor_natureza]}'."
        else:
            is_cultural = 0
            justificativa = f"Foco Comercial: Classificado primariamente como '{TRADUCAO_CULTURAL[vencedor_natureza]}'."
            
        return is_cultural, round(confianca_natureza * 100, 1), justificativa, f"{tema_1} | {tema_2}", clima
        
    except Exception as e:
        return 0, 0, "Erro na leitura", "Erro", "Erro"

# ==========================================================
# 🚀 5. EXECUÇÃO COM SISTEMA DE BACKUP
# ==========================================================
if __name__ == "__main__":
    
    QTD_FILMES = 1500
    df = preparar_dados(CAMINHO_ORIGINAL, limite=QTD_FILMES)
    
    df['É_Cultural'] = 0
    df['Certeza_IA_(%)'] = 0.0
    df['Justificativa_Validacao'] = ""
    df['Temas_Recomendacao'] = ""
    df['Atmosfera_Filme'] = ""
    
    print(f"\n🤖 Iniciando processamento de {QTD_FILMES} filmes...")
    print("⚠️ DICA: Desative a 'Suspensão de Tela/Dormir' do seu Windows para o PC não desligar no meio.")
    
    for index_interno, (index_df, row) in tqdm(enumerate(df.iterrows()), total=df.shape[0]):
        cult, conf, just, temas, clima = analisar_filme(row['title'], row['Generos'], row['overview'])
        
        df.at[index_df, 'É_Cultural'] = cult
        df.at[index_df, 'Certeza_IA_(%)'] = conf
        df.at[index_df, 'Justificativa_Validacao'] = just
        df.at[index_df, 'Temas_Recomendacao'] = temas
        df.at[index_df, 'Atmosfera_Filme'] = clima
        
        # SISTEMA DE BACKUP: Salva uma cópia a cada 250 filmes processados
        if (index_interno + 1) % 250 == 0:
            df_temp = df.drop(columns=['overview'])
            df_temp.to_excel(CAMINHO_BACKUP, index=False)
            print(f"\n💾 Backup automático salvo ({index_interno + 1}/{QTD_FILMES} filmes).")

    # Finalização
    df = df.drop(columns=['overview'])
    df.to_excel(CAMINHO_RESULTADO, index=False)
    print(f"\n✅ SUCESSO TOTAL! Base final com {QTD_FILMES} filmes salva em: {CAMINHO_RESULTADO}")