import os
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from jinja2 import Template

app = FastAPI(title="Projeto IA Análise de Filmes - Busca Assistida")

EXCEL_PATH = "base_recomendacao_avancada.xlsx"

# Dicionário de tradução para os gêneros
TRADUCAO_GENEROS = {
    "action": "Ação", "adventure": "Aventura", "animation": "Animação",
    "comedy": "Comédia", "crime": "Crime", "documentary": "Documentário",
    "drama": "Drama", "family": "Família", "fantasy": "Fantasia",
    "history": "História", "horror": "Terror", "music": "Música",
    "mystery": "Mistério", "romance": "Romance", "science fiction": "Ficção Científica",
    "sci-fi": "Ficção Científica", "thriller": "Suspense", "war": "Guerra", "western": "Faroeste"
}

def traduzir_lista_generos(string_generos):
    if not string_generos or pd.isna(string_generos):
        return "Geral"
    partes = [g.strip() for g in str(string_generos).split(",")]
    traduzidos = [TRADUCAO_GENEROS.get(p.lower(), p) for p in partes]
    return ", ".join(traduzidos)

def carregar_e_traduzir_dados():
    if os.path.exists(EXCEL_PATH):
        try:
            df = pd.read_excel(EXCEL_PATH)
            df['title'] = df['title'].fillna("Título Desconhecido").astype(str)
            df['Generos'] = df['Generos'].fillna("Geral").astype(str)
            df['Justificativa_Validacao'] = df['Justificativa_Validacao'].fillna("Análise padrão.").astype(str)
            df['Temas_Recomendacao'] = df['Temas_Recomendacao'].fillna("Geral").astype(str)
            df['Atmosfera_Filme'] = df['Atmosfera_Filme'].fillna("Neutra").astype(str)
            
            # --- MOTOR DE TRADUÇÃO DEFINITIVA (Títulos, Temas e Atmosfera) ---
            precisa_salvar = False
            
            if 'Titulo_PT' not in df.columns or 'Temas_PT' not in df.columns or 'Atmosfera_PT' not in df.columns:
                print("\n" + "="*60)
                print("⏳ TRADUZINDO DADOS RESTANTES PARA O PORTUGUÊS (PT-BR)")
                print("Aguarde, a guardar resultados diretamente no Excel...")
                print("="*60 + "\n")
                
                try:
                    from deep_translator import GoogleTranslator
                    tradutor = GoogleTranslator(source='en', target='pt')
                    
                    def traduzir_coluna(valores, nome_coluna):
                        print(f"Traduzindo {nome_coluna}...")
                        resultados = []
                        for v in valores:
                            try:
                                if str(v).strip() == "" or str(v).lower() == "nan":
                                    resultados.append("Não definido")
                                else:
                                    resultados.append(tradutor.translate(str(v)))
                            except:
                                resultados.append(str(v)) # Fallback em caso de erro na linha
                        return resultados

                    if 'Titulo_PT' not in df.columns:
                        df['Titulo_PT'] = traduzir_coluna(df['title'], "Títulos")
                        precisa_salvar = True
                        
                    if 'Temas_PT' not in df.columns:
                        df['Temas_PT'] = traduzir_coluna(df['Temas_Recomendacao'], "Temas da IA")
                        precisa_salvar = True
                        
                    if 'Atmosfera_PT' not in df.columns:
                        df['Atmosfera_PT'] = traduzir_coluna(df['Atmosfera_Filme'], "Atmosfera dos Filmes")
                        precisa_salvar = True

                    if precisa_salvar:
                        df.to_excel(EXCEL_PATH, index=False)
                        print("✅ Toda a base foi traduzida e guardada com sucesso!")
                        
                except ImportError:
                    print("⚠️ A biblioteca 'deep-translator' não está instalada.")
                    print("Para traduzir, cancele e rode: pip install deep-translator")
                    if 'Titulo_PT' not in df.columns: df['Titulo_PT'] = df['title']
                    if 'Temas_PT' not in df.columns: df['Temas_PT'] = df['Temas_Recomendacao']
                    if 'Atmosfera_PT' not in df.columns: df['Atmosfera_PT'] = df['Atmosfera_Filme']
            
            return df
        except Exception as e:
            print(f"Erro ao ler Excel: {e}")
            
    # Fallback caso dê erro
    df_fallback = pd.DataFrame({
        'title': ['The Dark Knight'], 'Titulo_PT': ['O Cavaleiro das Trevas'], 
        'Generos': ['Action'], 'vote_average': [8.3], 'Temas_PT': ['Guerra e Conflito'], 
        'Atmosfera_PT': ['Tensa'], 'É_Cultural': [1], 'Certeza_IA_(%)': [85]
    })
    return df_fallback

df_filmes = carregar_e_traduzir_dados()
lista_generos_pt = sorted(list(set(TRADUCAO_GENEROS.values())))

def executar_recomendacao(termo_busca, peso_knn, peso_ia):
    candidatos = []
    termo_busca = str(termo_busca).strip().lower()
    
    if not termo_busca:
        return []
        
    filme_alvo_df = df_filmes[df_filmes['Titulo_PT'].str.lower().str.contains(termo_busca, na=False)]
    
    if filme_alvo_df.empty: 
        filme_alvo_df = df_filmes[df_filmes['title'].str.lower().str.contains(termo_busca, na=False)]
        
    if filme_alvo_df.empty:
        return []
        
    alvo = filme_alvo_df.iloc[0]

    for _, row in df_filmes.iterrows():
        if str(row['title']).lower() == str(alvo['title']).lower():
            continue
            
        generos_traduzidos_row = traduzir_lista_generos(row['Generos'])
        
        # --- MATEMÁTICA DE SIMILARIDADE CORRIGIDA (ÍNDICE DE JACCARD) ---
        set_alvo = set([g.strip().lower() for g in str(alvo['Generos']).split(',')])
        set_row = set([g.strip().lower() for g in str(row['Generos']).split(',')])
        
        intersecao = len(set_alvo & set_row)
        uniao = len(set_alvo | set_row)
        
        # Barreira: Se não compartilha NENHUM gênero, descarta na hora (evita resultados exóticos)
        if intersecao == 0:
            continue
            
        sim_generos = intersecao / uniao if uniao > 0 else 0
        
        tema_alvo = str(alvo.get('Temas_PT', '')).strip().lower()
        tema_row = str(row.get('Temas_PT', '')).strip().lower()
        bonus_tema = 0.2 if (tema_alvo != 'geral' and tema_alvo == tema_row) else 0
        
        score_base_conteudo = min(sim_generos + bonus_tema, 1.0)
        
        # Barreira 2: Corte de similaridade muito fraca
        if score_base_conteudo < 0.2:
            continue
            
        confianca_ia = float(row.get('Certeza_IA_(%)', 0)) / 100.0
        multiplicador_cultural = 1.25 if int(row.get('É_Cultural', 0)) == 1 else 0.9
        
        # O Score da IA atua como multiplicador e modulador da similaridade
        score_combinado = (score_base_conteudo * peso_knn) + (score_base_conteudo * confianca_ia * multiplicador_cultural * peso_ia)
        
        peso_total = peso_knn + peso_ia
        if peso_total > 0:
            match_final = min(round((score_combinado / peso_total) * 100), 99)
        else:
            match_final = 0
            
        candidatos.append({
            "titulo": str(row.get('Titulo_PT', row['title'])),
            "titulo_original": str(row['title']),
            "ano": int(row.get('Ano', 2000)),
            "generos": generos_traduzidos_row,
            "nota": float(row.get('vote_average', 0)),
            "cultural": int(row.get('É_Cultural', 0)),
            "confianca": round(confianca_ia * 100),
            "justificativa": str(row.get('Justificativa_Validacao', '')),
            "temas": str(row.get('Temas_PT', 'Geral')), 
            "atmosfera": str(row.get('Atmosfera_PT', 'Neutra')), 
            "score": match_final
        })
        
    candidatos.sort(key=lambda x: x['score'], reverse=True)
    return candidatos[:12]

# --- INTERFACE VISUAL (HTML/CSS) ---
CODIGO_INTERFACE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Projeto IA Análise de Filmes</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --cinza-fundo: #0d0f12; --cinza-modulo: #161a22; --cinza-borda: #262f3d;
            --vermelho-marca: #990000; --vermelho-neon: #ff3333; --verde-neon: #39ff14;
            --texto-claro: #ffffff; --texto-palido: #8fa0b5;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', Roboto, sans-serif; }
        body { background-color: var(--cinza-fundo); color: var(--texto-claro); padding: 30px 4%; }

        header { border-bottom: 1px solid var(--cinza-borda); padding-bottom: 20px; margin-bottom: 30px; }
        .logo-container h1 { font-size: 22px; font-weight: 700; }
        .logo-container span { color: var(--vermelho-neon); }

        .grid-painel { display: grid; grid-template-columns: 1.2fr 2fr; gap: 30px; margin-bottom: 40px; }

        .bloco-assistente { background: var(--cinza-modulo); border: 1px solid var(--cinza-borda); padding: 25px; border-radius: 6px; }
        .bloco-assistente h2 { font-size: 14px; color: var(--texto-palido); text-transform: uppercase; margin-bottom: 15px; display: flex; align-items: center; gap: 8px; }
        
        select, input[type="text"] {
            width: 100%; padding: 12px 15px; background-color: var(--cinza-fundo);
            border: 1px solid var(--cinza-borda); border-radius: 4px; color: #fff; font-size: 14px; outline: none; margin-bottom: 15px;
        }
        input[type="text"]:focus, select:focus { border-color: var(--vermelho-neon); }

        .sugestoes-container { display: flex; flex-wrap: wrap; gap: 8px; }
        .sugestao-pill {
            background: rgba(153, 0, 0, 0.1); border: 1px solid rgba(153,0,0,0.4);
            color: #d1d9e6; padding: 6px 14px; border-radius: 20px; font-size: 12px; cursor: pointer; transition: 0.2s;
        }
        .sugestao-pill:hover { background: var(--vermelho-marca); color: #fff; border-color: var(--vermelho-neon); transform: translateY(-2px); }

        .bloco-motor { background: var(--cinza-modulo); border: 1px solid var(--cinza-borda); padding: 25px; border-radius: 6px; border-top: 3px solid var(--vermelho-marca); }
        .pesos-container { display: flex; gap: 30px; margin-top: 15px; background: var(--cinza-fundo); padding: 15px; border-radius: 4px; }
        .controle-peso { flex: 1; }
        .controle-peso label { display: block; font-size: 11px; color: var(--texto-palido); margin-bottom: 8px; text-transform: uppercase; }
        input[type="range"] { width: 100%; accent-color: var(--vermelho-marca); cursor: pointer; }

        .secao-titulo { font-size: 16px; font-weight: 600; margin-bottom: 20px; color: var(--texto-palido); }
        .modulo-filme {
            background-color: var(--cinza-modulo); border: 1px solid var(--cinza-borda); border-radius: 4px;
            padding: 20px; display: grid; grid-template-columns: 100px 1fr 280px; gap: 25px; align-items: center; margin-bottom: 15px; transition: 0.2s;
        }
        .modulo-filme:hover { border-color: var(--vermelho-marca); transform: translateX(4px); }

        .pct-match { color: var(--verde-neon); font-size: 22px; font-weight: 700; }
        .lbl-match { font-size: 10px; color: var(--texto-palido); text-transform: uppercase; }
        .info-principal h3 { font-size: 18px; margin-bottom: 6px; text-transform: capitalize; }
        .info-principal h4 { font-size: 11px; color: var(--texto-palido); margin-bottom: 8px; font-style: italic; }
        
        .badges-container { display: flex; gap: 8px; margin-top: 10px; }
        .badge { font-size: 10px; padding: 3px 8px; border-radius: 2px; text-transform: uppercase; font-weight: bold; }
        .badge-cult { background-color: rgba(255, 51, 51, 0.1); color: var(--vermelho-neon); border: 1px solid rgba(255, 51, 51, 0.2); }
        .badge-com { background-color: rgba(143, 160, 181, 0.1); color: var(--texto-palido); }
        .badge-ia { background-color: rgba(57, 255, 20, 0.05); color: var(--verde-neon); border: 1px solid rgba(57, 255, 20, 0.15); }

        .info-analitica { background-color: rgba(13, 15, 18, 0.5); padding: 12px; border-left: 3px solid var(--vermelho-marca); font-size: 11px; color: #d1d9e6; }
    </style>
</head>
<body>

    <header>
        <div class="logo-container">
            <h1>PROJETO IA <span>ANÁLISE DE FILMES</span></h1>
            <p style="font-size: 11px; color: var(--texto-palido); margin-top: 5px;">SISTEMA HÍBRIDO DE RECOMENDAÇÃO ACADÊMICA</p>
        </div>
    </header>

    <div class="grid-painel">
        <div class="bloco-assistente">
            <h2><i class="fa-solid fa-lightbulb"></i> Inspiração por Categoria</h2>
            <select id="filtroCategoria" onchange="carregarSugestoes()">
                <option value="">Selecione para ver exemplos...</option>
                {% for g in generos %}
                <option value="{{ g }}">{{ g }}</option>
                {% endfor %}
            </select>
            
            <p style="font-size: 11px; color: var(--texto-palido); margin-bottom: 10px;" id="txtSugestao">Exemplos populares (Clique para pesquisar):</p>
            <div class="sugestoes-container" id="areaSugestoes">
                </div>
        </div>

        <div class="bloco-motor">
            <h2 style="font-size: 14px; color: var(--texto-palido); text-transform: uppercase; margin-bottom: 15px;"><i class="fa-solid fa-microchip"></i> Motor de Recomendação</h2>
            <input type="text" id="campoBusca" list="listaTitulos" placeholder="Digite o nome de um filme (ex: A Origem)..." autocomplete="off">
            <datalist id="listaTitulos">
                {% for t in titulos %}
                <option value="{{ t }}"></option>
                {% endfor %}
            </datalist>

            <div class="pesos-container">
                <div class="controle-peso">
                    <label>Peso Matemático (KNN): <span id="txtKnn">0.6</span></label>
                    <input type="range" id="pesoKnn" min="0" max="1" step="0.1" value="0.6">
                </div>
                <div class="controle-peso">
                    <label>Peso de Conteúdo (IA BART): <span id="txtIa">0.4</span></label>
                    <input type="range" id="pesoIa" min="0" max="1" step="0.1" value="0.4">
                </div>
            </div>
        </div>
    </div>

    <div class="secao-titulo" id="tituloResultados">Aguardando filme de referência...</div>
    <div id="containerResultados"></div>

    <script>
        const campoBusca = document.getElementById('campoBusca');
        const pesoKnn = document.getElementById('pesoKnn');
        const pesoIa = document.getElementById('pesoIa');
        let timeout = null;

        async function carregarSugestoes() {
            const categoria = document.getElementById('filtroCategoria').value;
            if(!categoria) return;
            
            const response = await fetch(`/api/sugestoes?categoria=${encodeURIComponent(categoria)}`);
            const nomes = await response.json();
            
            const area = document.getElementById('areaSugestoes');
            area.innerHTML = '';
            
            nomes.forEach(nome => {
                const pill = document.createElement('div');
                pill.className = 'sugestao-pill';
                pill.textContent = nome;
                pill.onclick = () => {
                    campoBusca.value = nome;
                    sincronizarEBuscar();
                };
                area.appendChild(pill);
            });
        }

        function sincronizarEBuscar() {
            document.getElementById('txtKnn').textContent = pesoKnn.value;
            document.getElementById('txtIa').textContent = pesoIa.value;

            clearTimeout(timeout);
            timeout = setTimeout(requisitarDadosRecs, 400);
        }

        campoBusca.addEventListener('input', sincronizarEBuscar);
        pesoKnn.addEventListener('input', sincronizarEBuscar);
        pesoIa.addEventListener('input', sincronizarEBuscar);

        async function requisitarDadosRecs() {
            const termo = campoBusca.value;
            if(!termo || termo.length < 2) return;

            const url = `/api/recomendar?termo=${encodeURIComponent(termo)}&w_knn=${pesoKnn.value}&w_ia=${pesoIa.value}`;
            const resposta = await fetch(url);
            const filmes = await resposta.json();

            const container = document.getElementById('containerResultados');
            container.innerHTML = '';
            document.getElementById('tituloResultados').textContent = `Filmes Similares a "${termo}"`;

            filmes.forEach(f => {
                const isCult = f.cultural === 1;
                
                const temasFormatados = f.temas.replace(/\|/g, '•').replace(/\b\w/g, c => c.toUpperCase());
                const atmosferaFormatada = f.atmosfera.replace(/\b\w/g, c => c.toUpperCase());

                const linha = document.createElement('div');
                linha.className = 'modulo-filme';
                linha.innerHTML = `
                    <div style="text-align: center; border-right: 1px solid var(--cinza-borda); padding-right: 15px;">
                        <div class="pct-match">${f.score}%</div>
                        <div class="lbl-match">Match</div>
                    </div>
                    <div class="info-principal">
                        <h3>${f.titulo}</h3>
                        <h4>Título Original: ${f.titulo_original}</h4>
                        <div style="font-size: 12px; color: var(--texto-palido);">
                            Ano: ${f.ano} | Nota: <i class="fa-solid fa-star" style="color:#ffd700;"></i> ${f.nota} | Gêneros: ${f.generos}
                        </div>
                        <div class="badges-container">
                            <span class="badge ${isCult ? 'badge-cult' : 'badge-com'}">${isCult ? 'Relevância Cultural' : 'Foco Comercial'}</span>
                            <span class="badge badge-ia">IA Confiança: ${f.confianca}%</span>
                        </div>
                    </div>
                    <div class="info-analitica">
                        <div style="margin-bottom: 5px;"><strong>Temas:</strong> ${temasFormatados}</div>
                        <div style="margin-bottom: 5px;"><strong>Atmosfera:</strong> ${atmosferaFormatada}</div>
                        <div style="font-style: italic; color: var(--texto-palido);">"${f.justificativa}"</div>
                    </div>
                `;
                container.appendChild(linha);
            });
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def pagina_index():
    lista_titulos = sorted(df_filmes['Titulo_PT'].unique().tolist())
    t = Template(CODIGO_INTERFACE)
    return t.render(titulos=lista_titulos, generos=lista_generos_pt)

@app.get("/api/sugestoes")
async def api_sugestoes(categoria: str = Query("")):
    filmes_do_gen = df_filmes[df_filmes['Generos'].apply(
        lambda x: categoria.lower() in traduzir_lista_generos(str(x)).lower()
    )]
    
    if filmes_do_gen.empty:
        return []
        
    sugestoes = filmes_do_gen.sort_values(by='vote_count', ascending=False).head(12)
    return JSONResponse(content=sugestoes['Titulo_PT'].tolist())

@app.get("/api/recomendar")
async def api_endpoint_recomendar(termo: str = Query(""), w_knn: float = Query(0.6), w_ia: float = Query(0.4)):
    return JSONResponse(content=executar_recomendacao(termo, w_knn, w_ia))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)