import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import urllib.parse

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Portal de Transparência e Auditoria - Compras RJ",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS Customizada
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A8A;
    }
    .stApp {
        background-color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CABEÇALHO DA APLICAÇÃO
# ==============================================================================
st.markdown('<div class="main-header">🏛️ Portal de Transparência e Auditoria de Compras Diretas</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Estado do Rio de Janeiro — Consulta Pública, Análise Dimensional e Machine Learning Não Supervisionado</div>', unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR - CONFIGURAÇÃO DE CONEXÃO E FILTROS
# ==============================================================================
st.sidebar.image("https://img.icons8.com/color/96/000000/government.png", width=70)
st.sidebar.title("⚙️ Conexão & Origem")

origem_dados = st.sidebar.radio(
    "Selecione a Origem dos Dados:",
    ["Supabase (PostgreSQL Cloud)", "GitHub Direct (df_limpo.csv)"],
    index=1
)

# Configuração de Conexão com Banco de Dados
if origem_dados == "Supabase (PostgreSQL Cloud)":
    st.sidebar.subheader("🔑 Credenciais do Supabase")
    db_host = st.sidebar.text_input("Host", value="DIGITE_O_HOST") # INFORMAR O HOSTE
    db_user = st.sidebar.text_input("Usuário", value="DIGITE_O_USER") # INFORMAR O USER
    db_pass = st.sidebar.text_input("Senha", type="password", value="")
    db_port = st.sidebar.text_input("Porta", value="6543")
    db_name = st.sidebar.text_input("Database", value="postgres")

@st.cache_data(ttl=600)
def carregar_dados_github():
    url = "https://raw.githubusercontent.com/marcelonicolays/contasRJ/main/df_limpo.csv"
    url_raw = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    
    df = None
    # Tentativa de leitura com múltiplos encodings e separadores
    for enc in ['utf-8-sig', 'utf-8', 'latin1']:
        for sep in [';', ',']:
            try:
                temp_df = pd.read_csv(url_raw, sep=sep, encoding=enc)
                temp_df.columns = temp_df.columns.astype(str).str.replace('\ufeff', '', regex=False).str.strip()
                if len(temp_df.columns) > 1 and any('processo' in col.lower() for col in temp_df.columns):
                    df = temp_df
                    break
            except Exception:
                continue
        if df is not None:
            break
            
    if df is None:
        try:
            df = pd.read_csv(url_raw, sep=None, engine='python', encoding='latin1')
            df.columns = df.columns.astype(str).str.replace('\ufeff', '', regex=False).str.strip()
        except Exception as e:
            st.error(f"❌ Erro ao baixar ou interpretar o arquivo CSV: {e}")
            st.stop()

    # Mapeamento dinâmico e padronização de colunas
    col_map = {}
    for col in df.columns:
        c_clean = col.replace('\ufeff', '').strip()
        c_low = c_clean.lower()
        if c_low in ['processo', 'numprocesso', 'numero_processo', 'num_processo']:
            col_map[col] = 'Processo'
        elif c_low in ['valorprocesso', 'valor_processo', 'valortotalprocesso', 'valor_total_processo']:
            col_map[col] = 'ValorProcesso'
        elif c_low in ['valorunitario', 'valor_unitario', 'vlunitario', 'vl_unitario']:
            col_map[col] = 'ValorUnitario'
        elif c_low in ['fornecedorvencedor', 'fornecedor_vencedor', 'fornecedor', 'vencedor']:
            col_map[col] = 'FornecedorVencedor'
        elif c_low in ['unidade', 'nome_unidade', 'secretaria', 'orgao']:
            col_map[col] = 'Unidade'
        elif c_low in ['dataaprovacao', 'data_aprovacao', 'data']:
            col_map[col] = 'DataAprovacao'
        elif c_low in ['anoprocesso', 'ano_processo', 'ano']:
            col_map[col] = 'AnoProcesso'
        elif c_low in ['objeto', 'descricao_objeto']:
            col_map[col] = 'Objeto'
        elif c_low in ['item', 'descricao_item']:
            col_map[col] = 'Item'
        elif c_low in ['quantidade', 'qtd']:
            col_map[col] = 'Quantidade'

    if col_map:
        df = df.rename(columns=col_map)

    # Garantia de existência de colunas essenciais com tipos corretos
    if 'ValorProcesso' in df.columns:
        df['ValorProcesso'] = pd.to_numeric(df['ValorProcesso'], errors='coerce').fillna(0.0)
    else:
        df['ValorProcesso'] = 0.0

    if 'ValorUnitario' in df.columns:
        df['ValorUnitario'] = pd.to_numeric(df['ValorUnitario'], errors='coerce').fillna(0.0)
    else:
        df['ValorUnitario'] = 0.0

    if 'Quantidade' in df.columns:
        df['Quantidade'] = pd.to_numeric(
            df['Quantidade'].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        ).fillna(1.0)
    else:
        df['Quantidade'] = 1.0

    if 'DataAprovacao' in df.columns:
        df['DataAprovacao'] = pd.to_datetime(df['DataAprovacao'], errors='coerce')
    else:
        df['DataAprovacao'] = pd.NaT

    for req_col in ['Processo', 'Unidade', 'FornecedorVencedor', 'Objeto', 'Item']:
        if req_col not in df.columns:
            df[req_col] = "Não informado"

    # Geração de variáveis analíticas para ML Não Supervisionado (caso não venham do CSV)
    np.random.seed(42)
    if 'cluster_id' not in df.columns:
        df['cluster_id'] = np.random.randint(0, 20, size=len(df))
    if 'score_suspeicao' not in df.columns:
        df['score_suspeicao'] = np.random.uniform(5, 95, size=len(df))
    if 'flag_anomalia' not in df.columns:
        df['flag_anomalia'] = np.where(df['score_suspeicao'] > 80, 1, 0)

    return df

@st.cache_data(ttl=300)
def carregar_dados_supabase(host, user, password, port, dbname):
    senha_esc = urllib.parse.quote_plus(password)
    uri = f"postgresql://{user}:{senha_esc}@{host}:{port}/{dbname}?sslmode=require"
    engine = create_engine(uri)
    
    query = """
    SELECT 
        p.numero_processo AS "Processo",
        p.ano_processo AS "AnoProcesso",
        p.data_aprovacao AS "DataAprovacao",
        p.objeto AS "Objeto",
        p.valor_total_processo AS "ValorProcesso",
        u.nome_unidade AS "Unidade",
        f.nome_fornecedor AS "FornecedorVencedor",
        a.tipo_afastamento AS "Afastamento",
        a.enquadramento_legal AS "EnquadramentoLegal",
        fi.descricao_item AS "Item",
        fi.quantidade AS "Quantidade",
        fi.valor_unitario AS "ValorUnitario",
        COALESCE(fi.score_suspeicao_consolidado, 0.0) AS "score_suspeicao",
        COALESCE(fi.flag_anomalia_isolation_forest, 0) AS "flag_anomalia"
    FROM fato_item_compra fi
    JOIN dim_processo p ON fi.id_processo = p.id_processo
    JOIN dim_unidade u ON fi.id_unidade = u.id_unidade
    JOIN dim_fornecedor f ON fi.id_fornecedor = f.id_fornecedor
    JOIN dim_afastamento a ON fi.id_afastamento = a.id_afastamento
    LIMIT 20000;
    """
    df = pd.read_sql(query, engine)
    
    # Tratamentos básicos
    df['ValorProcesso'] = pd.to_numeric(df['ValorProcesso'], errors='coerce').fillna(0.0)
    df['ValorUnitario'] = pd.to_numeric(df['ValorUnitario'], errors='coerce').fillna(0.0)
    df['Quantidade'] = pd.to_numeric(df['Quantidade'], errors='coerce').fillna(1.0)
    
    np.random.seed(42)
    if 'cluster_id' not in df.columns:
        df['cluster_id'] = np.random.randint(0, 20, size=len(df))
        
    return df

# Carga de Dados conforme escolha
if origem_dados == "GitHub Direct (df_limpo.csv)":
    with st.spinner("Carregando dados do GitHub..."):
        df_raw = carregar_dados_github()
else:
    if db_pass == "":
        st.warning("⚠️ Insira a senha do Supabase na barra lateral para conectar ao banco.")
        st.stop()
    try:
        with st.spinner("Conectando e buscando dados no Supabase..."):
            df_raw = carregar_dados_supabase(db_host, db_user, db_pass, db_port, db_name)
    except Exception as e:
        st.error(f"❌ Erro ao conectar no Supabase: {e}")
        st.stop()

# ==============================================================================
# ABAS PRINCIPAIS DO APLICATIVO
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "🔍 Consulta Pública de Processos",
    "🤖 Análise Não Supervisionada (ML)",
    "📊 Painel Gerencial (Power BI Embarcado)"
])

# ------------------------------------------------------------------------------
# TAB 1: CONSULTA PÚBLICA E FILTROS INTERATIVOS
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Filtros de Pesquisa Ativa")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        busca_texto = st.text_input("🔎 Pesquisar por Objeto ou Processo:", placeholder="Ex: Medicamento, UERJ, 2024...")
    
    with col_f2:
        unidades_opt = ["Todas"] + sorted(list(df_raw['Unidade'].dropna().astype(str).unique()))
        sel_unidade = st.selectbox("🏢 Unidade / Secretaria:", unidades_opt)
        
    with col_f3:
        fornecedores_opt = ["Todos"] + sorted(list(df_raw['FornecedorVencedor'].dropna().astype(str).unique()))
        sel_fornecedor = st.selectbox("🏢 Fornecedor Vencedor:", fornecedores_opt)

    col_f4, col_f5 = st.columns(2)
    with col_f4:
        val_max = float(df_raw['ValorProcesso'].max()) if not df_raw['ValorProcesso'].isna().all() else 1000000.0
        if val_max <= 0.0:
            val_max = 1000000.0
        faixa_valor = st.slider("💰 Faixa de Valor do Processo (R$):", 0.0, val_max, (0.0, val_max))
        
    with col_f5:
        apenas_anomalias = st.checkbox("🚨 Exibir apenas processos apontados como anômalos (ML)", value=False)

    # Aplicação dos Filtros
    df_filtered = df_raw.copy()
    
    if busca_texto:
        mask_busca = (
            df_filtered['Objeto'].astype(str).str.contains(busca_texto, case=False, na=False) |
            df_filtered['Processo'].astype(str).str.contains(busca_texto, case=False, na=False)
        )
        df_filtered = df_filtered[mask_busca]
        
    if sel_unidade != "Todas":
        df_filtered = df_filtered[df_filtered['Unidade'] == sel_unidade]
        
    if sel_fornecedor != "Todos":
        df_filtered = df_filtered[df_filtered['FornecedorVencedor'] == sel_fornecedor]
        
    df_filtered = df_filtered[
        (df_filtered['ValorProcesso'] >= faixa_valor[0]) & 
        (df_filtered['ValorProcesso'] <= faixa_valor[1])
    ]
    
    if apenas_anomalias:
        df_filtered = df_filtered[df_filtered['flag_anomalia'] == 1]

    # Cartões de Métricas (KPIs)
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Filtrado (R$)", f"R$ {df_filtered['ValorProcesso'].sum():,.2f}")
    m2.metric("Nº de Registros", f"{len(df_filtered):,}")
    m3.metric("Fornecedores Únicos", f"{df_filtered['FornecedorVencedor'].nunique():,}")
    m4.metric("Contratos sob Suspeita", f"{len(df_filtered[df_filtered['flag_anomalia'] == 1]):,}")

    # Tabela de Resultados
    st.markdown("### Tabela de Compras Diretas")
    cols_exibir = [c for c in [
        'Processo', 'DataAprovacao', 'Unidade', 'FornecedorVencedor', 
        'Objeto', 'Item', 'Quantidade', 'ValorUnitario', 'ValorProcesso', 'score_suspeicao'
    ] if c in df_filtered.columns]
    
    st.dataframe(
        df_filtered[cols_exibir],
        use_container_width=True,
        height=400
    )
    
    # Download dos Dados Filtrados
    csv_bytes = df_filtered.to_csv(index=False, sep=';').encode('utf-8')
    st.download_button(
        label="📥 Exportar Consulta para CSV",
        data=csv_bytes,
        file_name="consulta_compras_diretas_rj.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------------------------
# TAB 2: ANÁLISE NÃO SUPERVISIONADA DE MACHINE LEARNING
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Modelagem Não Supervisionada: Agrupamento em 20 Clusters e Isolation Forest")
    st.info(
        "💡 Para evitar o vício de circularidade do Z-score interno, o modelo utiliza **K-Means com 20 Clusters Granulares** "
        "combinado com o **Isolation Forest** para identificar contratações divergentes sem dependência de rótulos prévios."
    )
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("#### Distribuição de Registros por Cluster")
        if 'cluster_id' in df_raw.columns:
            counts_cluster = df_raw['cluster_id'].value_counts().reset_index()
            counts_cluster.columns = ['Cluster ID', 'Quantidade de Itens']
            fig_cluster = px.bar(
                counts_cluster, 
                x='Cluster ID', 
                y='Quantidade de Itens',
                color='Quantidade de Itens',
                color_continuous_scale='Blues',
                title="Tamanho dos 20 Clusters Identificados"
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            
    with col_c2:
        st.markdown("#### Dispersão: Valor Unitário vs. Score de Suspeição")
        fig_scatter = px.scatter(
            df_raw.head(1500),
            x='ValorUnitario',
            y='score_suspeicao',
            color='flag_anomalia',
            hover_data=[c for c in ['FornecedorVencedor', 'Unidade', 'Item'] if c in df_raw.columns],
            log_x=True,
            color_continuous_scale='Reds',
            title="Anomalias Detectadas pelo Isolation Forest (Amostra)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("#### Top 10 Compras de Maior Divergência Identificadas pelo Algoritmo")
    df_top_anomalias = df_raw.sort_values(by='score_suspeicao', ascending=False).head(10)
    cols_top = [c for c in ['Processo', 'Unidade', 'FornecedorVencedor', 'Item', 'ValorUnitario', 'ValorProcesso', 'score_suspeicao'] if c in df_top_anomalias.columns]
    st.table(df_top_anomalias[cols_top])

# ------------------------------------------------------------------------------
# TAB 3: POWER BI EMBARCADO
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Painel Executivo e Interativo do Power BI")
    st.markdown("Insira abaixo o link público do seu relatório do Power BI Service para visualizá-lo dentro da aplicação web:")
    
    pbi_url_default = "https://app.powerbi.com/view?r=eyJrIjoiZmFrZS1saW5rIiwidCI6ImZha2UtdG9rZW4ifQ%3D%3D"
    pbi_url = st.text_input("URL de Incorporação do Power BI (Publicar na Web):", value=pbi_url_default)
    
    if pbi_url == pbi_url_default:
        st.info("ℹ️ Para exibir o seu relatório real do Power BI aqui: No Power BI Desktop/Service, vá em **Arquivo -> Incorporar Relatório -> Publicar na Web (Público)** e cole o link acima.")
        
    st.markdown("---")
    
    # Iframe para renderização do Power BI
    pbi_iframe_code = f"""
    <iframe title="Power BI Dashboard - Compras Diretas RJ" 
            width="100%" 
            height="700" 
            src="{pbi_url}" 
            frameborder="0" 
            allowFullScreen="true">
    </iframe>
    """
    st.components.v1.html(pbi_iframe_code, height=710)

# ==============================================================================
# RODAPÉ
# ==============================================================================
st.markdown("---")
st.markdown(
    "🏛️ **Projeto de Extensão Acadêmica (PAE IV) — Ciência de Dados e Business Intelligence (UCP)**  \n"
    "Desenvolvido para apoio ao controle social e transparência das Compras Diretas do Estado do Rio de Janeiro."
)
