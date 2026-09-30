"""
Script de ETL e Carga dos Dados de Compras Diretas do Estado do Rio de Janeiro
Versão 3 (v3 - Corrigida) - Tratamento Robusto de Separador CSV (Semicólon ';') e Supabase
"""

import os
import sys
import urllib.parse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# ==============================================================================
# CONFIGURAÇÕES DE CONEXÃO DO SUPABASE (ALTERE AQUI OS SEUS DADOS)
# ==============================================================================

# OPÇÃO A (RECOMENDADA): Preencha os campos individuais
SUPABASE_HOST = "INFORME_O_HOST"      # Ex: db.xxxx.supabase.co
SUPABASE_USER = "USUARIO_DO_BANCO_DE_DADOS"                  # Ex: postgres.aivrloittxfssdyw (Pooler) ou postgres
SUPABASE_PASSWORD = "SENHA_DO_BANCO_DE_DADOS"            # Sua senha do banco Supabase
SUPABASE_DB = "postgres"
SUPABASE_PORT = 6543                         # 5432 (Direto) ou 6543 (Pooler)

# OPÇÃO B: Se preferir colar a URI completa do Supabase (mantenha None se usar a OPÇÃO A)
DB_URI_CUSTOM = None

# Arquivo de dados (URL Raw do GitHub ou caminho local 'df_limpo.csv')
CSV_PATH = "https://raw.githubusercontent.com/marcelonicolays/contasRJ/main/df_limpo.csv"

# Arquivo SQL de criação do Schema
SCHEMA_SQL_PATH = "schema_postgresql_compras_rj.sql"
EXECUTE_SCHEMA_IF_EXISTS = False


def obter_db_uri() -> str:
    """Gera a URI de conexão segura com tratamento automático de senha e SSL."""
    if DB_URI_CUSTOM and "SuaSenhaAqui" not in DB_URI_CUSTOM:
        return DB_URI_CUSTOM
    
    senha_segura = urllib.parse.quote_plus(SUPABASE_PASSWORD)
    return f"postgresql://{SUPABASE_USER}:{senha_segura}@{SUPABASE_HOST}:{SUPABASE_PORT}/{SUPABASE_DB}?sslmode=require"


def carregar_dataframe_csv(caminho_ou_url: str) -> pd.DataFrame:
    """Carrega o CSV tratando corretamente o separador ponto e vírgula (;) do dataset do TCERJ."""
    url_final = caminho_ou_url
    if caminho_ou_url.startswith("http://") or caminho_ou_url.startswith("https://"):
        url_final = caminho_ou_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    
    print(f"   Lendo dados de: {url_final}")
    
    # 1. Tenta primeiramente com ponto e vírgula (;) que é o padrão original do df_limpo.csv
    try:
        df = pd.read_csv(url_final, sep=';', encoding='utf-8')
        # Se por acaso leu tudo em 1 única coluna, tenta com vírgula
        if len(df.columns) == 1 and ';' not in df.columns[0]:
            df = pd.read_csv(url_final, sep=',', encoding='utf-8')
    except Exception:
        df = pd.read_csv(url_final, sep=',', encoding='utf-8')
        
    # Limpa nomes das colunas e remove caracteres especiais/BOM
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    
    # Validação de segurança: se a coluna 'Processo' não existir, tenta auto-detecção com engine='python'
    if 'Processo' not in df.columns:
        df = pd.read_csv(url_final, sep=None, engine='python', encoding='utf-8')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

    if 'Processo' not in df.columns:
        raise KeyError(f"Coluna 'Processo' não encontrada. Colunas identificadas no arquivo: {list(df.columns)}")

    return df


def executar_etl_e_carga():
    """Executa o pipeline completo de leitura, tratamento e carga no Supabase."""
    db_uri = obter_db_uri()
    
    if "SuaSenhaAqui" in db_uri or "db.xxxxxx.supabase.co" in db_uri:
        print("❌ ERRO: Você precisa preencher a senha e o host do seu projeto Supabase no início do código!")
        print("   Exemplo: SUPABASE_HOST = 'aws-0-sa-east-1.pooler.supabase.com'")
        print("   Exemplo: SUPABASE_USER = 'postgres.aivrloittxzbmbfssdyw'")
        print("   Exemplo: SUPABASE_PASSWORD = 'minha_senha_super_segura'")
        return

    # 1. Testando Conexão
    print("1. Testando conexão com o Supabase...")
    try:
        engine = create_engine(db_uri, connect_args={"connect_timeout": 15})
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();")).fetchone()
            print("   ✅ Conexão estabelecida com sucesso!")
            print(f"   Versão do Banco: {result[0][:50]}...")
    except Exception as e:
        print("\n❌ FALHA NA CONEXÃO COM O SUPABASE!")
        print("----------------------------------------------------------------------")
        print(f"Detalhes do erro:\n{e}")
        print("----------------------------------------------------------------------")
        return

    # 2. Leitura dos Dados
    print(f"\n2. Lendo dados do dataset...")
    try:
        df = carregar_dataframe_csv(CSV_PATH)
        print(f"   ✅ Dataset carregado com sucesso! ({len(df)} linhas x {len(df.columns)} colunas)")
        print(f"   Colunas identificadas: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Erro ao ler o CSV: {e}")
        return

    # 3. Executando DDL se solicitado
    if EXECUTE_SCHEMA_IF_EXISTS and SCHEMA_SQL_PATH and os.path.exists(SCHEMA_SQL_PATH):
        print(f"\n3. Executando DDL de Schema ({SCHEMA_SQL_PATH})...")
        try:
            with open(SCHEMA_SQL_PATH, 'r', encoding='utf-8') as f:
                sql_schema = f.read()
            with engine.begin() as conn:
                conn.execute(text(sql_schema))
            print("   ✅ Schema de tabelas verificado/criado com sucesso!")
        except Exception as e:
            print(f"   ⚠️ Aviso ao criar schema: {e}. Prosseguindo com a inserção...")

    # 4. Tratamento dos Dados
    print("\n4. Aplicando tratamentos de tipos e nulos...")
    df['DataAprovacao'] = pd.to_datetime(df['DataAprovacao'], errors='coerce')
    df['ValorProcesso'] = pd.to_numeric(df['ValorProcesso'], errors='coerce')
    df['ValorUnitario'] = pd.to_numeric(df['ValorUnitario'], errors='coerce')
    df['AnoProcesso'] = pd.to_numeric(df['AnoProcesso'], errors='coerce').astype('Int64')
    
    if 'Quantidade' in df.columns:
        df['Quantidade'] = pd.to_numeric(
            df['Quantidade'].astype(str).str.replace(',', '.', regex=False), 
            errors='coerce'
        ).fillna(1.0)
    else:
        df['Quantidade'] = 1.0

    df['Objeto'] = df['Objeto'].fillna('Não informado').astype(str).str.strip()
    df['Unidade'] = df['Unidade'].fillna('Não informada').astype(str).str.strip()
    df['FornecedorVencedor'] = df['FornecedorVencedor'].fillna('Não informado').astype(str).str.strip()
    df['Afastamento'] = df['Afastamento'].fillna('Não informado').astype(str).str.strip()
    df['EnquadramentoLegal'] = df['EnquadramentoLegal'].fillna('Não informado').astype(str).str.strip()
    df['Item'] = df['Item'].fillna('Item não especificado').astype(str).str.strip()
    df['UnidadeMedida'] = df['UnidadeMedida'].fillna('UN').astype(str).str.strip()

    # Limpeza preventiva do banco para re-execuções limpas (sem duplicar chaves)
    try:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE fato_item_compra, dim_processo, dim_unidade, dim_fornecedor, dim_afastamento RESTART IDENTITY CASCADE;"))
    except Exception:
        pass

    # 5. Carga nas Tabelas de Dimensão
    print("\n5. Populando 'dim_unidade'...")
    df_unidades = pd.DataFrame({'nome_unidade': sorted(df['Unidade'].unique())})
    df_unidades.to_sql('dim_unidade', engine, if_exists='append', index=False)
    dim_unidades_db = pd.read_sql('SELECT id_unidade, nome_unidade FROM dim_unidade', engine)

    print("6. Populando 'dim_fornecedor'...")
    df_fornecedores = pd.DataFrame({'nome_fornecedor': sorted(df['FornecedorVencedor'].unique())})
    df_fornecedores.to_sql('dim_fornecedor', engine, if_exists='append', index=False)
    dim_fornecedores_db = pd.read_sql('SELECT id_fornecedor, nome_fornecedor FROM dim_fornecedor', engine)

    print("7. Populando 'dim_afastamento'...")
    df_afastamentos = df[['Afastamento', 'EnquadramentoLegal']].drop_duplicates()
    df_afastamentos.columns = ['tipo_afastamento', 'enquadramento_legal']
    df_afastamentos.to_sql('dim_afastamento', engine, if_exists='append', index=False)
    dim_afastamento_db = pd.read_sql('SELECT id_afastamento, tipo_afastamento, enquadramento_legal FROM dim_afastamento', engine)

    # Mapeamentos
    df_mapped = df.merge(dim_unidades_db, left_on='Unidade', right_on='nome_unidade', how='left')
    df_mapped = df_mapped.merge(dim_fornecedores_db, left_on='FornecedorVencedor', right_on='nome_fornecedor', how='left')
    df_mapped = df_mapped.merge(
        dim_afastamento_db, 
        left_on=['Afastamento', 'EnquadramentoLegal'], 
        right_on=['tipo_afastamento', 'enquadramento_legal'], 
        how='left'
    )

    print("8. Populando 'dim_processo'...")
    df_processos = df_mapped[[
        'Processo', 'AnoProcesso', 'DataAprovacao', 'Objeto', 'ValorProcesso',
        'id_unidade', 'id_fornecedor', 'id_afastamento'
    ]].drop_duplicates(subset=['Processo']).copy()

    df_processos.columns = [
        'numero_processo', 'ano_processo', 'data_aprovacao', 'objeto', 'valor_total_processo',
        'id_unidade', 'id_fornecedor', 'id_afastamento'
    ]
    df_processos.to_sql('dim_processo', engine, if_exists='append', index=False)
    dim_processo_db = pd.read_sql('SELECT id_processo, numero_processo FROM dim_processo', engine)

    df_mapped = df_mapped.merge(dim_processo_db, left_on='Processo', right_on='numero_processo', how='left')

    print("9. Populando 'fato_item_compra'...")
    df_fato = pd.DataFrame({
        'id_processo': df_mapped['id_processo'],
        'id_unidade': df_mapped['id_unidade'],
        'id_fornecedor': df_mapped['id_fornecedor'],
        'id_afastamento': df_mapped['id_afastamento'],
        'descricao_item': df_mapped['Item'],
        'unidade_medida': df_mapped['UnidadeMedida'],
        'quantidade': df_mapped['Quantidade'],
        'valor_unitario': df_mapped['ValorUnitario']
    })

    df_fato.to_sql('fato_item_compra', engine, if_exists='append', index=False)
    print("\n🎉 CARGA CONCLUÍDA COM SUCESSO NO SUPABASE!")


if __name__ == "__main__":
    executar_etl_e_carga()
