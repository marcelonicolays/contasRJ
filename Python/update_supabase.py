import pandas as pd
from sqlalchemy import create_engine, text
import urllib.parse

# ==============================================================================
# 1. CONFIGURAÇÕES DO SUPABASE (Use a mesma senha do seu script ETL v3)
# ==============================================================================
SUPABASE_HOST = "DIGITE_O_HOST"# <-- Substitua pelo HOST
SUPABASE_USER = "DIGITE_O_USER"# <-- Substitua pelo user do banco de dados
SUPABASE_PASSWORD = "DIGITE_A_SENHA"  # <-- Substitua pela sua senha real do Supabase
SUPABASE_DB = "postgres"
SUPABASE_PORT = 6543

def atualizar_banco_com_ml():
    print("1. A ler o ficheiro de resultados de Machine Learning...")
    try:
        # Lê o ficheiro CSV que anexou
        df_ml = pd.read_csv("df_ml_nao_supervisionado_resultados.csv")
    except FileNotFoundError:
        print("❌ Erro: O ficheiro 'df_ml_nao_supervisionado_resultados.csv' não foi encontrado na pasta.")
        return

    # Selecionar apenas as colunas que precisamos para a atualização
    # (Mapeamento baseado no script pipeline_ml_nao_supervisionado.py)
    colunas_necessarias = ['Processo', 'Item', 'flag_iforest_anomalia', 'score_suspeicao_nao_supervisionado', 'cluster_id']
    df_update = df_ml[colunas_necessarias].copy()

    print("2. A estabelecer ligação ao Supabase...")
    senha_segura = urllib.parse.quote_plus(SUPABASE_PASSWORD)
    db_uri = f"postgresql://{SUPABASE_USER}:{senha_segura}@{SUPABASE_HOST}:{SUPABASE_PORT}/{SUPABASE_DB}?sslmode=require"
    engine = create_engine(db_uri)

    try:
        with engine.begin() as conn:
            print("3. A enviar dados para a tabela temporária no PostgreSQL...")
            # Envia o dataframe para o Supabase como uma tabela temporária
            df_update.to_sql('temp_ml_resultados', engine, if_exists='replace', index=False)

            print("4. A executar o Bulk Update na tabela fato_item_compra...")
            # Este comando SQL cruza a tabela temporária com a dim_processo para encontrar o ID correto,
            # e atualiza os scores e flags na fato_item_compra num único passo rápido.
            # Nota: O seu modelo DB pede -1 para anomalia e o ML gera 1. O CASE WHEN trata essa conversão.
            sql_update = """
            UPDATE fato_item_compra f
            SET 
                score_suspeicao_consolidado = t.score_suspeicao_nao_supervisionado,
                cluster_dbscan = t.cluster_id,
                flag_anomalia_isolation_forest = CASE WHEN t.flag_iforest_anomalia = 1 THEN -1 ELSE 1 END
            FROM temp_ml_resultados t
            JOIN dim_processo p ON p.numero_processo = t."Processo"
            WHERE f.id_processo = p.id_processo
              AND f.descricao_item = t."Item";
            """
            conn.execute(text(sql_update))

            print("5. A limpar a base de dados (Remover tabela temporária)...")
            conn.execute(text("DROP TABLE temp_ml_resultados;"))

        print("\n✅ Atualização concluída com sucesso! Os resultados do Machine Learning foram injetados no Supabase.")

    except Exception as e:
        print(f"\n❌ Ocorreu um erro durante a atualização da base de dados: {e}")

if __name__ == "__main__":
    atualizar_banco_com_ml()