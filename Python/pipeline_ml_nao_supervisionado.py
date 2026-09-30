"""
Pipeline de Machine Learning Não Supervisionado para Detecção de Anomalias
em Compras Diretas do Estado do Rio de Janeiro (contasRJ)

Metodologia:
1. Pré-processamento e Normalização (RobustScaler / StandardScaler)
2. Análise de Agrupamento (K-Means) com busca de K via Método do Cotovelo e Silhueta (K >= 20)
3. Avaliação de Divergência entre Clusters (tamanho do grupo, distância do centroide global, desvio de preços)
4. Detecção de Outliers Multidimensionais com Isolation Forest
5. Matriz de Risco e Consenso de Anomalias (Cluster Divergente + Isolation Forest)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')

def preparar_features(df):
    """
    Realiza a engenharia de atributos e preparação dos dados numéricos e categóricos.
    """
    df_prep = df.copy()
    
    # Tratamento de valores nulos e conversão de tipos
    for col in ['ValorProcesso', 'ValorUnitario', 'QuantidadeItem', 'ValorTotalItem']:
        if col in df_prep.columns:
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce').fillna(0)
            
    # Criar transformações logarítmicas para atenuar assimetria financeira
    df_prep['log_valor_unitario'] = np.log1p(np.maximum(0, df_prep.get('ValorUnitario', 0)))
    df_prep['log_valor_processo'] = np.log1p(np.maximum(0, df_prep.get('ValorProcesso', 0)))
    df_prep['log_quantidade'] = np.log1p(np.maximum(0, df_prep.get('QuantidadeItem', 0)))
    
    # Razão entre valor do item e valor total do processo (se disponíveis)
    if 'ValorTotalItem' in df_prep.columns and 'ValorProcesso' in df_prep.columns:
        df_prep['proporcao_item_processo'] = np.where(
            df_prep['ValorProcesso'] > 0,
            df_prep['ValorTotalItem'] / df_prep['ValorProcesso'],
            0
        )
    else:
        df_prep['proporcao_item_processo'] = 0.0

    features_numericas = ['log_valor_unitario', 'log_valor_processo', 'log_quantidade', 'proporcao_item_processo']
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_prep[features_numericas])
    
    return df_prep, X_scaled, features_numericas, scaler

def avaliar_cotovelo_silhueta(X_scaled, k_range=range(2, 31, 2), sample_size=5000):
    """
    Executa testes de K-Means para diferentes valores de K, calculando a Inércia (Cotovelo)
    e o Coeficiente de Silhueta.
    """
    print("--- Executando Análise de Cotovelo e Silhueta ---")
    resultados = []
    
    # Subamostragem para cálculo rápido da silhueta caso o dataset seja volumoso
    n_samples = min(sample_size, X_scaled.shape[0])
    indices_sub = np.random.choice(X_scaled.shape[0], size=n_samples, replace=False) if X_scaled.shape[0] > sample_size else np.arange(X_scaled.shape[0])
    X_sub = X_scaled[indices_sub]
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inercia = kmeans.inertia_
        
        sil = silhouette_score(X_sub, labels[indices_sub])
        
        resultados.append({
            'k': k,
            'inercia': inercia,
            'silhueta': sil
        })
        print(f"K = {k:2d} | Inércia: {inercia:12.2f} | Silhueta: {sil:.4f}")
        
    return pd.DataFrame(resultados)

def treinar_clustering_divergente(df_prep, X_scaled, n_clusters=20):
    """
    Treina o modelo K-Means com K selecionado (mínimo recomendado de 20 clusters)
    e identifica os clusters mais divergentes/anômalos.
    """
    print(f"\n--- Treinando K-Means com K = {n_clusters} ---")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    
    # Calcular distância de cada ponto ao seu próprio centroide
    distancias_centroide_proprio = np.linalg.norm(X_scaled - centroids[labels], axis=1)
    
    # Calcular centroide global para medir a divergência do grupo
    centroide_global = np.mean(X_scaled, axis=0)
    distancia_clusters_global = np.linalg.norm(centroids - centroide_global, axis=1)
    
    # Analisar perfil de cada cluster
    df_prep['cluster_id'] = labels
    df_prep['dist_centroide_proprio'] = distancias_centroide_proprio
    
    cluster_metrics = []
    total_amostras = len(df_prep)
    
    for c_id in range(n_clusters):
        mask = (labels == c_id)
        tamanho = np.sum(mask)
        percentual = (tamanho / total_amostras) * 100
        
        # Média e mediana dos preços unitários originais do cluster
        val_unit_med = df_prep.loc[mask, 'ValorUnitario'].median() if 'ValorUnitario' in df_prep.columns else 0
        val_unit_mean = df_prep.loc[mask, 'ValorUnitario'].mean() if 'ValorUnitario' in df_prep.columns else 0
        
        dist_global = distancia_clusters_global[c_id]
        
        # Score de Divergência do Cluster (combina raridade do grupo e distância do centroide global)
        # Clusters pequenos e afastados do centro populacional recebem maior pontuação de divergência
        score_divergencia = dist_global * (1 / np.log2(tamanho + 2))
        
        cluster_metrics.append({
            'cluster_id': c_id,
            'tamanho_amostras': tamanho,
            'percentual_base': percentual,
            'distancia_centro_global': dist_global,
            'valor_unitario_medio': val_unit_mean,
            'valor_unitario_mediana': val_unit_med,
            'score_divergencia_cluster': score_divergencia
        })
        
    df_clusters = pd.DataFrame(cluster_metrics).sort_values(by='score_divergencia_cluster', ascending=False)
    
    # Classificar os N clusters de maior divergência (top 20%)
    top_divergentes = set(df_clusters.head(int(np.ceil(n_clusters * 0.2)))['cluster_id'])
    df_prep['flag_cluster_divergente'] = df_prep['cluster_id'].isin(top_divergentes).astype(int)
    
    return df_prep, kmeans, df_clusters

def treinar_isolation_forest(X_scaled, contamination=0.05):
    """
    Aplica o algoritmo Isolation Forest para detectar anomalias multidimensionais
    sem necessidade de rótulos prévios.
    """
    print(f"\n--- Treinando Isolation Forest (Contaminação esperada: {contamination*100:.1f}%) ---")
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    
    # Isolation forest retorna -1 para anomalias e 1 para inliers
    preds = iso_forest.fit_predict(X_scaled)
    flags_anomalia = np.where(preds == -1, 1, 0)
    
    # Score de decisão inverter sinal (quanto mais negativo/alto, mais anômalo)
    decision_scores = -iso_forest.score_samples(X_scaled)
    
    return flags_anomalia, decision_scores, iso_forest

def consolidar_consenso_anomalias(df_prep, df_clusters):
    """
    Consolida os resultados dos modelos não supervisionados (Clustering + Isolation Forest)
    criando uma matriz de risco isenta de viés de rotulagem sintética interna.
    """
    print("\n--- Consolidando Matriz de Risco Não Supervisionada ---")
    
    # Normalização min-max do score do Isolation Forest para escala 0 - 100
    min_s, max_s = df_prep['iforest_score'].min(), df_prep['iforest_score'].max()
    score_if_norm = ((df_prep['iforest_score'] - min_s) / (max_s - min_s + 1e-9)) * 100
    
    # Mapear score de divergência do cluster para cada registro
    mapa_divergencia = dict(zip(df_clusters['cluster_id'], df_clusters['score_divergencia_cluster']))
    df_prep['score_divergencia_cluster'] = df_prep['cluster_id'].map(mapa_divergencia)
    
    min_d, max_d = df_prep['score_divergencia_cluster'].min(), df_prep['score_divergencia_cluster'].max()
    score_cluster_norm = ((df_prep['score_divergencia_cluster'] - min_d) / (max_d - min_d + 1e-9)) * 100
    
    # Score Consolidado de Suspeição (50% Divergência do Cluster + 50% Isolation Forest)
    df_prep['score_suspeicao_nao_supervisionado'] = (0.5 * score_cluster_norm) + (0.5 * score_if_norm)
    
    # Flag de Consenso de Anomalia (Quando pertence a cluster divergente E é marcado pelo Isolation Forest)
    df_prep['consenso_anomalia'] = np.where(
        (df_prep['flag_cluster_divergente'] == 1) & (df_prep['flag_iforest_anomalia'] == 1),
        1,
        0
    )
    
    print("Resumo do Consenso de Anomalias:")
    print(f"Total de Registros Analisados: {len(df_prep)}")
    print(f"Registros em Clusters Divergentes: {df_prep['flag_cluster_divergente'].sum()}")
    print(f"Anomalias apontadas pelo Isolation Forest: {df_prep['flag_iforest_anomalia'].sum()}")
    print(f"Anomalias com Consenso Duplo (Cluster + IF): {df_prep['consenso_anomalia'].sum()}")
    
    return df_prep

def executar_pipeline_completo(caminho_csv='df_limpo.csv', n_clusters_escolhido=20):
    """
    Função principal para carregar os dados, executar avaliação de K,
    clusterização com N >= 20, Isolation Forest e salvar o dataset enriquecido.
    """
    if not os.path.exists(caminho_csv):
        print(f"Arquivo {caminho_csv} não encontrado. Gerando dados sintéticos para demonstração técnica...")
        np.random.seed(42)
        n = 1000
        df_dummy = pd.DataFrame({
            'Processo': [f"PROC-{i//5}" for i in range(n)],
            'ValorProcesso': np.random.exponential(scale=50000, size=n),
            'ValorUnitario': np.random.exponential(scale=200, size=n),
            'QuantidadeItem': np.random.randint(1, 100, size=n),
            'ValorTotalItem': np.random.exponential(scale=1000, size=n)
        })
        # Injetar alguns outliers deliberados
        df_dummy.loc[10:15, 'ValorUnitario'] *= 50
        df_prep, X_scaled, features, scaler = preparar_features(df_dummy)
    else:
        df_raw = pd.read_csv(caminho_csv)
        df_prep, X_scaled, features, scaler = preparar_features(df_raw)
        
    # 1. Teste de Cotovelo e Silhueta para escolha do K
    df_avaliacoes_k = avaliar_cotovelo_silhueta(X_scaled, k_range=range(5, 31, 5))
    
    # 2. Clusterização com K >= 20
    df_prep, model_kmeans, df_clusters = treinar_clustering_divergente(df_prep, X_scaled, n_clusters=n_clusters_escolhido)
    
    # 3. Isolation Forest
    flags_if, scores_if, model_if = treinar_isolation_forest(X_scaled, contamination=0.05)
    df_prep['flag_iforest_anomalia'] = flags_if
    df_prep['iforest_score'] = scores_if
    
    # 4. Consolidar Matriz de Risco Não Supervisionada
    df_resultado = consolidar_consenso_anomalias(df_prep, df_clusters)
    
    # 5. Salvar Resultados
    caminho_saida = 'df_ml_nao_supervisionado_resultados.csv'
    df_resultado.to_csv(caminho_saida, index=False)
    print(f"\n✅ Processo concluído! Dataset enriquecido salvo em '{caminho_saida}'.")
    
    return df_resultado, df_clusters, df_avaliacoes_k

if __name__ == '__main__':
    executar_pipeline_completo()
