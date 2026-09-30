-- ==============================================================================
-- MODELO DE BANCO DE DADOS POSTGRESQL - MODELO ESTRELA (STAR SCHEMA)
-- Projeto: Análise de Compras Diretas do Estado do Rio de Janeiro (contasRJ)
-- Base de Origem: df_limpo.csv (19.676 registros, 13 colunas originais)
-- ==============================================================================

-- Remove tabelas se já existirem (para execução limpa/idempotente)
DROP TABLE IF EXISTS fato_item_compra CASCADE;
DROP TABLE IF EXISTS dim_processo CASCADE;
DROP TABLE IF EXISTS dim_afastamento CASCADE;
DROP TABLE IF EXISTS dim_fornecedor CASCADE;
DROP TABLE IF EXISTS dim_unidade CASCADE;

-- ------------------------------------------------------------------------------
-- 1. TABELA DIMENSÃO: DIM_UNIDADE
-- Registra as Secretarias, Órgãos e Unidades Gestoras do Estado.
-- ------------------------------------------------------------------------------
CREATE TABLE dim_unidade (
    id_unidade SERIAL PRIMARY KEY,
    nome_unidade VARCHAR(255) NOT NULL UNIQUE
);

COMMENT ON TABLE dim_unidade IS 'Dimensão de Unidades Gestoras e Secretarias contratantes do Estado do RJ';
COMMENT ON COLUMN dim_unidade.nome_unidade IS 'Nome completo da unidade gestora (ex: SEEDUC, FSERJ, UERJ)';

-- ------------------------------------------------------------------------------
-- 2. TABELA DIMENSÃO: DIM_FORNECEDOR
-- Registra as empresas, institutos e prestadores de serviço vencedores.
-- ------------------------------------------------------------------------------
CREATE TABLE dim_fornecedor (
    id_fornecedor SERIAL PRIMARY KEY,
    nome_fornecedor VARCHAR(255) NOT NULL UNIQUE
);

COMMENT ON TABLE dim_fornecedor IS 'Dimensão de Fornecedores e empresas contratadas';
COMMENT ON COLUMN dim_fornecedor.nome_fornecedor IS 'Razão social / Nome do fornecedor vencedor da compra direta';

-- ------------------------------------------------------------------------------
-- 3. TABELA DIMENSÃO: DIM_AFASTAMENTO
-- Registra os enquadramentos legais e fundamentações para dispensa/inexigibilidade.
-- ------------------------------------------------------------------------------
CREATE TABLE dim_afastamento (
    id_afastamento SERIAL PRIMARY KEY,
    tipo_afastamento VARCHAR(255) NOT NULL,
    enquadramento_legal TEXT NOT NULL,
    CONSTRAINT unq_afastamento_enquadramento UNIQUE (tipo_afastamento, enquadramento_legal)
);

COMMENT ON TABLE dim_afastamento IS 'Dimensão de Modalidades de Afastamento de Licitação e Legislação aplicada';
COMMENT ON COLUMN dim_afastamento.tipo_afastamento IS 'Tipo de afastamento (ex: Dispensa - Pequenas Compras, Inexigibilidade)';
COMMENT ON COLUMN dim_afastamento.enquadramento_legal IS 'Artigo e lei de fundamentação (ex: Lei nº 14.133/2021, Art. 74º, I)';

-- ------------------------------------------------------------------------------
-- 4. TABELA DIMENSÃO: DIM_PROCESSO
-- Registra o processo administrativo único e seus metadados agregados.
-- ------------------------------------------------------------------------------
CREATE TABLE dim_processo (
    id_processo SERIAL PRIMARY KEY,
    numero_processo VARCHAR(100) NOT NULL UNIQUE,
    ano_processo INT NOT NULL CHECK (ano_processo >= 2000 AND ano_processo <= 2100),
    data_aprovacao DATE NOT NULL,
    objeto TEXT NOT NULL DEFAULT 'Não informado',
    valor_total_processo NUMERIC(15, 2) NOT NULL CHECK (valor_total_processo >= 0),
    id_unidade INT NOT NULL REFERENCES dim_unidade(id_unidade),
    id_fornecedor INT NOT NULL REFERENCES dim_fornecedor(id_fornecedor),
    id_afastamento INT NOT NULL REFERENCES dim_afastamento(id_afastamento)
);

COMMENT ON TABLE dim_processo IS 'Dimensão de Processos Administrativos de Compras Diretas';
COMMENT ON COLUMN dim_processo.numero_processo IS 'Identificador do processo administrativo (ex: SEI-120001/009348/2022)';
COMMENT ON COLUMN dim_processo.valor_total_processo IS 'Valor global homologado para o processo';

-- ------------------------------------------------------------------------------
-- 5. TABELA FATO: FATO_ITEM_COMPRA
-- Tabela de fatos contendo os itens fracionados de cada processo e métricas de anomalia.
-- ------------------------------------------------------------------------------
CREATE TABLE fato_item_compra (
    id_item_compra BIGSERIAL PRIMARY KEY,
    id_processo INT NOT NULL REFERENCES dim_processo(id_processo) ON DELETE CASCADE,
    id_unidade INT NOT NULL REFERENCES dim_unidade(id_unidade),
    id_fornecedor INT NOT NULL REFERENCES dim_fornecedor(id_fornecedor),
    id_afastamento INT NOT NULL REFERENCES dim_afastamento(id_afastamento),
    
    -- Atributos do Item
    descricao_item TEXT NOT NULL,
    unidade_medida VARCHAR(50) NOT NULL DEFAULT 'UN',
    quantidade NUMERIC(15, 4) NOT NULL CHECK (quantidade >= 0),
    valor_unitario NUMERIC(15, 2) NOT NULL CHECK (valor_unitario >= 0),
    valor_calculado_item NUMERIC(15, 2) GENERATED ALWAYS AS (quantidade * valor_unitario) STORED,
    
    -- Colunas de Machine Learning Não Supervisionado & Análise de Risco
    zscore_preco_item NUMERIC(10, 4) DEFAULT NULL,
    flag_anomalia_isolation_forest SMALLINT DEFAULT 0 CHECK (flag_anomalia_isolation_forest IN (-1, 0, 1)),
    cluster_dbscan INT DEFAULT NULL,
    score_suspeicao_consolidado NUMERIC(5, 2) DEFAULT 0.00 CHECK (score_suspeicao_consolidado BETWEEN 0.00 AND 100.00)
);

COMMENT ON TABLE fato_item_compra IS 'Tabela fato dos itens de compra direta com flags de ML não supervisionado';
COMMENT ON COLUMN fato_item_compra.zscore_preco_item IS 'Desvio padrão do preço unitário em relação à média histórica do item';
COMMENT ON COLUMN fato_item_compra.flag_anomalia_isolation_forest IS '-1 para anomalia detectada pelo Isolation Forest, 1 para padrão normal';
COMMENT ON COLUMN fato_item_compra.score_suspeicao_consolidado IS 'Pontuação de risco de 0 a 100 baseada em consenso não supervisionado';

-- ------------------------------------------------------------------------------
-- ÍNDICES PARA OTIMIZAÇÃO DE CONSULTAS E CONSULTAS NO POWER BI
-- ------------------------------------------------------------------------------
CREATE INDEX idx_fato_processo ON fato_item_compra(id_processo);
CREATE INDEX idx_fato_unidade ON fato_item_compra(id_unidade);
CREATE INDEX idx_fato_fornecedor ON fato_item_compra(id_fornecedor);
CREATE INDEX idx_fato_anomalia ON fato_item_compra(flag_anomalia_isolation_forest);
CREATE INDEX idx_fato_score_suspeicao ON fato_item_compra(score_suspeicao_consolidado DESC);

CREATE INDEX idx_processo_data ON dim_processo(data_aprovacao);
CREATE INDEX idx_processo_ano ON dim_processo(ano_processo);
CREATE INDEX idx_processo_numero ON dim_processo(numero_processo);

-- ------------------------------------------------------------------------------
-- VIEW AUXILIAR DE AUDITORIA: PROCESSOS ANÔMALOS
-- Visão consolidada para facilitar consumo direto no Power BI ou relatórios SQL.
-- ------------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_compras_anomalas AS
SELECT 
    p.numero_processo,
    p.ano_processo,
    p.data_aprovacao,
    u.nome_unidade,
    f.nome_fornecedor,
    a.tipo_afastamento,
    fi.descricao_item,
    fi.quantidade,
    fi.unidade_medida,
    fi.valor_unitario,
    fi.valor_calculado_item,
    p.valor_total_processo,
    fi.zscore_preco_item,
    fi.flag_anomalia_isolation_forest,
    fi.score_suspeicao_consolidado
FROM fato_item_compra fi
JOIN dim_processo p ON fi.id_processo = p.id_processo
JOIN dim_unidade u ON fi.id_unidade = u.id_unidade
JOIN dim_fornecedor f ON fi.id_fornecedor = f.id_fornecedor
JOIN dim_afastamento a ON fi.id_afastamento = a.id_afastamento
WHERE fi.flag_anomalia_isolation_forest = -1 OR fi.score_suspeicao_consolidado >= 50.00;

COMMENT ON VIEW vw_compras_anomalas IS 'Visão de conveniência para auditoria de contratos com alta probabilidade de anomalia';
