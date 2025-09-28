import re
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================================================
# --- CONFIGURAÇÕES E PARÂMETROS ---
# Altere estes valores para ajustar a sensibilidade do sistema
LIMITE_EQUIVALENCIA = 0.90  # Acima deste valor, os textos são considerados idênticos em significado.
LIMITE_INVESTIGACAO = 0.65  # Abaixo deste valor, os textos são considerados muito diferentes.

# O "meio-termo" entre os dois limites ativa a análise detalhada.
# ======================================================================================


# --- Funções de Apoio ---

def preparar_nltk():
    """Verifica e baixa os pacotes necessários do NLTK."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except:
        print("Baixando pacotes de tokenização ('punkt' e 'punkt_tab') do NLTK...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("Downloads do NLTK concluídos.")

def extrair_sentencas(texto: str) -> list[str]:
    """Usa o NLTK para dividir o texto em sentenças."""
    texto_limpo = re.sub(r'\n+', ' ', texto).strip()
    return nltk.sent_tokenize(texto_limpo, language='portuguese')

def analise_granular(sentencas_a: list[str], sentencas_b: list[str], model):
    """Executa e imprime a análise comparativa sentença por sentença."""
    print("\n" + "="*30)
    print("INICIANDO ANÁLISE GRANULAR DETALHADA")
    print("="*30)
    print("Os textos parecem ser sobre o mesmo tópico, mas podem conter divergências. Investigando...\n")

    embeddings_a = model.encode(sentencas_a)
    embeddings_b = model.encode(sentencas_b)
    
    similarity_matrix = cosine_similarity(embeddings_a, embeddings_b)
    
    divergencias_encontradas = 0
    LIMITE_DIVERGENCIA_SENTENCA = 0.70 

    for i, sentenca_a in enumerate(sentencas_a):
        best_match_index = np.argmax(similarity_matrix[i])
        best_match_score = similarity_matrix[i][best_match_index]
        sentenca_b_correspondente = sentencas_b[best_match_index]

        print(f"SENTENÇA ORIGINAL: \"{sentenca_a}\"")
        print(f"MELHOR CORRESPONDÊNCIA: \"{sentenca_b_correspondente}\"")
        print(f"--> Similaridade da Sentença: {best_match_score:.2%}\n")

        if best_match_score < LIMITE_DIVERGENCIA_SENTENCA:
            divergencias_encontradas += 1
            print(f"*** ALERTA DE DIVERGÊNCIA! *** A similaridade está abaixo de {LIMITE_DIVERGENCIA_SENTENCA:.0%}.\n")
            print("-" * 50 + "\n")

    if divergencias_encontradas > 0:
        print(f"CONCLUSÃO GRANULAR: Foram encontradas {divergencias_encontradas} divergências significativas entre os textos.")
    else:
        print("CONCLUSÃO GRANULAR: Apesar de não serem paráfrases exatas, não foram encontradas divergências clínicas significativas.")


# --- SCRIPT PRINCIPAL ---

if __name__ == "__main__":
    preparar_nltk()

    # --- INICIO DA ÁREA DE TESTE ---
    # Cole seus textos aqui. Descomente os pares para testar cada cenário.

    # Cenário 1: Textos diferentes, mesmo significado (Esperado: ALTA SIMILARIDADE)
    texto_1 = """Mamografia evidencia nódulo espiculado no quadrante superior externo da mama esquerda, medindo 1,5 cm em seu maior eixo. Observam-se microcalcificações pleomórficas agrupadas associadas à lesão. Impressão: Achado mamográfico altamente suspeito de malignidade (BI-RADS 5). Recomenda-se correlação histopatológica."""
    texto_2 = """Foi identificada uma massa de contornos irregulares e espiculados na mama esquerda, localizada na união dos quadrantes superiores. A lesão principal tem dimensão de 1,5 cm e está acompanhada por um agrupamento de microcalcificações. Conclusão: O aspecto da lesão é de alta suspeição para câncer, sendo classificada como BI-RADS 5. É sugerida a realização de biópsia para confirmação."""

    # Cenário 2: Textos parecidos, significados opostos (Esperado: ANÁLISE GRANULAR com alertas)
    # texto_1 = """Fígado com dimensões normais, contornos regulares, apresentando lesão focal hipodensa no segmento IV, com realce periférico, suspeita para metástase. Presença de múltiplas adenomegalias retroperitoneais, a maior medindo 1.8 cm. Pâncreas e rins sem alterações. Conclusão: Achados compatíveis com doença secundária."""
    # texto_2 = """Fígado com dimensões normais e contornos regulares, sem evidências de lesão focal. O parênquima realça de forma homogênea. Ausência de adenomegalias retroperitoneais. Pâncreas e rins de aspecto preservado. Conclusão: Exame dentro dos limites da normalidade, sem sinais de doença ativa."""
    
    # --- FIM DA ÁREA DE TESTE ---

    print("Carregando modelo de linguagem... (pode demorar na primeira execução)")
    model = SentenceTransformer('stsb-xlm-r-multilingual')

    # 1. ANÁLISE GLOBAL (TRIAGEM)
    print("Realizando análise global...")
    embeddings_global = model.encode([texto_1, texto_2])
    similaridade_global = cosine_similarity([embeddings_global[0]], [embeddings_global[1]])[0][0]

    print("\n--- RESULTADO DA ANÁLISE ---")
    print(f"Similaridade Global: {similaridade_global:.2%}\n")

    # 2. DECISÃO INTELIGENTE
    if similaridade_global >= LIMITE_EQUIVALENCIA:
        print("CONCLUSÃO: Alta similaridade. Os textos são considerados semanticamente equivalentes. ✅")
    
    elif similaridade_global < LIMITE_INVESTIGACAO:
        print("CONCLUSÃO: Baixa similaridade. Os textos são muito diferentes, contraditórios ou sobre temas distintos. ❌")
    
    else: # Zona Cinzenta: Ativar análise granular
        sentencas_1 = extrair_sentencas(texto_1)
        sentencas_2 = extrair_sentencas(texto_2)
        analise_granular(sentencas_1, sentencas_2, model)