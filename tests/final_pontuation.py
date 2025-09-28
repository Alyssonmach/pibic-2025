import re
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================================================
# --- CONFIGURAÇÕES E PARÂMETROS ---
LIMITE_EQUIVALENCIA = 0.90
LIMITE_INVESTIGACAO = 0.65
# NOVO PARÂMETRO: Ajuste o quão severa é a punição por cada divergência. (Valores entre 0.5 e 0.9)
FATOR_PENALIDADE_DIVERGENCIA = 0.7 
# ======================================================================================


# --- Funções de Apoio ---

def preparar_nltk():
    """Verifica e baixa os pacotes necessários do NLTK."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Baixando pacotes de tokenização ('punkt' e 'punkt_tab') do NLTK...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("Downloads do NLTK concluídos.")

def extrair_sentencas(texto: str) -> list[str]:
    """Usa o NLTK para dividir o texto em sentenças."""
    texto_limpo = re.sub(r'\n+', ' ', texto).strip()
    return nltk.sent_tokenize(texto_limpo, language='portuguese')

# --- MODIFICADO PARA APLICAR PENALIDADE ---
def analise_granular(sentencas_a: list[str], sentencas_b: list[str], model) -> float:
    """
    Executa a análise sentença por sentença, imprime os detalhes
    e retorna a pontuação de similaridade média COM PENALIDADE por divergências.
    """
    print("\n" + "="*30)
    print("INICIANDO ANÁLISE GRANULAR DETALHADA")
    print("="*30)
    print("Os textos parecem ser sobre o mesmo tópico, mas podem conter divergências. Investigando...\n")

    embeddings_a = model.encode(sentencas_a)
    embeddings_b = model.encode(sentencas_b)
    
    similarity_matrix = cosine_similarity(embeddings_a, embeddings_b)
    
    lista_de_scores = []
    divergencias_encontradas = 0
    LIMITE_DIVERGENCIA_SENTENCA = 0.70 

    for i, sentenca_a in enumerate(sentencas_a):
        best_match_index = np.argmax(similarity_matrix[i])
        best_match_score = similarity_matrix[i][best_match_index]
        sentenca_b_correspondente = sentencas_b[best_match_index]
        
        lista_de_scores.append(best_match_score)

        print(f"SENTENÇA ORIGINAL: \"{sentenca_a}\"")
        print(f"MELHOR CORRESPONDÊNCIA: \"{sentenca_b_correspondente}\"")
        print(f"--> Similaridade da Sentença: {best_match_score:.2%}\n")

        if best_match_score < LIMITE_DIVERGENCIA_SENTENCA:
            divergencias_encontradas += 1
            print(f"*** ALERTA DE DIVERGÊNCIA! *** A similaridade está abaixo de {LIMITE_DIVERGENCIA_SENTENCA:.0%}.\n")
            print("-" * 50 + "\n")

    pontuacao_media = np.mean(lista_de_scores)
    
    # --- LÓGICA DE PENALIDADE ---
    pontuacao_final_penalizada = pontuacao_media * (FATOR_PENALIDADE_DIVERGENCIA ** divergencias_encontradas)
    
    print(f"CONCLUSÃO GRANULAR: Foram encontradas {divergencias_encontradas} divergências significativas.")
    print(f"-> Pontuação Média (sem penalidade): {pontuacao_media:.2%}")
    print(f"-> Pontuação Final (com penalidade): {pontuacao_final_penalizada:.2%}")

    return pontuacao_final_penalizada


# --- SCRIPT PRINCIPAL ---

if __name__ == "__main__":
    preparar_nltk()

    # Cenário 1: Textos diferentes, mesmo significado (Esperado: ALTA SIMILARIDADE)
    texto_1 = """Mamografia evidencia nódulo espiculado no quadrante superior externo da mama esquerda, medindo 1,5 cm em seu maior eixo. Observam-se microcalcificações pleomórficas agrupadas associadas à lesão. Impressão: Achado mamográfico altamente suspeito de malignidade (BI-RADS 5). Recomenda-se correlação histopatológica."""
    texto_2 = """Foi identificada uma massa de contornos irregulares e espiculados na mama esquerda, localizada na união dos quadrantes superiores. A lesão principal tem dimensão de 1,5 cm e está acompanhada por um agrupamento de microcalcificações. Conclusão: O aspecto da lesão é de alta suspeição para câncer, sendo classificada como BI-RADS 5. É sugerida a realização de biópsia para confirmação."""

    # Cenário 2: Textos parecidos, significados opostos
    # texto_1 = """Fígado com dimensões normais, contornos regulares, apresentando lesão focal hipodensa no segmento IV, com realce periférico, suspeita para metástase. Presença de múltiplas adenomegalias retroperitoneais, a maior medindo 1.8 cm. Pâncreas e rins sem alterações. Conclusão: Achados compatíveis com doença secundária."""
    # texto_2 = """Fígado com dimensões normais e contornos regulares, sem evidências de lesão focal. O parênquima realça de forma homogênea. Ausência de adenomegalias retroperitoneais. Pâncreas e rins de aspecto preservado. Conclusão: Exame dentro dos limites da normalidade, sem sinais de doença ativa."""
    
    print("Carregando modelo de linguagem... (pode demorar na primeira execução)")
    model = SentenceTransformer('stsb-xlm-r-multilingual')

    print("Realizando análise global de triagem...")
    embeddings_global = model.encode([texto_1, texto_2])
    similaridade_global = cosine_similarity([embeddings_global[0]], [embeddings_global[1]])[0][0]

    print("\n--- RESULTADO DA ANÁLISE ---")
    print(f"Similaridade Global (Triagem Inicial): {similaridade_global:.2%}\n")

    if similaridade_global >= LIMITE_EQUIVALENCIA:
        print(f"PONTUAÇÃO FINAL: {similaridade_global:.2%}")
        print("CONCLUSÃO: Alta similaridade. Os textos são considerados semanticamente equivalentes. ✅")
    
    elif similaridade_global < LIMITE_INVESTIGACAO:
        print(f"PONTUAÇÃO FINAL: {similaridade_global:.2%}")
        print("CONCLUSÃO: Baixa similaridade. Os textos são muito diferentes, contraditórios ou sobre temas distintos. ❌")
    
    else:
        sentencas_1 = extrair_sentencas(texto_1)
        sentencas_2 = extrair_sentencas(texto_2)
        
        pontuacao_final = analise_granular(sentencas_1, sentencas_2, model)
        
        print("\n" + "#"*50)
        print("              RESULTADO FINAL CONSOLIDADO")
        print("#"*50)
        print(f"\nA similaridade de triagem foi de {similaridade_global:.2%}, ativando a análise detalhada.")
        print(f"A PONTUAÇÃO FINAL (com Penalidade por Divergência) é de: {pontuacao_final:.2%}")
        print("#"*50)