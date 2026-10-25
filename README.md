# PIBIC 2025 — geração e avaliação de laudos radiológicos

Repositório com os dados de um projeto de pesquisa, envolvendo a preparação de dados de radiografias de tórax, recuperação de laudos semelhantes e comparação de textos gerados pelos modelos MedGemma e Moondream com laudos de referência.

## Organização

```text
.
├── data-pre-processing/       # Notebook para preparar metadados, imagens e laudos
├── generate-prompts/          # Template e geração de prompts de avaliação
├── IA-models-generation/      # Geração de laudos com MedGemma e Moondream
├── RAG-methods/               # Amostragem cardíaca, recuperação e notebooks RAG
├── evaluative-metrics/        # Métricas automáticas e resultados agregados
├── resources/                  # Dados de entrada e saídas intermediárias/geradas
├── results/logs/              # Registros de execução guardados no repositório
└── utils/                      # Conversão de JSONL para JSON
```

Os scripts usam caminhos relativos ao diretório de execução. Portanto, execute cada script a partir da pasta indicada:

| Etapa | Comando (a partir da raiz do repositório) |
|---|---|
| Criar a amostra cardíaca | `cd RAG-methods && python create_cardiac_subsample.py` |
| Gerar laudos com MedGemma | `cd IA-models-generation && python medgemma_rad_evaluate.py` |
| Gerar laudos com Moondream | `cd IA-models-generation && python moondream_rad_evaluate.py` |
| Converter resultados JSONL em JSON | `cd utils && python convert_jsonl_to_json.py` |
| Calcular métricas por exame e resumo | `cd evaluative-metrics && python quantitative_metrics.py` |
| Preencher um prompt para avaliação manual | `cd generate-prompts && python fill_evaluation_prompt.py --model moondream --row 0` |

Os notebooks de `data-pre-processing/` e `RAG-methods/` devem ser abertos com a raiz do repositório como diretório de trabalho. A preparação de dados grava `all_data.csv`, `train_data.csv` e `test_data.csv` em `resources/`, que é onde os scripts posteriores esperam encontrá-los.

## Ambiente

Use Python 3.10 ou superior e crie um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Os pesos dos modelos e alguns recursos do NLTK são baixados na primeira execução. Para baixar MedGemma, aceite os termos de acesso na [página do modelo](https://huggingface.co/google/medgemma-4b-pt) e autentique-se no Hugging Face com `hf auth login` (o CLI é fornecido por `huggingface_hub`). A instalação do PyTorch deve corresponder ao sistema e ao hardware disponíveis; use o [seletor oficial de instalação do PyTorch](https://pytorch.org/get-started/locally/) para escolher entre CPU e CUDA.

## Dados

O notebook `data-pre-processing/dataset-preparation.ipynb` espera o conjunto MIMIC-CXR localmente em `mimic-cxr-dataset/`, incluindo `metadata.csv`, as imagens e os relatórios. Esse conjunto não está incluído neste repositório. É necessário obter acesso e preparar os arquivos conforme a licença e as condições de uso do MIMIC-CXR.

Os arquivos `resources/train_data.csv`, `resources/test_data.csv` e `resources/all_data.csv` são dados intermediários. `create_cardiac_subsample.py` combina treino e teste e seleciona até 150 exemplos a partir dos termos em `resources/common-findings.json`. Os arquivos `*_generated_reports.jsonl` são saídas incrementais da geração; os arquivos JSON correspondentes são versões convertidas usadas pela avaliação. A geração pode ser demorada e requer memória de sistema ou GPU adequada ao modelo.

## Fluxo sugerido

1. Prepare os dados com o notebook de pré-processamento, depois de disponibilizar MIMIC-CXR localmente.
2. Gere a amostra de relatórios cardíacos com `create_cardiac_subsample.py`.
3. Gere os laudos para a amostra com um ou ambos os scripts em `IA-models-generation/`.
4. Converta os resultados `.jsonl` em `.json` com `utils/convert_jsonl_to_json.py`.
5. Calcule as métricas com `evaluative-metrics/quantitative_metrics.py` ou use `quantitative_metrics_verbose.py` para o resumo agregado no terminal.
6. Use `generate-prompts/fill_evaluation_prompt.py` para montar prompts de revisão manual.

Os notebooks e scripts de `RAG-methods/` exploram recuperação semântica e avaliação de vizinhos. `evaluate_rag_retrievers.py` e `semantic_text_comparator.py` devem ser executados a partir de `RAG-methods/`.

## Observações

- Os resultados existentes em `resources/` e `evaluative-metrics/` são artefatos já gerados e podem não corresponder a uma execução limpa ou completa.
- A geração de MedGemma retoma o processamento a partir dos `study_id` já presentes no JSONL. A geração de Moondream acrescenta registros ao arquivo em cada execução; remova ou arquive a saída anterior antes de começar uma nova rodada completa.
- Os valores reportados pelas métricas são medidas automáticas de similaridade textual e semântica; não substituem avaliação clínica.
