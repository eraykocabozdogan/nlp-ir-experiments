# NLP & IR Experiments: from RNNs to RAG

Five controlled experiments that trace the path from recurrent models to
retrieval-augmented generation, implemented in PyTorch and Hugging Face. Each one has
a notebook with saved outputs and is written up in the accompanying
[8-page report](Latex.pdf).

> Originally done for CENG543 Information Retrieval (IZTECH, Fall 2025).

## Experiments and results

| # | Experiment | Data | Key result |
|---|---|---|---|
| 1 | BiLSTM / BiGRU × static (GloVe) vs contextual (DistilBERT) embeddings | SST-2 | Contextual embeddings: 0.839 → **0.885** accuracy (BiLSTM). t-SNE shows clearer class separation |
| 2 | Additive (Bahdanau), multiplicative (Luong) and dot-product attention in Seq2Seq | Multi30k | Additive attention is best (BLEU **21.98**, ROUGE-L 0.533). Alignment maps included |
| 3 | Transformer vs Seq2Seq, with a layer/head ablation | Multi30k | 3 layers × 4 heads gives BLEU **25.14** vs 21.88 for the base model. 1 layer drops to 13.87 |
| 4 | RAG with BM25 retrieval + FLAN-T5 generation | SQuAD | Recall@1 0.515, BLEU 0.228, ROUGE-L 0.452, BERTScore **0.753**. Includes a hallucination vs faithfulness analysis |
| 5 | Interpretability and error analysis of the Transformer from #3 | Multi30k | Per-token prediction entropy highlights uncertain tokens; sentence-level BLEU separates success and failure cases |

## Layout

```
Notebook/   Ceng543_q1–q5.ipynb   one notebook per experiment, outputs saved
Outputs/    figures and CSVs (convergence, t-SNE, attention maps, ablation)
Latex.pdf   full report
```

## Run

Developed on Google Colab with a T4 GPU.

```bash
pip install -r requirements.txt   # or: conda env create -f environment.yml
jupyter notebook Notebook/
```
