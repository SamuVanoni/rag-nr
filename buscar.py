# -*- coding: utf-8 -*-
"""
ETAPA 2a do RAG: achar os trechos certos. SEM IA NENHUMA.

Este arquivo existe separado de proposito. RAG tem duas metades:

    1. ACHAR o trecho certo   <- este arquivo
    2. ESCREVER a resposta    <- responder.py

Se a resposta final sair ruim, voce precisa saber QUAL das duas falhou. Rodando
este aqui sozinho voce ve exatamente o que a busca devolveu, sem modelo nenhum no
meio para inventar coisa. E de graca: nao gasta cota de API.

Uso:
    python buscar.py "de quanto em quanto tempo refaco o treinamento de altura?"
"""

import json
import pathlib
import sys

import numpy as np
from fastembed import TextEmbedding

AQUI = pathlib.Path(__file__).parent
INDICE_VETORES = AQUI / "indice.npy"
INDICE_TRECHOS = AQUI / "indice.json"

_cache = {}


def carregar():
    """Le o indice do disco uma vez e guarda na memoria."""
    if not _cache:
        if not INDICE_VETORES.exists():
            sys.exit("Indice nao encontrado. Rode antes:  python ingest.py")
        dados = json.loads(INDICE_TRECHOS.read_text(encoding="utf-8"))
        _cache["vetores"] = np.load(INDICE_VETORES)
        _cache["trechos"] = dados["trechos"]
        # O MESMO modelo da ingestao, obrigatoriamente. Numeros gerados por modelos
        # diferentes nao sao comparaveis -- a busca nao daria erro, daria resultado
        # ruim em silencio, que e pior.
        _cache["modelo"] = TextEmbedding(model_name=dados["modelo"])
    return _cache


def buscar(pergunta: str, quantos: int = 5):
    """Devolve [(nota_de_parecido, trecho), ...] do mais parecido para o menos."""
    c = carregar()

    # A pergunta vira numeros pelo mesmo caminho que os trechos viraram.
    v_pergunta = np.array(list(c["modelo"].embed([pergunta]))[0], dtype=np.float32)
    v_pergunta /= np.linalg.norm(v_pergunta)

    # Similaridade de cosseno. Como todos os vetores ja tem tamanho 1, a conta inteira
    # e uma multiplicacao de matrizes: 690 comparacoes de uma vez, em milissegundos.
    # Nota perto de 1 = falando da mesma coisa. Perto de 0 = nada a ver.
    notas = c["vetores"] @ v_pergunta

    melhores = np.argsort(-notas)[:quantos]
    return [(float(notas[i]), c["trechos"][i]) for i in melhores]


def rotulo(t: dict) -> str:
    """Como este trecho deve ser citado: 'NR-35, item 35.4.2.2'."""
    base = f"{t['norma']}, item {t['item']}"
    return f"{base} (parte {t['parte']})" if t.get("parte") else base


def main():
    if len(sys.argv) < 2:
        sys.exit('Uso: python buscar.py "sua pergunta aqui"')
    pergunta = " ".join(sys.argv[1:])

    print(f'Pergunta: "{pergunta}"\n')
    for nota, t in buscar(pergunta):
        print(f"[{nota:.3f}]  {rotulo(t)}  (pagina {t['pagina']})")
        print(f"          {t['texto'][:300]}")
        print()


if __name__ == "__main__":
    main()
