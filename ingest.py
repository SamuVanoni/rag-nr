# -*- coding: utf-8 -*-
"""
ETAPA 1 de 2 do RAG: preparar o indice. Roda uma vez (ou quando um PDF mudar).

O que acontece aqui, em ordem:

    PDF  ->  texto  ->  pedacos  ->  numeros  ->  arquivo de indice

O "numeros" e a parte que costuma assustar. Um embedding e so uma lista de numeros
que representa o SENTIDO de um texto. Dois textos que falam da mesma coisa viram
listas parecidas, mesmo sem repetir as mesmas palavras. E por isso que a busca
acha "de quanto em quanto tempo refaco o treinamento" dentro de um item que diz
"o treinamento periodico deve ser realizado a cada dois anos" -- nenhuma palavra
em comum.
"""

import json
import pathlib
import re
import sys

import numpy as np
from fastembed import TextEmbedding
from pypdf import PdfReader

PASTA_DADOS = pathlib.Path(__file__).parent / "dados"
INDICE_VETORES = pathlib.Path(__file__).parent / "indice.npy"
INDICE_TRECHOS = pathlib.Path(__file__).parent / "indice.json"

# O modelo que traduz texto em numeros. Fica salvo no seu PC depois da primeira vez.
# Multilingue de proposito: os modelos so-ingles erram feio em portugues juridico.
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Arquivos que NAO entram no indice, e o motivo de cada um. Esta lista e uma decisao
# de produto, nao uma limitacao tecnica -- por isso mora aqui, visivel, e nao escondida
# num filtro qualquer.
FORA_DO_INDICE = {
    "nr-01-atualizada-2024-i-1.pdf":
        "versao de 2024, substituida pela de 2025. Se as duas entrassem, a busca acharia "
        "trecho nas duas e poderia responder a regra VELHA citando a fonte certa -- um erro "
        "que ninguem percebe lendo a resposta.",
    "manual_gro_pgr_da_nr_1.pdf":
        "e manual de orientacao, nao e a norma. Misturar os dois faz a resposta perder a "
        "diferenca entre 'a lei obriga' e 'o ministerio sugere'.",
}

# Linhas de cabecalho/rodape que se repetem em toda pagina e nao dizem nada.
LIXO = [
    re.compile(r"^Este texto n[ãa]o substitui o publicado no DOU\s*$", re.I),
    re.compile(r"^\s*\d{1,3}\s*$"),           # numero de pagina sozinho
    re.compile(r"^\s*$"),
]

# Um item de norma comeca com o numero dele no inicio da linha: 35.4.2.2, 32.2.4, 1.5 ...
INICIO_DE_ITEM = re.compile(r"^(\d{1,2}(?:\.\d{1,3})+)\s+(.*)$")

# Titulo de capitulo tambem comeca com numero, mas e curto e sem ponto final.
TAMANHO_MAXIMO = 1200   # caracteres por pedaco; item maior que isso e quebrado


def nome_da_norma(arquivo: str) -> str:
    """De 'nr-35-atualizada-2025-1.pdf' tira 'NR-35'."""
    m = re.match(r"nr-?0*(\d+)", arquivo, re.I)
    return f"NR-{int(m.group(1)):02d}" if m else arquivo


def ler_texto(caminho: pathlib.Path) -> list[tuple[int, str]]:
    """Devolve [(numero_da_pagina, linha), ...] ja sem o lixo repetido."""
    linhas = []
    for n_pagina, pagina in enumerate(PdfReader(str(caminho)).pages, start=1):
        for linha in (pagina.extract_text() or "").split("\n"):
            linha = linha.strip()
            if any(p.match(linha) for p in LIXO):
                continue
            linhas.append((n_pagina, linha))
    return linhas


def quebrar_em_itens(linhas, norma):
    """
    Agrupa as linhas por item da norma.

    A regra: uma linha que comeca com um numero tipo 35.4.2 abre um item novo.
    Tudo que vier depois (inclusive as alineas a), b), c)) pertence a ele, ate o
    proximo numero aparecer.

    Cortar assim, e nao a cada X caracteres, e o que permite a resposta dizer
    "NR-35, item 35.4.2.2" em vez de "trecho 7 do arquivo". E e tambem o jeito como
    a norma e citada na vida real.
    """
    itens, atual = [], None

    for n_pagina, linha in linhas:
        m = INICIO_DE_ITEM.match(linha)
        if m:
            if atual:
                itens.append(atual)
            atual = {
                "norma": norma,
                "item": m.group(1),
                "texto": m.group(2).strip(),
                "pagina": n_pagina,
            }
        elif atual:
            atual["texto"] += " " + linha
        # linha antes do primeiro item (capa, ementa) e descartada de proposito

    if atual:
        itens.append(atual)

    # Item gigante vira varios pedacos, mas todos guardam o mesmo numero de item.
    saida = []
    for it in itens:
        texto = " ".join(it["texto"].split())
        if len(texto) < 25:          # item que e so titulo de capitulo
            continue
        if len(texto) <= TAMANHO_MAXIMO:
            saida.append({**it, "texto": texto, "parte": None})
            continue
        pedacos = [texto[i:i + TAMANHO_MAXIMO] for i in range(0, len(texto), TAMANHO_MAXIMO)]
        for i, p in enumerate(pedacos, start=1):
            saida.append({**it, "texto": p, "parte": f"{i}/{len(pedacos)}"})
    return saida


def main():
    if not PASTA_DADOS.is_dir():
        sys.exit(f"Pasta nao encontrada: {PASTA_DADOS}")

    pdfs = sorted(PASTA_DADOS.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"Nenhum PDF em {PASTA_DADOS}")

    trechos = []
    print("Lendo os PDFs\n")
    for pdf in pdfs:
        if pdf.name in FORA_DO_INDICE:
            print(f"  [fora] {pdf.name}")
            print(f"         {FORA_DO_INDICE[pdf.name]}\n")
            continue
        norma = nome_da_norma(pdf.name)
        novos = quebrar_em_itens(ler_texto(pdf), norma)
        trechos.extend(novos)
        print(f"  [ok]   {pdf.name} -> {norma}, {len(novos)} trechos")

    if not trechos:
        sys.exit("Nenhum trecho extraido.")

    print(f"\nTotal: {len(trechos)} trechos de {len({t['norma'] for t in trechos})} normas")

    print(f"\nCarregando o modelo de embedding ({MODELO_EMBEDDING})")
    print("(na primeira vez ele baixa ~220 MB; depois fica no cache)")
    modelo = TextEmbedding(model_name=MODELO_EMBEDDING)

    print("Transformando cada trecho em numeros...")
    # Junta o nome da norma ao texto: ajuda a busca a separar NR-32 de NR-35 quando
    # a pergunta menciona a norma pelo numero.
    entradas = [f"{t['norma']} item {t['item']}: {t['texto']}" for t in trechos]
    vetores = np.array(list(modelo.embed(entradas)), dtype=np.float32)

    # Normaliza: com todos os vetores de tamanho 1, a similaridade de cosseno vira
    # uma simples multiplicacao. Menos conta na hora da busca.
    vetores /= np.linalg.norm(vetores, axis=1, keepdims=True)

    np.save(INDICE_VETORES, vetores)
    INDICE_TRECHOS.write_text(
        json.dumps({"modelo": MODELO_EMBEDDING, "trechos": trechos},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    print(f"\nPronto.")
    print(f"  {INDICE_VETORES.name}  {vetores.shape[0]} trechos x {vetores.shape[1]} numeros cada")
    print(f"  {INDICE_TRECHOS.name}  o texto de cada trecho e de onde ele veio")


if __name__ == "__main__":
    main()
