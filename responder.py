# -*- coding: utf-8 -*-
"""
ETAPA 2b do RAG: escrever a resposta usando os trechos achados.

O "R" de RAG e de Retrieval (recuperacao) e o "G" e de Generation (geracao). Este
arquivo e o G, e ele e burro de proposito: nao sabe nada sobre NR. Tudo o que ele
sabe esta no texto que o buscar.py entregou.

E isso e a coisa toda. Perguntar sobre NR-35 direto para um modelo e pedir para ele
lembrar. Perguntar DEPOIS de entregar o item 35.4.2.2 na mao dele e pedir para ele
LER. A segunda forma erra muito menos, e da para conferir a fonte.

Uso:
    python responder.py "de quanto em quanto tempo refaco o treinamento de altura?"
"""

import os
import pathlib
import sys

import requests
from dotenv import load_dotenv

from buscar import buscar, rotulo

load_dotenv(pathlib.Path(__file__).parent / ".env")

# A Groq nao tem modelo de embedding -- so geracao de texto. Por isso o projeto usa
# modelo local para achar e a Groq para escrever. Sao trabalhos diferentes.
MODELO = "openai/gpt-oss-120b"
URL = "https://api.groq.com/openai/v1/chat/completions"

QUANTOS_TRECHOS = 5

# A nota vai de 0 a 1. Abaixo disso o trecho achado nao tem nada a ver com a pergunta,
# e manda-lo para o modelo so aumenta a chance de ele inventar uma ligacao que nao existe.
NOTA_MINIMA = 0.45

INSTRUCAO = """Voce responde duvidas sobre Normas Regulamentadoras brasileiras.

Regras, nesta ordem de importancia:

1. Responda SOMENTE com o que estiver nos trechos fornecidos abaixo. Voce nao tem
   outra fonte.
2. Se os trechos nao responderem a pergunta, diga exatamente isso: que a resposta
   nao esta nos trechos consultados. Nao complete com conhecimento proprio, nao
   deduza e nao chute. Nao saber e uma resposta aceitavel; inventar nao e.
3. Cite a fonte de cada afirmacao no formato (NR-XX, item X.X.X). Quando cair na
   regra 2, NAO cite fonte nenhuma: citar um item junto de "nao sei" faz parecer
   que o item foi conferido e nao respondeu, o que e outra coisa.
4. Seja curto e direto. Nao repita a pergunta."""


def montar_prompt(pergunta, achados):
    """Junta os trechos num texto so, cada um com o endereco dele."""
    blocos = []
    for nota, t in achados:
        blocos.append(f"--- {rotulo(t)} ---\n{t['texto']}")
    return (
        f"{INSTRUCAO}\n\n"
        f"TRECHOS CONSULTADOS:\n\n" + "\n\n".join(blocos) + "\n\n"
        f"PERGUNTA: {pergunta}"
    )


def perguntar_ao_modelo(prompt):
    chave = os.environ.get("GROQ_API_KEY")
    if not chave:
        sys.exit("GROQ_API_KEY nao encontrada. Copie o .env.example para .env e preencha.")

    r = requests.post(
        URL,
        headers={"Authorization": f"Bearer {chave}"},
        json={
            "model": MODELO,
            # temperatura 0 = sempre a resposta mais provavel, sem variacao criativa.
            # Para texto de norma isso e o que se quer: a mesma pergunta tem que dar
            # a mesma resposta.
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=90,
    )
    if r.status_code != 200:
        sys.exit(f"Groq respondeu {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"].strip()


def main():
    if len(sys.argv) < 2:
        sys.exit('Uso: python responder.py "sua pergunta aqui"')
    pergunta = " ".join(sys.argv[1:])

    achados = buscar(pergunta, quantos=QUANTOS_TRECHOS)
    bons = [(n, t) for n, t in achados if n >= NOTA_MINIMA]

    print(f'Pergunta: "{pergunta}"\n')

    if not bons:
        # Importante: a gente barra ANTES de chamar o modelo. Se nada parecido foi
        # achado, nao existe resposta para dar -- e chamar o modelo assim mesmo seria
        # convidar ele a inventar.
        print("Nenhum trecho parecido o suficiente foi encontrado nas normas indexadas.")
        print(f"(melhor nota: {achados[0][0]:.3f}, minimo exigido: {NOTA_MINIMA})")
        return

    print("Trechos entregues ao modelo:")
    for nota, t in bons:
        print(f"  [{nota:.3f}] {rotulo(t)}")

    print("\nResposta:\n")
    print(perguntar_ao_modelo(montar_prompt(pergunta, bons)))


if __name__ == "__main__":
    main()
