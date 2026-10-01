# rag-nr — perguntas e respostas sobre Normas Regulamentadoras

Um assistente que responde dúvidas sobre NRs brasileiras **citando o item exato da
norma**. Escrito sem framework de RAG, porque o objetivo era entender cada etapa.

```
$ python responder.py "de quanto em quanto tempo refaço o treinamento de altura?"

Trechos entregues ao modelo:
  [0.725] NR-35, item 35.4.2.2
  [0.711] NR-35, item 35.4.2.1
  [0.657] NR-01, item 1.7.1.2.3

Resposta:

O treinamento periódico de trabalho em altura deve ser realizado a cada dois anos,
com carga horária mínima de oito horas (NR-35, item 35.4.2.2).
```

## O problema que isso resolve

Perguntar sobre NR-35 direto para um modelo de IA é pedir para ele **lembrar**. Ele
responde com confiança e às vezes inventa o número do item — e num texto sobre
obrigação legal, um item errado é pior que nenhuma resposta.

RAG inverte isso: primeiro busca o trecho da norma, depois entrega o trecho ao modelo
e pede para ele **ler**. A resposta passa a ter fonte conferível.

## As duas metades, em dois arquivos

| Arquivo | O que faz | Precisa de IA? |
|---|---|---|
| `ingest.py` | PDF → texto → pedaços → números → índice | não (roda local) |
| `buscar.py` | pergunta → números → acha os trechos parecidos | **não** |
| `responder.py` | monta o prompt com os trechos e chama o modelo | sim (Groq) |

**`buscar.py` existir sozinho é a decisão de projeto mais importante aqui.** RAG falha
de duas formas diferentes: ou a busca trouxe o trecho errado, ou o modelo leu o trecho
certo e respondeu mal. São problemas opostos e consertos opostos. Rodando a busca
isolada você vê exatamente o que ela devolveu, sem modelo nenhum no meio — de graça e
sem gastar cota.

## Decisões que valem explicar

**Os pedaços seguem a numeração da norma, não um tamanho fixo.** O corte normal em RAG
é "a cada 500 caracteres". Aqui cada pedaço é **um item** (`35.4.2.2`), porque é assim
que a norma é citada na vida real. Isso é o que permite a resposta dizer
*"NR-35, item 35.4.2.2"* em vez de *"trecho 7 do arquivo"*.

**O modelo de embedding roda no seu PC, não por API.** Os números guardados no índice
só fazem sentido para o modelo que os gerou. Se fosse um modelo de API e o fornecedor
o aposentasse, o índice inteiro viraria lixo — algo que aconteceu comigo em outro
projeto, com o `gemini-2.5-flash` saindo do ar para chaves novas. Com o arquivo em
disco, ninguém aposenta nada e funciona offline.

**Busca por sentido, não por palavra.** A pergunta de exemplo usa *"de quanto em quanto
tempo"* e *"refaço"*. O item da norma diz *"periódico"* e *"a cada dois anos"*. Nenhuma
palavra em comum, e mesmo assim é o primeiro resultado. Um Ctrl+F não acharia.

**A Groq não tem modelo de embedding**, só geração de texto. Por isso o projeto usa
dois lugares diferentes: modelo local para achar, Groq para escrever.

**Se nada parecido for encontrado, o modelo nem é chamado.** Abaixo da nota 0,45 o
`responder.py` para antes da API. Mandar trechos sem relação junto com a pergunta é
convidar o modelo a inventar uma ligação que não existe.

**Duas versões da mesma norma não podem coexistir no índice.** A NR-01 de 2024 está na
pasta mas fica **fora** do índice (ver `FORA_DO_INDICE` no `ingest.py`). Com as duas, a
busca acharia trecho nas duas e poderia responder a regra revogada citando a fonte
corretamente — o erro mais perigoso possível aqui, porque não dá para perceber lendo a
resposta.

## O que este projeto NÃO faz

- **Não vale como assessoria jurídica.** Ele cita a norma; a interpretação é sua.
- **Não sabe o que não foi indexado.** São 4 normas (NR-01, NR-06, NR-32, NR-35).
- **Não acompanha atualização de norma.** Se o ministério publicar uma versão nova,
  alguém precisa trocar o PDF e rodar o `ingest.py` de novo.
- **Não trata a pergunta que precisa de duas normas ao mesmo tempo.** A busca traz os 5
  trechos mais parecidos, e se a resposta exigir cruzar NR-01 com NR-32 em pontos
  distantes, pode faltar contexto.
- **Não tem memória de conversa.** Cada pergunta começa do zero.

## Como rodar

**1. Baixe as normas** (PDF) de
[Normas Regulamentadoras Vigentes — gov.br](https://www.gov.br/trabalho-e-emprego/pt-br/acesso-a-informacao/participacao-social/conselhos-e-orgaos-colegiados/comissao-tripartite-partitaria-permanente/normas-regulamentadora/normas-regulamentadoras-vigentes)
e coloque em `dados/`. As usadas aqui: NR-01 (2025), NR-06, NR-32, NR-35.

**2. Ambiente e dependências:**

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**3. A chave da Groq** — pegue em https://console.groq.com/keys (tem tier gratuito):

```bash
cp .env.example .env     # e preencha GROQ_API_KEY
```

**4. Monte o índice** (uma vez; baixa ~220 MB de modelo na primeira execução):

```bash
python ingest.py
```

**5. Pergunte:**

```bash
python buscar.py "vacinação em clínica odontológica"      # só a busca, sem IA
python responder.py "vacinação em clínica odontológica"   # resposta completa
```

## Como foi feito

- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (384 dimensões), rodando local via `fastembed`/ONNX.
- **Busca:** similaridade de cosseno com `numpy`. Os vetores são normalizados na
  ingestão, então a busca inteira é uma multiplicação de matrizes — 690 comparações em
  milissegundos. Um banco vetorial não se justifica nessa escala.
- **Geração:** `openai/gpt-oss-120b` (modelo de pesos abertos da OpenAI) servido pela
  **Groq**, com `temperature=0` para a mesma pergunta dar sempre a mesma resposta.
- **Corpus atual:** 690 trechos de 4 normas (105 páginas).
