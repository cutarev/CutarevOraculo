# 🔮 Oráculo

> Assistente de IA com superpoderes — analise CSVs, gere gráficos, pesquise na web e entenda vídeos do YouTube, tudo em um chat.

---

<img width="1869" height="922" alt="{86AD0236-1483-42F4-AD9C-5D69C062CB68}" src="https://github.com/user-attachments/assets/163dbdb9-8ac6-49c2-a207-c4099b74334f" />

## ✨ Funcionalidades

- 📊 **Análise de CSV e XLSX** — faça perguntas em linguagem natural sobre seus dados
- 📈 **Geração de gráficos** — bar, line, scatter, pie, histogram e box plot direto no chat
- 🔍 **Busca na web** — pesquisa via DuckDuckGo sem precisar de URL
- 🌐 **Web scraping** — cole qualquer URL e peça um resumo
- 🎬 **Transcrição do YouTube** — analise e faça perguntas sobre qualquer vídeo
- 💡 **Sugestões inteligentes** — ao carregar um arquivo, o Oráculo sugere perguntas automaticamente
- ⚡ **Streaming de respostas** — respostas em tempo real, palavra por palavra
- 🤖 **Multi-provedor** — suporte a Groq e OpenAI

---

## 🚀 Como usar

### 1. Clone o repositório

```bash
git clone https://github.com/cutarev/CutarevOraculo.git
cd CutarevOraculo
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure as variáveis de ambiente

Copie o arquivo de exemplo e adicione suas chaves:

```bash
cp .env.example .env
```
(Windows)

```CMD (Windows)
copy .env.example .env
```

Edite o `.env`:

```env
GROQ_API_KEY=sua_chave_aqui
OPENAI_API_KEY=sua_chave_aqui  # opcional
```

> Você também pode inserir a chave diretamente na interface sem precisar do `.env`.

### 4. Rode o app

```bash
streamlit run oraculo_page.py
```
(Windows)

```CMD (Windows)
py -m streamlit run oraculo_page.py
```

---

## 🛠️ Tecnologias

| Tecnologia | Uso |
|---|---|
| [Streamlit](https://streamlit.io) | Interface web |
| [LangChain](https://langchain.com) | Orquestração do agente |
| [Groq](https://groq.com) | LLM ultrarrápido (padrão) |
| [OpenAI](https://openai.com) | LLM alternativo |
| [Pandas](https://pandas.pydata.org) | Análise de dados |
| [Matplotlib](https://matplotlib.org) | Geração de gráficos |
| [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) | Busca na web |
| [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) | Web scraping |
| [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/) | Transcrição de vídeos |

---

## 📁 Estrutura do projeto

```
projeto-oraculo/
├── oraculo_page.py      # App principal
├── requirements.txt     # Dependências
├── .env.example         # Exemplo de variáveis de ambiente
├── .gitignore
└── README.md
```

---

## 🧠 Modelos suportados

**Groq**
- `llama-3.3-70b-versatile` ⭐ recomendado
- `llama-3.1-8b-instant`
- `gemma2-9b-it`
- `mixtral-8x7b-32768`

**OpenAI**
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-3.5-turbo`

---

## 📄 Licença

MIT — sinta-se livre para usar, modificar e distribuir.
