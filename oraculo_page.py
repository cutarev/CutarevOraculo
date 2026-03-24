import streamlit as st
import pandas as pd
import os
import re
import json
import uuid
from dotenv import load_dotenv

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from youtube_transcript_api import YouTubeTranscriptApi
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field

load_dotenv()

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Oráculo",
    page_icon="🔮",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0e0e1a; }
    [data-testid="stSidebar"] { background-color: #14142b; border-right: 1px solid #2a2a4a; }
    [data-testid="stChatMessage"] { background-color: #1a1a2e; border-radius: 12px; margin-bottom: 8px; }
    [data-testid="stChatInput"] textarea {
        background-color: #1a1a2e !important;
        border: 1px solid #4a4a8a !important;
        color: #e0e0ff !important;
        border-radius: 12px !important;
    }
    h1 { color: #a78bfa; }
    h2, h3 { color: #c4b5fd; }
    .stCaption { color: #7c7ca8; }
    hr { border-color: #2a2a4a; }
</style>
""", unsafe_allow_html=True)

# ─── Tool helpers ──────────────────────────────────────────────────────────────

def _extract_youtube_id(url: str) -> str | None:
    patterns = [
        r'(?:v=|/v/)([^&\n?#]{11})',
        r'youtu\.be/([^&\n?#]{11})',
        r'embed/([^&\n?#]{11})',
        r'shorts/([^&\n?#]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# ─── Column type inference ────────────────────────────────────────────────────

def _infer_column_types(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """
    Retorna lista de (coluna, dtype_pandas, tipo_semantico) para cada coluna.
    Tipos semânticos detectados:
        ID único, booleano, data/hora, numérico contínuo, numérico discreto,
        numérico binário, categórico (N cat.), texto livre, data como string
    """
    rows = []
    n_total = len(df)

    for col in df.columns:
        series   = df[col].dropna()
        n_unique = series.nunique()
        dtype    = df[col].dtype

        if pd.api.types.is_bool_dtype(dtype):
            semantic = "booleano"

        elif pd.api.types.is_datetime64_any_dtype(dtype):
            semantic = "data/hora"

        elif pd.api.types.is_numeric_dtype(dtype):
            if n_unique <= 2:
                semantic = "binário (0/1)"
            elif n_unique <= 10:
                semantic = f"numérico categórico ({n_unique} valores)"
            elif pd.api.types.is_integer_dtype(dtype) and n_unique == n_total:
                semantic = "ID / chave única"
            elif pd.api.types.is_float_dtype(dtype):
                semantic = "numérico contínuo"
            else:
                semantic = "numérico discreto"

        elif pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype):
            # tenta detectar datas em strings
            try:
                pd.to_datetime(series.dropna().head(30), format="mixed", dayfirst=True)
                semantic = "data como string"
            except Exception:
                ratio = n_unique / n_total if n_total else 1
                if n_unique == n_total:
                    semantic = "texto livre / ID único"
                elif ratio < 0.05 or n_unique <= 20:
                    semantic = f"categórico ({n_unique} categorias)"
                else:
                    semantic = f"texto alta cardinalidade ({n_unique} únicos)"

        else:
            semantic = str(dtype)

        rows.append((col, str(dtype), semantic))

    return rows


# ─── Tool input schemas ────────────────────────────────────────────────────────

class QueryInput(BaseModel):
    query: str = Field(description="Pergunta ou instrução")

class UrlInput(BaseModel):
    url: str = Field(description="URL completa iniciando com https://")

class ChartInput(BaseModel):
    chart_type: str = Field(description="Tipo do gráfico: bar, line, scatter, pie, histogram, box")
    x_column: str = Field(description="Nome da coluna para o eixo X (ou fatias no pie)")
    y_column: str = Field(default="", description="Nome da coluna para o eixo Y (opcional em histogram e pie)")
    title: str = Field(default="", description="Título do gráfico")
    file_name: str = Field(default="", description="Nome do arquivo CSV; deixe vazio para usar o primeiro carregado")
    color_column: str = Field(default="", description="Coluna para diferenciar séries por cor (opcional)")


def _build_tools(dataframes: dict, enable_search: bool = False, enable_charts: bool = True) -> list:
    """Cria as ferramentas do agente, fechando sobre os dataframes atuais."""

    def analyze_csv(query: str) -> str:
        if not dataframes:
            return "Nenhum arquivo CSV foi carregado. Peça ao usuário para fazer upload de um CSV na barra lateral."

        # Limites de tamanho para decidir estratégia de envio
        SMALL_ROWS  = 300    # envia tudo
        MEDIUM_ROWS = 2000   # amostra head+random+tail
        SMALL_CHARS = 20_000 # fallback por caracteres (independente de linhas)

        parts = []
        for name, df in dataframes.items():
            n, ncols = df.shape
            col_types = _infer_column_types(df)
            type_table = "\n".join(
                f"  {col:<30} {dtype:<12} → {semantic}"
                for col, dtype, semantic in col_types
            )

            parts.append(f"### Arquivo: {name}")
            parts.append(f"- Shape: {n} linhas × {ncols} colunas")
            parts.append(f"- Tipos de coluna detectados:\n{type_table}")
            parts.append(f"- Valores nulos por coluna:\n{df.isnull().sum().to_string()}")

            full_str = df.to_string(index=False)

            if n <= SMALL_ROWS or len(full_str) <= SMALL_CHARS:
                # Arquivo pequeno — envia tudo
                parts.append(f"\nDados completos ({n} linhas):\n{full_str}")

            elif n <= MEDIUM_ROWS:
                # Arquivo médio — head + amostra aleatória + tail
                n_sample = min(100, n - 100)
                sample   = df.sample(n_sample, random_state=42)
                parts.append(f"\n[Arquivo médio: {n} linhas — enviando amostra representativa]")
                parts.append(f"Primeiras 50 linhas:\n{df.head(50).to_string(index=False)}")
                parts.append(f"Amostra aleatória ({n_sample} linhas):\n{sample.to_string(index=False)}")
                parts.append(f"Últimas 50 linhas:\n{df.tail(50).to_string(index=False)}")

            else:
                # Arquivo grande — amostra pequena + value_counts de colunas categóricas
                parts.append(f"\n[Arquivo grande: {n} linhas — enviando amostra estratificada]")
                parts.append(f"Primeiras 20 linhas:\n{df.head(20).to_string(index=False)}")
                parts.append(f"Amostra aleatória (30 linhas):\n{df.sample(30, random_state=42).to_string(index=False)}")
                parts.append(f"Últimas 20 linhas:\n{df.tail(20).to_string(index=False)}")
                cat_cols = df.select_dtypes(include=["object", "category"]).columns[:5]
                for col in cat_cols:
                    parts.append(f"\nTop valores em '{col}':\n{df[col].value_counts().head(10).to_string()}")

            # Estatísticas numéricas (sempre)
            numeric = df.select_dtypes(include="number")
            if not numeric.empty:
                parts.append(f"\nEstatísticas numéricas:\n{numeric.describe().to_string()}")

        return "\n\n".join(parts)

    def scrape_website(url: str) -> str:
        try:
            ua = UserAgent()
            headers = {"User-Agent": ua.random}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            lines = [l.strip() for l in soup.get_text(separator="\n").splitlines() if l.strip()]
            text = "\n".join(lines)
            limit = 4000
            return text[:limit] + ("\n\n[... conteúdo truncado ...]" if len(text) > limit else "")
        except Exception as e:
            return f"Erro ao fazer scraping de '{url}': {e}"

    def search_web(query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=6))
            if not results:
                return "Nenhum resultado encontrado para essa busca."
            parts = []
            for r in results:
                parts.append(f"**{r['title']}**\n{r['href']}\n{r['body']}")
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            return f"Erro ao buscar '{query}': {e}"

    def generate_chart(
        chart_type: str,
        x_column: str,
        y_column: str = "",
        title: str = "",
        file_name: str = "",
        color_column: str = "",
    ) -> str:
        if not dataframes:
            return "Nenhum arquivo CSV carregado. Peça ao usuário para fazer upload de um CSV na barra lateral."

        df = dataframes.get(file_name) if file_name else next(iter(dataframes.values()))
        fname = file_name or next(iter(dataframes))

        if x_column not in df.columns:
            return f"Coluna '{x_column}' não encontrada em '{fname}'. Disponíveis: {list(df.columns)}"
        if y_column and y_column not in df.columns:
            return f"Coluna '{y_column}' não encontrada em '{fname}'. Disponíveis: {list(df.columns)}"

        chart_title = title or f"{chart_type.title()} — {fname}"
        ct = chart_type.lower()

        try:
            bg = "#1a1a2e"
            fg = "#e0e0ff"
            accent = "#a78bfa"

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2a4a")
            ax.tick_params(colors=fg)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)

            if ct == "bar":
                ax.bar(df[x_column].astype(str), df[y_column], color=accent)
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                plt.xticks(rotation=45, ha="right")
            elif ct == "line":
                ax.plot(df[x_column], df[y_column], color=accent, linewidth=2, marker="o", markersize=3)
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
            elif ct == "scatter":
                ax.scatter(df[x_column], df[y_column], color=accent, alpha=0.7)
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
            elif ct == "pie":
                vals = df[y_column] if y_column else df[x_column].value_counts()
                labels = df[x_column] if y_column else vals.index
                ax.pie(vals, labels=labels, autopct="%1.1f%%",
                       textprops={"color": fg},
                       colors=plt.cm.Purples(
                           [0.3 + 0.5 * i / max(len(vals) - 1, 1) for i in range(len(vals))]
                       ))
            elif ct == "histogram":
                ax.hist(df[x_column].dropna(), bins=20, color=accent, edgecolor="#0e0e1a")
                ax.set_xlabel(x_column)
                ax.set_ylabel("Frequência")
            elif ct == "box":
                data = [df[c].dropna().values for c in ([y_column] if y_column else [x_column])]
                bp = ax.boxplot(data, patch_artist=True,
                                medianprops={"color": "#f0abfc"},
                                whiskerprops={"color": fg},
                                capprops={"color": fg},
                                flierprops={"markerfacecolor": accent})
                for patch in bp["boxes"]:
                    patch.set_facecolor(accent)
                    patch.set_alpha(0.7)
                ax.set_xticklabels([y_column or x_column])
            else:
                plt.close(fig)
                return f"Tipo '{chart_type}' não suportado. Use: bar, line, scatter, pie, histogram, box."

            ax.set_title(chart_title, pad=12)
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=bg)
            plt.close(fig)
            buf.seek(0)
            img_bytes = buf.read()

        except Exception as e:
            plt.close("all")
            return f"Erro ao criar gráfico: {e}"

        chart_id = str(uuid.uuid4())
        if "charts" not in st.session_state:
            st.session_state.charts = {}
        st.session_state.charts[chart_id] = img_bytes

        return f"[CHART:{chart_id}]\nGráfico **{chart_title}** gerado com sucesso."

    def youtube_transcript(url: str) -> str:
        video_id = _extract_youtube_id(url)
        if not video_id:
            return "Não consegui identificar o ID do vídeo. Verifique se a URL é válida."
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["pt", "pt-BR", "en", "es"]
            )
            text = " ".join(t["text"] for t in transcript_list)
            limit = 5000
            return text[:limit] + ("\n\n[... transcrição truncada ...]" if len(text) > limit else "")
        except Exception as e:
            return f"Erro ao obter transcrição do vídeo '{video_id}': {e}"

    tools = []

    if enable_charts:
        tools.append(StructuredTool.from_function(
            func=generate_chart,
            name="generate_chart",
            description=(
                "Gera gráficos a partir dos dados CSV carregados. "
                "Use sempre que o usuário pedir um gráfico, visualização, plot ou chart. "
                "Tipos disponíveis: bar, line, scatter, pie, histogram, box."
            ),
            args_schema=ChartInput,
        ))

    tools += [
        StructuredTool.from_function(
            func=analyze_csv,
            name="analyze_csv_data",
            description="Analisa os arquivos CSV carregados pelo usuário. Use para responder perguntas sobre dados, colunas, estatísticas ou tendências nos CSVs.",
            args_schema=QueryInput,
        ),
        StructuredTool.from_function(
            func=scrape_website,
            name="scrape_website",
            description="Faz web scraping de uma URL e retorna o conteúdo textual da página. Use quando o usuário fornecer um link de site (não YouTube).",
            args_schema=UrlInput,
        ),
        StructuredTool.from_function(
            func=youtube_transcript,
            name="youtube_transcript",
            description="Obtém a transcrição de um vídeo do YouTube. Use quando o usuário quiser resumir, analisar ou fazer perguntas sobre um vídeo do YouTube.",
            args_schema=UrlInput,
        ),
    ]

    if enable_search:
        tools.append(
            StructuredTool.from_function(
                func=search_web,
                name="search_web",
                description="Busca informações na internet via DuckDuckGo sem precisar de uma URL. Use quando o usuário quiser pesquisar um tema, notícia ou assunto.",
                args_schema=QueryInput,
            )
        )

    return tools


# ─── LLM factory ──────────────────────────────────────────────────────────────

def _get_llm(provider: str, api_key: str, model: str):
    if provider == "Groq":
        return ChatGroq(api_key=api_key, model=model, temperature=0.3)
    return ChatOpenAI(api_key=api_key, model=model, temperature=0.3)


# ─── Agent response ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Você é o **Oráculo**, um assistente de IA sábio, preciso e poderoso. "
    "Você possui ferramentas para analisar arquivos CSV, gerar gráficos interativos, fazer web scraping, buscar na internet e obter transcrições do YouTube. "
    "Use as ferramentas sempre que necessário para responder com precisão. "
    "Quando o usuário pedir um gráfico, chart ou visualização, use generate_chart. "
    "Quando a busca na web estiver disponível e o usuário perguntar sobre notícias, fatos atuais ou qualquer tema sem URL, use search_web. "
    "Responda sempre em português, de forma clara e organizada, usando markdown quando útil."
)

TOOL_LABELS: dict[str, str] = {
    "analyze_csv_data":  "📊 Analisando CSV...",
    "generate_chart":    "📈 Gerando gráfico...",
    "scrape_website":    "🌐 Fazendo scraping do site...",
    "search_web":        "🔍 Pesquisando na web...",
    "youtube_transcript":"🎬 Obtendo transcrição do YouTube...",
}


def _generate_suggestions(df: pd.DataFrame, fname: str, llm) -> list[str]:
    """Gera 4 perguntas sugeridas sobre o dataframe usando o LLM."""
    col_types = _infer_column_types(df)
    cols_desc = ", ".join(f"{col} ({sem})" for col, _, sem in col_types[:12])
    prompt = (
        f"Tenho um arquivo chamado '{fname}' com {len(df)} linhas e colunas: {cols_desc}. "
        "Gere exatamente 4 perguntas curtas e práticas que eu poderia fazer sobre esses dados. "
        "Retorne apenas as perguntas, uma por linha, sem numeração nem prefixos."
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return [l.strip() for l in resp.content.strip().splitlines() if l.strip()][:4]
    except Exception:
        return []

_CHART_RE = re.compile(r"\[CHART:([a-f0-9-]{36})\]")


def render_message(content: str) -> None:
    """Renderiza uma mensagem, substituindo marcadores [CHART:uuid] pela imagem PNG."""
    parts = _CHART_RE.split(content)
    charts = st.session_state.get("charts", {})
    # split alterna: texto, uuid, texto, uuid, ...
    for i, part in enumerate(parts):
        if i % 2 == 1:  # uuid capturado pelo grupo de regex
            img_bytes = charts.get(part)
            if img_bytes:
                st.image(img_bytes, use_container_width=True)
        elif part.strip():
            st.markdown(part)

_MALFORMED_RE = re.compile(
    r"<function=(?P<name>\w+)(?P<args>\{.*?\})</function>", re.DOTALL
)


def _run_malformed_tool_call(error_str: str, tool_map: dict) -> str | None:
    """Parseia e executa o formato inválido que o Groq às vezes gera."""
    match = _MALFORMED_RE.search(error_str)
    if not match:
        return None
    name = match.group("name")
    tool = tool_map.get(name)
    if not tool:
        return None
    try:
        args = json.loads(match.group("args"))
        return str(tool.invoke(args))
    except Exception:
        return None


def run_agent(
    prompt: str,
    llm,
    tools: list,
    history: list,
    on_tool_use=None,
) -> tuple[list, list[str]]:
    """
    Executa o loop de tool-calling até não restar mais chamadas.
    Retorna (messages, chart_markers) prontos para a chamada final de streaming.
    on_tool_use(tool_name): callback chamado antes de cada ferramenta.
    """
    tool_map = {t.name: t for t in tools}
    lm = llm.bind_tools(tools) if tools else llm
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=prompt)]
    chart_markers: list[str] = []

    for _ in range(6):
        try:
            response = lm.invoke(messages)
        except Exception as e:
            raw = _run_malformed_tool_call(str(e), tool_map)
            if raw:
                for m in _CHART_RE.finditer(raw):
                    chart_markers.append(f"[CHART:{m.group(1)}]")
                messages += [AIMessage(content=""), ToolMessage(content=raw, tool_call_id="fallback")]
                return messages, chart_markers
            raise

        if not getattr(response, "tool_calls", None):
            # sem mais tool calls — mensagens prontas para streaming final
            return messages, chart_markers

        messages.append(response)
        for tc in response.tool_calls:
            if on_tool_use:
                on_tool_use(tc["name"])
            tool = tool_map.get(tc["name"])
            result = str(tool.invoke(tc["args"])) if tool else f"Ferramenta '{tc['name']}' não encontrada."
            for m in _CHART_RE.finditer(result):
                chart_markers.append(f"[CHART:{m.group(1)}]")
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    return messages, chart_markers


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔮 Oráculo")
    st.caption("Assistente de IA com superpoderes")
    st.divider()

    st.subheader("⚙️ Configuração da IA")

    provider = st.radio("Provedor", ["Groq", "OpenAI"], horizontal=True)

    if provider == "Groq":
        default_key = os.getenv("GROQ_API_KEY", "")
        api_key = st.text_input("Groq API Key", value=default_key, type="password")
        model = st.selectbox("Modelo", [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
        ])
    else:
        default_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key", value=default_key, type="password")
        model = st.selectbox("Modelo", [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
        ])

    st.divider()

    st.subheader("🛠️ Ferramentas")
    search_engine = st.selectbox(
        "🔍 Busca na Web",
        options=["Desligado", "DuckDuckGo"],
        help="Quando ativado, o Oráculo pode pesquisar na internet sem precisar de uma URL.",
    )
    enable_search = search_engine != "Desligado"

    enable_charts = st.toggle(
        "📈 Geração de Gráficos",
        value=True,
        help="Quando ativado, o Oráculo pode criar gráficos a partir dos dados CSV.",
    )

    st.divider()

    st.subheader("📂 Arquivos CSV")
    uploaded_files = st.file_uploader(
        "Carregar arquivos",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if "dataframes" not in st.session_state:
        st.session_state.dataframes = {}
    if "suggestions" not in st.session_state:
        st.session_state.suggestions = {}
    if "pending_suggestions" not in st.session_state:
        st.session_state.pending_suggestions = set()

    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.dataframes:
                if f.name.endswith(".xlsx"):
                    st.session_state.dataframes[f.name] = pd.read_excel(f, engine="openpyxl")
                else:
                    st.session_state.dataframes[f.name] = pd.read_csv(f)
                st.session_state.pending_suggestions.add(f.name)

    if st.session_state.dataframes:
        for name, df in st.session_state.dataframes.items():
            with st.expander(f"📄 {name}"):
                st.caption(f"{df.shape[0]} linhas × {df.shape[1]} colunas")
                st.dataframe(df.head(5), width='stretch')
        if st.button("🗑️ Remover todos os CSVs", width='stretch'):
            st.session_state.dataframes = {}
            st.session_state.suggestions = {}
            st.session_state.pending_suggestions = set()
            st.rerun()
    else:
        st.caption("Nenhum arquivo carregado.")

    st.divider()

    if st.button("🧹 Limpar conversa", width='stretch'):
        st.session_state.messages = []
        st.rerun()

# ─── Main chat ─────────────────────────────────────────────────────────────────

st.title("🔮 Oráculo")
st.caption("Pergunte qualquer coisa • analise CSVs • pesquise na web • entenda vídeos do YouTube")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Gerar sugestões para arquivos novos ───────────────────────────────────────
if api_key and st.session_state.get("pending_suggestions"):
    llm_sug = _get_llm(provider, api_key, model)
    for fname in list(st.session_state.pending_suggestions):
        df_sug = st.session_state.dataframes.get(fname)
        if df_sug is not None:
            with st.spinner(f"Gerando sugestões para {fname}..."):
                st.session_state.suggestions[fname] = _generate_suggestions(df_sug, fname, llm_sug)
        st.session_state.pending_suggestions.discard(fname)

# ── Histórico ─────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🔮" if msg["role"] == "assistant" else None):
        render_message(msg["content"])

# ── Boas-vindas + sugestões ───────────────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🔮"):
        st.markdown(
            "Olá! Sou o **Oráculo**, seu assistente de IA. Posso te ajudar a:\n\n"
            "- 📊 **Analisar arquivos CSV** — carregue um arquivo na barra lateral\n"
            "- 📈 **Gerar gráficos** — peça um gráfico e eu crio direto aqui no chat\n"
            "- 🌐 **Fazer scraping de sites** — cole uma URL e peça um resumo\n"
            "- 🔍 **Buscar na internet** — ative a busca na barra lateral e pergunte qualquer coisa\n"
            "- 🎬 **Entender vídeos do YouTube** — cole o link e faça suas perguntas\n\n"
            "Como posso te ajudar hoje?"
        )

# Sugestões como botões clicáveis
all_suggestions = [q for qs in st.session_state.get("suggestions", {}).values() for q in qs]
if all_suggestions and not st.session_state.messages:
    st.markdown("**💡 Sugestões para começar:**")
    cols = st.columns(2)
    for i, q in enumerate(all_suggestions):
        if cols[i % 2].button(q, key=f"sug_{i}", use_container_width=True):
            st.session_state.pending_input = q
            st.session_state.suggestions = {}
            st.rerun()

# ── Input do chat ─────────────────────────────────────────────────────────────
typed_input = st.chat_input("Pergunte ao Oráculo...")
user_input  = typed_input or st.session_state.pop("pending_input", None)

if user_input:
    if not api_key:
        st.error("⚠️ Insira a API Key na barra lateral antes de continuar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🔮"):
        tool_status = st.empty()
        answer = ""
        try:
            llm   = _get_llm(provider, api_key, model)
            tools = _build_tools(st.session_state.dataframes, enable_search, enable_charts)

            history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    history.append(HumanMessage(content=msg["content"]))
                else:
                    history.append(AIMessage(content=msg["content"]))

            def on_tool_use(name: str):
                tool_status.markdown(f"*{TOOL_LABELS.get(name, f'⚙️ {name}...')}*")

            messages, chart_markers = run_agent(user_input, llm, tools, history, on_tool_use)
            tool_status.empty()

            # Exibe gráficos antes do stream
            for marker in chart_markers:
                m = _CHART_RE.match(marker)
                if m:
                    img = st.session_state.get("charts", {}).get(m.group(1))
                    if img:
                        st.image(img, use_container_width=True)

            # Stream da resposta final palavra por palavra
            def _stream():
                for chunk in llm.stream(messages):
                    if chunk.content:
                        yield chunk.content

            streamed_text = st.write_stream(_stream())
            chart_prefix  = "\n".join(chart_markers) + "\n\n" if chart_markers else ""
            answer        = chart_prefix + (streamed_text or "")

        except Exception as e:
            tool_status.empty()
            answer = f"❌ Erro ao processar sua pergunta: {e}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
