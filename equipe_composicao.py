# -*- coding: utf-8 -*-
"""
equipe_composicao.py — Composição de equipe por grupo do C4AI (2021–2025)
==========================================================================
Figura-par da matriz de bolhas de publicações (4_heatmap_grupo_ano_bolhas.png /
bolhas_publicacoes.py). Mesma diagramação de heatmap-bolhas com colorbar vertical
à direita, sem números dentro das bolhas — a mesma da versão viridis anterior,
apenas com uma escala de cor diferente. O tamanho E a cor da bolha indicam o
total de pesquisadores declarado no relatório anual à FAPESP (em vez do número
de publicações), numa escala sequencial monocromática branco -> vermelho
Okabe-Ito (#D55E00). O matiz vermelho a diferencia da figura-par de publicações,
que usa a escala branco -> azul Okabe-Ito.

Fonte dos dados: seções "Human resources" (relatórios 2021–2022) e "Team"
(relatórios 2023–2025) de cada capítulo dos relatórios científicos do C4AI à
FAPESP, conforme tabulado em composicao_equipe_c4ai_2021_2025.md (curadoria
manual).

Observações metodológicas importantes:

- O ano aqui é o ANO DO RELATÓRIO FAPESP (período ago. ano-1–jul. ano), não o ano
  civil de publicação usado na figura-par (2020–2024). As duas escalas temporais
  NÃO foram forçadas a coincidir célula a célula.
- Marcadores em × cinza = grupo sem capítulo de equipe próprio nesse relatório
  (dado ausente, não equipe de tamanho zero).
- Células marcadas com "*" agregam mais de um subdesafio sob uma única contagem
  de equipe no relatório de origem (ver NOTAS_CELULAS abaixo).

Uso:
    python equipe_composicao.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from scipy.interpolate import PchipInterpolator

# ──────────────────────────────────────────────────────────────────────────────
# DADOS
# ──────────────────────────────────────────────────────────────────────────────

# Mesma ordem de grupos (alfabética) usada no heatmap de publicações, para
# permitir leitura pareada linha a linha entre as duas figuras.
GRUPOS = [
    "AGRIBIO",
    "AI HEALTH",
    "HUMANITIES",
    "KEML",
    "MClimate",
    "NLP2",
    "OceanML",
    "PROINDL",
]

ANOS = [2021, 2022, 2023, 2024, 2025]

# Total de pesquisadores declarado por grupo/ano de relatório. NaN = grupo sem
# capítulo de equipe próprio nesse relatório (dado ausente, não zero).
TOTAIS = pd.DataFrame(
    {
        2021: [42, 19, np.nan, 35, np.nan, 95, np.nan, np.nan],
        2022: [26, 18, 18, 47, np.nan, 93, np.nan, np.nan],
        2023: [31, 18, 21, 50, np.nan, 110, np.nan, 24],
        2024: [np.nan, np.nan, 17, 44, np.nan, 108, 15, 24],
        2025: [np.nan, np.nan, 17, 57, 29, 100, 17, 31],
    },
    index=GRUPOS,
)

# Notas por célula (grupo, ano) -> texto da nota, para as células marcadas com "*".
NOTAS_CELULAS = {
    ("AGRIBIO", 2023): (
        "Relatório rotula o capítulo \"AgriBio (com subdesafio MClimate)\"; os 31 "
        "pesquisadores cobrem as duas frentes juntas, sem contagem nominal separada "
        "para MClimate nesse ano."
    ),
    ("MClimate", 2025): (
        "Relatório rotula o capítulo \"MClimate (incorpora subseção AgriBio)\"; os "
        "29 pesquisadores cobrem as duas frentes juntas, sem contagem nominal "
        "separada para AgriBio nesse ano."
    ),
    ("NLP2", 2023): (
        "Total usa o valor nominal contável de mestrandos (29), que diverge da "
        "frase-resumo do relatório (\"29 mestrandos\" declarados vs. 27 nomes "
        "listáveis) — discrepância não resolvida na fonte."
    ),
    ("HUMANITIES", 2021): (
        "Não computado: no relatório de 2021 os participantes de AI Humanities são "
        "listados por projeto, com sobreposição de nomes entre projetos, e uma "
        "contagem simples infla o total."
    ),
}

# Paleta categórica Okabe-Ito (colorblind-safe), usada só no streamgraph
# (Figura 13, alternativa aos mesmos dados). A matriz de bolhas (Figura 12)
# não usa cor por grupo — usa a escala sequencial CMAP_SEQUENCIAL_EQUIPE.
CORES_OKABE_ITO = {
    "AGRIBIO":    "#0072B2",
    "AI HEALTH":  "#E69F00",
    "HUMANITIES": "#009E73",
    "KEML":       "#CC79A7",
    "MClimate":   "#56B4E9",
    "NLP2":       "#D55E00",
    "OceanML":    "#F0E442",
    "PROINDL":    "#999999",
}


# ──────────────────────────────────────────────────────────────────────────────
# ESTILO
# ──────────────────────────────────────────────────────────────────────────────

COR_TEXTO = "#404040"
COR_NOTA = "#8a8a8a"
COR_AUSENTE = "#999999"  # cinza da paleta categórica Okabe-Ito (slot neutro)

# Escala sequencial monocromática (branco -> vermelho Okabe-Ito #D55E00), não
# viridis. Matiz distinto do usado em bolhas_publicacoes.py (azul Okabe-Ito
# #0072B2), para diferenciar as duas matrizes de bolhas entre si.
COR_ANCORA_EQUIPE = "#D55E00"
CMAP_SEQUENCIAL_EQUIPE = LinearSegmentedColormap.from_list(
    "vermelho_okabe_ito", ["#ffffff", COR_ANCORA_EQUIPE],
)

# Piso de saturação da cor no valor mínimo (só afeta a cor, não o tamanho).
COR_FRAC_MIN = 0.18

plt.rcParams["font.family"] = "DejaVu Sans"


def cor_texto_sobre(cor) -> str:
    """Réplica da regra de contraste de estilo_rede.py: branco se luminância < 0,55."""
    r, g, b = to_rgb(cor)
    luminancia = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminancia < 0.55 else COR_TEXTO


def save(fig, outdir: Path, filename: str):
    path = outdir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {path}")


def plot_composicao_bolhas(totais: pd.DataFrame, outdir: Path):
    grupos = list(totais.index)
    anos = list(totais.columns)

    fig, ax = plt.subplots(figsize=(11, 6))

    valores = totais.values.astype(float)
    vmin, vmax = float(np.nanmin(valores)), float(np.nanmax(valores))
    norma = Normalize(vmin=vmin, vmax=vmax)

    tamanho_min, tamanho_max = 30, 900

    xs_dado, ys_dado, tamanhos, cores = [], [], [], []
    xs_ausente, ys_ausente = [], []
    for i, grupo in enumerate(grupos):
        for j, ano in enumerate(anos):
            total = totais.loc[grupo, ano]
            if pd.isna(total):
                xs_ausente.append(j)
                ys_ausente.append(i)
                continue
            frac = norma(float(total))
            xs_dado.append(j)
            ys_dado.append(i)
            tamanhos.append(tamanho_min + frac * (tamanho_max - tamanho_min))
            cores.append(CMAP_SEQUENCIAL_EQUIPE(COR_FRAC_MIN + (1 - COR_FRAC_MIN) * frac))

    ax.scatter(xs_ausente, ys_ausente, s=45, marker="x", color=COR_AUSENTE,
               linewidths=1.2, zorder=2)
    ax.scatter(xs_dado, ys_dado, s=tamanhos, c=cores, edgecolors="white",
               linewidths=1.0, zorder=3)

    ax.set_xticks(range(len(anos)))
    ax.set_xticklabels([f"{a}" for a in anos], color=COR_TEXTO)
    ax.set_yticks(range(len(grupos)))
    ax.set_yticklabels(grupos, color=COR_TEXTO)
    ax.invert_yaxis()

    ax.set_xlim(-0.6, len(anos) - 0.4)
    ax.set_ylim(len(grupos) - 0.4, -0.6)

    ax.set_xlabel("ano do relatório FAPESP (período ago. ano−1–jul. ano)",
                  color=COR_TEXTO)
    ax.tick_params(colors=COR_TEXTO)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6, color=COR_NOTA)
    ax.set_axisbelow(True)

    # colorbar vertical à direita, na mesma escala do gráfico (branco -> vermelho),
    # com o mesmo piso de saturação COR_FRAC_MIN aplicado às bolhas.
    cmap_display = LinearSegmentedColormap.from_list(
        "vermelho_display",
        [CMAP_SEQUENCIAL_EQUIPE(COR_FRAC_MIN + (1 - COR_FRAC_MIN) * t)
         for t in np.linspace(0, 1, 256)],
    )
    sm = ScalarMappable(norm=norma, cmap=cmap_display)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=25)
    cbar.set_label("pesquisadores (tamanho e cor da bolha)",
                   color=COR_TEXTO, fontsize=9.5)
    cbar.ax.tick_params(colors=COR_TEXTO, labelsize=9)
    cbar.outline.set_visible(False)

    # marcador de ausência (× cinza) explicado em nota curta abaixo do eixo x
    nota = (
        "× = grupo sem capítulo de equipe próprio no relatório (dado ausente, não equipe de tamanho zero)."
    )
    fig.text(0.02, -0.04, nota, fontsize=8.5, color=COR_NOTA, style="italic",
             ha="left", va="top")

    plt.tight_layout()
    save(fig, outdir, "12_composicao_equipe_bolhas.png")


def plot_composicao_streamgraph(totais: pd.DataFrame, outdir: Path):
    """Alternativa não convencional à bolha: rio temporal (streamgraph), baseline
    'sym' (silhueta centrada, estilo ThemeRiver). Cada grupo é uma faixa cuja
    largura é o total de pesquisadores; grupo sem capítulo próprio afina até
    largura zero, em vez de precisar de um marcador de exceção.

    Limite metodológico herdado da própria forma: como o rio não distingue
    "zero pesquisadores" de "capítulo ausente/fundido em outro", a leitura fina
    de cada célula continua sendo tarefa da Figura 12 (bolhas) — este gráfico é
    deliberadamente impressionista, não substitui a leitura célula a célula.
    """
    grupos = list(totais.index)
    anos = np.array(totais.columns, dtype=float)

    # NaN -> 0 apenas para a pilha visual (ver ressalva acima e na legenda da figura)
    y_conhecido = totais.fillna(0.0).values

    anos_finos = np.linspace(anos.min(), anos.max(), 400)
    y_fino = np.vstack([
        PchipInterpolator(anos, y_conhecido[i])(anos_finos).clip(min=0)
        for i in range(len(grupos))
    ])

    # baseline 'sym' replicado manualmente (mesma fórmula do stackplot do matplotlib)
    # para poder localizar o centro de cada faixa e rotulá-la diretamente.
    cumsum = np.cumsum(y_fino, axis=0)
    linha_base = -np.sum(y_fino, axis=0) * 0.5
    topos = linha_base + cumsum
    bases = np.vstack([linha_base, topos[:-1]])
    centros = (bases + topos) / 2

    cores = [CORES_OKABE_ITO[g] for g in grupos]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.stackplot(anos_finos, y_fino, colors=cores, baseline="sym", linewidth=0)

    margem = (anos_finos.max() - anos_finos.min()) * 0.05
    for i, grupo in enumerate(grupos):
        idx_max = np.argmax(y_fino[i])
        if y_fino[i, idx_max] < 1:
            continue
        cor_fundo = to_rgb(cores[i])

        x_rotulo = anos_finos[idx_max]
        ha = "center"
        if x_rotulo <= anos_finos.min() + margem:
            x_rotulo = anos_finos.min() + margem
            ha = "left"
        elif x_rotulo >= anos_finos.max() - margem:
            x_rotulo = anos_finos.max() - margem
            ha = "right"

        ax.text(
            x_rotulo, centros[i, idx_max], grupo,
            ha=ha, va="center", fontsize=9.5,
            color=cor_texto_sobre(cor_fundo), zorder=4,
        )

    ax.set_xticks(anos)
    ax.set_xticklabels([f"{int(a)}" for a in anos], color=COR_TEXTO)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("Ano do relatório FAPESP (período ago. ano−1–jul. ano)", color=COR_TEXTO)
    ax.tick_params(colors=COR_TEXTO)

    handles = [plt.Rectangle((0, 0), 1, 1, color=CORES_OKABE_ITO[g]) for g in grupos]
    ax.legend(
        handles, grupos, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4,
        frameon=False, fontsize=10, labelcolor=COR_TEXTO, title="Grupo de Pesquisa",
        title_fontsize=10,
    )

    nota_rodape = (
        "Largura da faixa = total de pesquisadores; curvas suavizadas (interpolação PCHIP) entre os 5 pontos "
        "anuais conhecidos — recurso visual do streamgraph, não medição contínua entre relatórios.\n"
        "Largura zero não distingue \"grupo sem capítulo próprio nesse ano\" de \"zero pesquisadores\": ao contrário "
        "da Figura 12, este gráfico não tem símbolo de exceção para dado ausente — ver Figura 12 para a leitura célula a célula.\n"
        "Escala de ano distinta da usada no heatmap de publicações (ano civil, 2020–2024): aqui o ano é o período de "
        "relatório à FAPESP (ago.–jul.)."
    )
    fig.text(0.02, -0.09, nota_rodape, fontsize=8.5, color=COR_NOTA, style="italic", ha="left", va="top")

    plt.tight_layout()
    save(fig, outdir, "13_composicao_equipe_streamgraph.png")


def export_tabela(totais: pd.DataFrame, outdir: Path):
    tabela = totais.reset_index().rename(columns={"index": "Grupo"})
    path = outdir / "c4ai_composicao_equipe.xlsx"
    with pd.ExcelWriter(path) as writer:
        tabela.to_excel(writer, sheet_name="Total_por_grupo_ano", index=False)
        notas = pd.DataFrame(
            [{"Grupo": g, "Ano": a, "Nota": n} for (g, a), n in NOTAS_CELULAS.items()]
        )
        notas.to_excel(writer, sheet_name="Notas_metodologicas", index=False)
    print(f"  ✓  {path}")


def main():
    for outdir_name in ("output", "figuras"):
        outdir = Path(outdir_name)
        outdir.mkdir(parents=True, exist_ok=True)
        plot_composicao_bolhas(TOTAIS, outdir)
        plot_composicao_streamgraph(TOTAIS, outdir)

    export_tabela(TOTAIS, Path("output"))
    print("\nComposição de equipe concluída.\n")


if __name__ == "__main__":
    main()
