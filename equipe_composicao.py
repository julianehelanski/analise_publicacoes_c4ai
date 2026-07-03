# -*- coding: utf-8 -*-
"""
equipe_composicao.py — Composição de equipe por grupo do C4AI (2021–2025)
==========================================================================
Figura-par do heatmap de publicações (4_heatmap_grupo_ano.png / analise_publicacoes):
mesma grade grupo × ano, trocando a cor (nº de publicações) pelo tamanho da bolha
(nº de pesquisadores declarado no relatório anual à FAPESP).

Fonte dos dados: seções "Human resources" (relatórios 2021–2022) e "Team"
(relatórios 2023–2025) de cada capítulo dos relatórios científicos do C4AI à FAPESP,
conforme tabulado em composicao_equipe_c4ai_2021_2025.md (curadoria manual).

Observações metodológicas importantes (ver notas no próprio gráfico):

- O ano aqui é o ANO DO RELATÓRIO FAPESP (período ago. ano-1–jul. ano), não o ano
  civil de publicação usado no heatmap (2020–2024). As duas escalas temporais NÃO
  foram forçadas a coincidir célula a célula — cada figura mantém seu próprio eixo,
  e a leitura comparativa entre as duas fica a cargo do texto da tese.
- Células em branco (marcador cinza) = grupo sem capítulo de equipe próprio no
  relatório daquele ano (não é equipe de tamanho zero: é dado ausente).
- Células marcadas com "*" agregam mais de um subdesafio sob uma única contagem de
  equipe no relatório de origem (ver NOTAS_CELULAS abaixo) — não é possível
  desagregar sem contagem nominal própria.

Uso:
    python equipe_composicao.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

CELULAS_COM_NOTA = set(NOTAS_CELULAS.keys())


# ──────────────────────────────────────────────────────────────────────────────
# PLOT
# ──────────────────────────────────────────────────────────────────────────────

# ── Identidade visual da tese (estilo_rede.py: viridis sequencial, "bolinha" com
# borda branca, DejaVu Sans, tinta cinza #404040 sem negrito, nota em itálico
# #8a8a8a, sem título embutido na imagem — a legenda do LaTeX titula) ──────────

COR_TEXTO = "#404040"
COR_NOTA = "#8a8a8a"
COR_AUSENTE = "#999999"  # cinza da paleta categórica Okabe-Ito (slot neutro)

plt.rcParams["font.family"] = "DejaVu Sans"


def cor_texto_sobre(rgba) -> str:
    """Réplica da regra de contraste de estilo_rede.py: branco se luminância < 0,55."""
    r, g, b = rgba[0], rgba[1], rgba[2]
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

    fig, ax = plt.subplots(figsize=(16, 10))

    valores = totais.values.astype(float)
    vmin, vmax = np.nanmin(valores), np.nanmax(valores)
    cmap = plt.get_cmap("viridis")

    # tamanho da bolha por área (não por raio), para não exagerar a diferença visual
    tamanho_min, tamanho_max = 400, 4200

    for i, grupo in enumerate(grupos):
        for j, ano in enumerate(anos):
            total = totais.loc[grupo, ano]

            if pd.isna(total):
                # dado ausente: grupo sem capítulo de equipe próprio nesse relatório
                ax.scatter(
                    j, i, s=90, marker="x", color=COR_AUSENTE, linewidths=1.5, zorder=2,
                )
                continue

            frac = (total - vmin) / (vmax - vmin)
            tamanho = tamanho_min + frac * (tamanho_max - tamanho_min)
            cor = cmap(frac)

            ax.scatter(
                j, i, s=tamanho, color=cor, edgecolors="white", linewidths=1.5,
                zorder=3,
            )

            rotulo = f"{total:.0f}"
            if (grupo, ano) in CELULAS_COM_NOTA:
                rotulo += "*"

            ax.text(
                j, i, rotulo, ha="center", va="center", fontsize=11,
                color=cor_texto_sobre(cor), zorder=4,
            )

    ax.set_xticks(range(len(anos)))
    ax.set_xticklabels([f"{a}" for a in anos], color=COR_TEXTO)
    ax.set_yticks(range(len(grupos)))
    ax.set_yticklabels(grupos, color=COR_TEXTO)
    ax.invert_yaxis()

    ax.set_xlim(-0.6, len(anos) - 0.4)
    ax.set_ylim(len(grupos) - 0.4, -0.6)

    ax.set_xlabel("Ano do relatório FAPESP (período ago. ano−1–jul. ano)", color=COR_TEXTO)
    ax.set_ylabel("Grupo de Pesquisa", color=COR_TEXTO)
    ax.tick_params(colors=COR_TEXTO)
    for spine in ax.spines.values():
        spine.set_color(COR_NOTA)

    # sem título embutido na imagem: a legenda do LaTeX/Markdown titula a figura

    ax.grid(True, alpha=0.2, linewidth=0.8, color=COR_NOTA)
    ax.set_axisbelow(True)

    # legenda de tamanho (3 pontos de referência)
    for valor_ref in (20, 60, 100):
        frac = (valor_ref - vmin) / (vmax - vmin)
        frac = min(max(frac, 0), 1)
        tamanho = tamanho_min + frac * (tamanho_max - tamanho_min)
        ax.scatter(
            [], [], s=tamanho, color=cmap(frac), edgecolors="white",
            linewidths=1.5, label=f"{valor_ref} pesquisadores",
        )
    ax.scatter([], [], s=90, marker="x", color=COR_AUSENTE, linewidths=1.5,
               label="sem capítulo de equipe próprio")
    legenda = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False,
        fontsize=10, labelcolor=COR_TEXTO,
    )

    nota_rodape = (
        "* célula agrega mais de uma frente sob uma única contagem de equipe no relatório de origem, "
        "ou apresenta discrepância entre o texto do relatório e a contagem nominal — ver notas metodológicas no script.\n"
        "Escala de ano distinta da usada no heatmap de publicações (ano civil, 2020–2024): "
        "aqui o ano é o período de relatório à FAPESP (ago.–jul.), não alinhado célula a célula com a produção."
    )
    fig.text(0.02, -0.06, nota_rodape, fontsize=8.5, color=COR_NOTA, style="italic", ha="left", va="top")

    plt.tight_layout()
    save(fig, outdir, "12_composicao_equipe_bolhas.png")


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

    export_tabela(TOTAIS, Path("output"))
    print("\nComposição de equipe concluída.\n")


if __name__ == "__main__":
    main()
