# -*- coding: utf-8 -*-
"""
bolhas_publicacoes.py — Matriz de bolhas de publicações por grupo × ano (C4AI)
===============================================================================
Variante em bolhas da Figura 4 (4_heatmap_grupo_ano.png, mapa de calor), usada
especificamente na versão do capítulo 3 da tese: mesma grade grupo × ano, com
o tamanho da bolha indicando o número de publicações e a cor identificando
categoricamente o grupo de pesquisa (paleta Okabe-Ito, à prova de daltonismo),
em vez de uma escala de cor sequencial (viridis) redundante com o tamanho.

A ordem dos grupos (eixo y) segue o ranking de produtividade (decrescente),
igual à Figura 1/Tabela 2 do relatório. A atribuição de cor por grupo nessa
ordem é deliberadamente diferente da atribuição usada em
equipe_composicao.py (ordem alfabética) — assim, o mesmo grupo aparece em
cores diferentes nas duas figuras-par (publicações × equipe), e as duas
figuras ficam visualmente diferenciáveis entre si, sem recorrer a paletas
fora da identidade da tese (ver notas de estilo em equipe_composicao.py,
espelhando infranodus/estilo_rede.py).

Fonte dos dados: output/c4ai_matriz_grupo_ano.xlsx (crosstab gerado por
analise_publicacoes a partir de c4ai_publicacoes.xlsx).

Uso:
    python bolhas_publicacoes.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# DADOS
# ──────────────────────────────────────────────────────────────────────────────

INPUT_MATRIZ = "output/c4ai_matriz_grupo_ano.xlsx"

# Ordem de produtividade decrescente (igual à Figura 1 / Tabela 2 do RELATORIO.md)
GRUPOS_ORDEM = [
    "NLP2", "KEML", "AGRIBIO", "AI HEALTH", "HUMANITIES",
    "PROINDL", "MClimate", "OceanML",
]

# Paleta categórica Okabe-Ito (colorblind-safe), na ordem de produtividade —
# deliberadamente distinta da ordem alfabética usada em equipe_composicao.py,
# para que cada grupo tenha cor diferente nas duas figuras-par.
CORES_POR_GRUPO = {
    "NLP2":       "#0072B2",  # azul
    "KEML":       "#E69F00",  # laranja
    "AGRIBIO":    "#009E73",  # verde
    "AI HEALTH":  "#CC79A7",  # magenta
    "HUMANITIES": "#56B4E9",  # azul claro
    "PROINDL":    "#D55E00",  # vermelho
    "MClimate":   "#F0E442",  # amarelo
    "OceanML":    "#999999",  # cinza
}

# ──────────────────────────────────────────────────────────────────────────────
# ESTILO (mesma identidade visual de equipe_composicao.py / estilo_rede.py)
# ──────────────────────────────────────────────────────────────────────────────

COR_TEXTO = "#404040"
COR_NOTA = "#8a8a8a"
COR_LEGENDA_TAMANHO = "#5a5a5a"  # cinza neutro: legenda de tamanho não usa cor de grupo

plt.rcParams["font.family"] = "DejaVu Sans"


def cor_texto_sobre(hex_cor: str) -> str:
    """Réplica da regra de contraste usada nas demais figuras da tese:
    branco se luminância < 0,55, cinza escuro caso contrário."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_cor)
    luminancia = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminancia < 0.55 else COR_TEXTO


def save(fig, outdir: Path, filename: str):
    path = outdir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {path}")


def carregar_matriz(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, index_col=0)
    df = df.drop(columns=["All"], errors="ignore")
    df = df.drop(index=["All"], errors="ignore")
    df = df.reindex(GRUPOS_ORDEM)
    return df


def plot_bolhas_publicacoes(matriz: pd.DataFrame, outdir: Path):
    grupos = list(matriz.index)
    anos = list(matriz.columns)

    fig, ax = plt.subplots(figsize=(14, 9))

    valores = matriz.values.astype(float)
    vmin, vmax = valores.min(), valores.max()
    tamanho_min, tamanho_max = 300, 3800

    for i, grupo in enumerate(grupos):
        cor = CORES_POR_GRUPO[grupo]
        for j, ano in enumerate(anos):
            total = matriz.loc[grupo, ano]
            frac = (total - vmin) / (vmax - vmin)
            tamanho = tamanho_min + frac * (tamanho_max - tamanho_min)

            ax.scatter(
                j, i, s=tamanho, color=cor, edgecolors="white", linewidths=1.5,
                zorder=3,
            )
            ax.text(
                j, i, f"{int(total)}", ha="center", va="center", fontsize=11,
                color=cor_texto_sobre(cor), zorder=4,
            )

    ax.set_xticks(range(len(anos)))
    ax.set_xticklabels([f"{a}" for a in anos], color=COR_TEXTO)
    ax.set_yticks(range(len(grupos)))
    ax.set_yticklabels(grupos, color=COR_TEXTO)
    ax.invert_yaxis()

    ax.set_xlim(-0.6, len(anos) - 0.4)
    ax.set_ylim(len(grupos) - 0.4, -0.6)

    ax.set_xlabel("Ano", color=COR_TEXTO)
    ax.set_ylabel("Grupo de Pesquisa", color=COR_TEXTO)
    ax.tick_params(colors=COR_TEXTO)
    for spine in ax.spines.values():
        spine.set_color(COR_NOTA)

    ax.grid(True, alpha=0.2, linewidth=0.8, color=COR_NOTA)
    ax.set_axisbelow(True)

    # legenda de tamanho (cor neutra: aqui a cor do marcador identifica o
    # grupo, já dado pelo eixo y — a legenda trata apenas da escala de tamanho).
    # Escala reduzida (não literalmente igual à do gráfico) só para caber sem
    # sobrepor o texto — a proporção relativa entre os três pontos é preservada.
    legenda_tamanho_min, legenda_tamanho_max = 90, 700
    for valor_ref in (10, 30, 60):
        frac = (valor_ref - vmin) / (vmax - vmin)
        frac = min(max(frac, 0), 1)
        tamanho = legenda_tamanho_min + frac * (legenda_tamanho_max - legenda_tamanho_min)
        ax.scatter(
            [], [], s=tamanho, color=COR_LEGENDA_TAMANHO, edgecolors="white",
            linewidths=1.5, label=f"{valor_ref} publicações",
        )
    legenda = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False,
        fontsize=10, labelcolor=COR_TEXTO, columnspacing=3.0, handletextpad=1.0,
        borderaxespad=1.5,
    )

    nota_rodape = (
        "Cor da bolha = grupo de pesquisa (paleta categórica Okabe-Ito, à prova de daltonismo); "
        "tamanho da bolha = número de publicações no ano. "
        "Atribuição de cor por grupo distinta da usada na Figura 12 (composição de equipe), "
        "para diferenciar visualmente as duas matrizes de bolhas."
    )
    fig.text(0.02, -0.08, nota_rodape, fontsize=8.5, color=COR_NOTA, style="italic", ha="left", va="top")

    plt.tight_layout()
    save(fig, outdir, "4_heatmap_grupo_ano_bolhas.png")


def main():
    matriz = carregar_matriz(INPUT_MATRIZ)
    for outdir_name in ("output", "figuras"):
        outdir = Path(outdir_name)
        outdir.mkdir(parents=True, exist_ok=True)
        plot_bolhas_publicacoes(matriz, outdir)
    print("\nMatriz de bolhas de publicações concluída.\n")


if __name__ == "__main__":
    main()
