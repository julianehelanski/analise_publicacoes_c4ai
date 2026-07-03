# -*- coding: utf-8 -*-
"""
bolhas_publicacoes.py — Matriz de bolhas de publicações por grupo × ano (C4AI)
===============================================================================
Variante em bolhas da Figura 4 (4_heatmap_grupo_ano.png, mapa de calor), usada
especificamente na versão do capítulo 3 da tese: mesma grade grupo × ano, com
o tamanho E a cor da bolha indicando o número de publicações no ano (escala
sequencial monocromática, do claro ao escuro), em vez da escala viridis, que
foge da identidade visual da tese.

A escala sequencial é construída sobre o azul do núcleo categórico Okabe-Ito
(#0072B2), do branco ao azul saturado, e não sobre um grupo específico — a cor
aqui não identifica o grupo de pesquisa (esse já está no eixo y), só o valor da
célula. A figura-par de composição de equipe (equipe_composicao.py) usa uma
escala análoga, mas ancorada num matiz diferente (vermelho Okabe-Ito), para que
as duas matrizes de bolhas fiquem diferenciáveis entre si mesmo compartilhando
a mesma paleta-mestra da tese (ver notas de estilo em equipe_composicao.py,
espelhando infranodus/estilo_rede.py).

Fonte dos dados: output/c4ai_matriz_grupo_ano.xlsx (crosstab gerado por
analise_publicacoes a partir de c4ai_publicacoes.xlsx).

Uso:
    python bolhas_publicacoes.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# ──────────────────────────────────────────────────────────────────────────────
# DADOS
# ──────────────────────────────────────────────────────────────────────────────

INPUT_MATRIZ = "output/c4ai_matriz_grupo_ano.xlsx"

# Ordem de produtividade decrescente (igual à Figura 1 / Tabela 2 do RELATORIO.md)
GRUPOS_ORDEM = [
    "NLP2", "KEML", "AGRIBIO", "AI HEALTH", "HUMANITIES",
    "PROINDL", "MClimate", "OceanML",
]

# ──────────────────────────────────────────────────────────────────────────────
# ESTILO (mesma identidade visual de equipe_composicao.py / estilo_rede.py)
# ──────────────────────────────────────────────────────────────────────────────

COR_TEXTO = "#404040"
COR_NOTA = "#8a8a8a"

# Escala sequencial monocromática (branco -> azul Okabe-Ito #0072B2), não
# viridis. Ancorada em cor distinta da usada em equipe_composicao.py (vermelho
# Okabe-Ito), para diferenciar as duas matrizes de bolhas entre si.
COR_ANCORA = "#0072B2"
CMAP_SEQUENCIAL = LinearSegmentedColormap.from_list(
    "azul_okabe_ito", ["#ffffff", COR_ANCORA],
)

plt.rcParams["font.family"] = "DejaVu Sans"


COR_FRAC_MIN = 0.18  # piso de saturação: evita bolhas quase brancas (invisíveis sobre o fundo)


def cor_escala(frac: float, cmap) -> tuple:
    """Mapeia frac (0-1) na escala, com piso de saturação para o valor mínimo
    não desaparecer contra o fundo branco da figura."""
    return cmap(COR_FRAC_MIN + (1 - COR_FRAC_MIN) * frac)


def cor_texto_sobre(hex_cor: str) -> str:
    """Réplica da regra de contraste usada nas demais figuras da tese:
    branco se luminância < 0,55, cinza escuro caso contrário."""
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
        for j, ano in enumerate(anos):
            total = matriz.loc[grupo, ano]
            frac = (total - vmin) / (vmax - vmin)
            tamanho = tamanho_min + frac * (tamanho_max - tamanho_min)
            cor = cor_escala(frac, CMAP_SEQUENCIAL)

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

    # legenda de tamanho e cor combinadas (cada bolha de referência usa a cor
    # que a escala sequencial atribui ao próprio valor, e não uma cor neutra:
    # tamanho e cor da legenda seguem juntos a mesma escala do gráfico).
    # Escala de tamanho reduzida (não literalmente igual à do gráfico) só para
    # caber sem sobrepor o texto — a proporção relativa entre os pontos é
    # preservada; a cor de cada ponto é a exata da escala, sem redução.
    legenda_tamanho_min, legenda_tamanho_max = 90, 700
    handles_tamanho = []
    for valor_ref in (10, 30, 60):
        frac = (valor_ref - vmin) / (vmax - vmin)
        frac = min(max(frac, 0), 1)
        tamanho = legenda_tamanho_min + frac * (legenda_tamanho_max - legenda_tamanho_min)
        handles_tamanho.append(ax.scatter(
            [], [], s=tamanho, color=cor_escala(frac, CMAP_SEQUENCIAL), edgecolors="white",
            linewidths=1.5, label=f"{valor_ref} publicações",
        ))
    legenda_tamanho = ax.legend(
        handles=handles_tamanho, loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=3, frameon=False, fontsize=10, labelcolor=COR_TEXTO,
        columnspacing=3.0, handletextpad=1.0,
        title="Tamanho e cor da bolha", title_fontsize=10,
    )

    nota_rodape = (
        "Tamanho e cor da bolha = número de publicações no ano (escala sequencial branco-azul, "
        "não viridis). A cor não identifica o grupo de pesquisa, já dado pelo eixo y. "
        "Escala ancorada em matiz distinto da usada na Figura 12 (composição de equipe, "
        "escala branco-vermelho), para diferenciar visualmente as duas matrizes de bolhas."
    )
    fig.text(0.02, -0.18, nota_rodape, fontsize=8.5, color=COR_NOTA, style="italic", ha="left", va="top")

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
