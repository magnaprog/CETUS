"""Format the public class-ranking tables for LaTeX."""


def table(columns, header, rows):
    return "\n".join([
        rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule",
        header + r" \\", r"\midrule",
        *[" & ".join(row) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}", "",
    ])
