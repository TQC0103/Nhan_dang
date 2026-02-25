# Reusable LaTeX Report Template

This repository is configured for writing academic reports in LaTeX with an English-first scaffold.

## Included Structure

- `cover.tex`: title page template.
- `main.tex`: root document file.
- `preface.tex`: preface chapter.
- `info.tex`: general report metadata.
- `Glossary.tex`: abbreviations and terms tables.
- `Week1.tex`, `Week2.tex`: sample chapter content.
- `sections/`: modular section scaffold for future expansion.
- `references.bib`: bibliography source (`biblatex` + `biber`).

## Build

Recommended pipeline:

1. `xelatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex`
2. `biber main`
3. `xelatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex`

VS Code LaTeX Workshop is preconfigured with the recipe:

`xelatex synctex -> biber -> xelatex synctex`

## Notes

- Keep labels for figures/tables/equations and cross-reference them in text.
- Keep bibliography keys in `references.bib` synchronized with your citations.
- Place new images under `assets/` (or update `\graphicspath` if needed).
