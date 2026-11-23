# TOUMAII research project page

A buildless, responsive project page for https://github.com/tum-ai/toumaii-gen.
Research content follows the supplied TOUMAII manuscript and generator repository. The paper-first layout is inspired by the user-provided MeshFeat and HASSL pages.

## Preview

From this folder:

```sh
python3 -m http.server 8765
```

Open http://localhost:8765. No build or dependencies are required. The theme follows the device by default; visitors can choose Light or Dark in the header, or return to System. IBM Plex Sans and IBM Plex Mono load from Google Fonts when available; system fallbacks work offline.

## Deploy in tum-ai/toumaii-gen

GitHub Pages publishes the root of the `gh-pages` branch. In repository Settings → Pages, the source is **Deploy from a branch**, with `gh-pages` and `/(root)` selected.

To update the site, commit and push changes to `gh-pages`. Keep `.nojekyll` at the branch root so GitHub Pages serves the files as-is. The site is served at https://tum-ai.github.io/toumaii-gen/. Relative asset links work at that subpath, and no changes to `main` or the generator code are needed.

## Content and provenance

- Manuscript: `Formatting_Instructions_For_NeurIPS_2026-3/neurips_2026.tex` provided by the user.
- Original composition and reward SVGs copied from that manuscript's figures directory.
- Original camera-ready pipeline PDF linked from the method section.
- Exact count: 4,515,503 retained pairs. Counts and validation results match the supplied manuscript, not unrelated workspace experiments.
- Examples are abridged appendix examples, with their reported ToolRL rewards; generation is not executed in the browser.
- Generator quickstart checked against repository README/config and export schema at source revision recorded in SOURCE_REVISION.txt.
- No invented paper DOI, arXiv identifier, conference acceptance or dataset identifier. Add a public paper link when available.

The data-loading example reads an actual generator DPO export; released Hub configurations may differ and are intentionally delegated to their dataset cards.

## Editing

- `index.html`: research narrative, results, author attribution and links.
- `style.css`: responsive design and motion; respects reduced-motion settings.
- `app.js`: four qualitative examples, figure tabs, keyboard navigation and syntax highlighting for the examples, guides and citation.
- `theme.js`: device-aware theme overrides and sticky-header anchor spacing.
- `assets/`: supplied paper figures and the licensed Toumaï illustration, used as the emblem and favicon.

## Academic revision

Paper title and author/affiliation block; abstract summary; contributions; canonical generation notation; reward definition and filtering details; exact dataset counts; controlled DPO+SFT experimental protocol; limitations; manuscript BibTeX and selected references. No acceptance claim or unverified paper link is added.

The Toumaï illustration is by Sisyphos23, based on Mateus Zica, under CC BY-SA 3.0. See assets/ATTRIBUTION.md and the on-page footer.
