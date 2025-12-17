# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data science education website built with MkDocs Material. The site serves as an interactive 15-week course on "AI Based Data Science with Python" featuring MicroSims (interactive simulations) and comprehensive learning materials for teaching data science concepts to high school and college students.

**Site URL**: https://dmccreary.github.io/data-science-course/
**License**: Creative Commons BY-NC-SA 4.0 (non-commercial, attribution required)

## Site Architecture

- **Documentation Source**: `docs/` - All Markdown content and assets
- **Generated Site**: `site/` - Built HTML files (should not be edited directly)
- **Python Scripts**: `src/` - Data processing utilities and example scripts
- **Configuration**: `mkdocs.yml` - Site configuration and navigation

### Key Content Structure

- `docs/chapters/` - 15 course chapters (00-setup through 15-projects) covering foundations through capstone
- `docs/labs/` - Hands-on lab exercises (dataframes, data exploration, statistics)
- `docs/sims/` - Interactive MicroSims with `index.md` (documentation) and `.html` files (interactive components)
  - `learning-graph/` - vis.js network visualization of concept dependencies
  - `template/` - Base template for creating new MicroSims
  - Various statistical simulations (normal-dice-distribution, bell-curve, least-squares, etc.)
- `docs/prompts/` - AI-generated content prompts AND CSV data files for concept management
  - `ds-concepts.csv` - Core concept definitions (ConceptID, ConceptLabel, Dependencies, TaxonomyID)
  - `dependencies.csv` - Concept prerequisite relationships
  - `enumerated-concepts.csv` - Categorized concept listings
- `src/csv-to-json/` - Python utility to convert CSV data to vis.js JSON format (graph-data.csv → graph-data.json)
- `src/line-plot/` - Simple matplotlib example scripts
- `site/` - Auto-generated HTML output (never edit directly)

## Development Commands

### Primary MkDocs Commands

```bash
# Run development server with live reload (main development command)
mkdocs serve

# Build the site locally (generates site/ directory)
mkdocs build

# Deploy to GitHub Pages (builds and pushes to gh-pages branch)
mkdocs gh-deploy
```

### Testing and Validation
- No formal test suite - validation is done by building successfully with `mkdocs build`
- Interactive components are tested manually by running `mkdocs serve` and checking functionality
- MicroSims require browser testing for JavaScript functionality

### Python Environment Setup

The course uses conda for environment management:

```bash
# Create and activate data science environment
conda create -n ds python=3
conda activate ds

# Install required packages
pip install matplotlib pandas numpy mkdocs mkdocs-material
```

For MkDocs Material with social imaging support:
```bash
pip install mkdocs "mkdocs-material[imaging]"
```

### Data Processing Scripts

```bash
# Convert CSV concept data to JSON for visualizations
cd src/csv-to-json
python csv-to-json.py

# Run example plotting scripts
cd src/line-plot
python line-plot.py
```

## Content Management

### Educational Content Structure

The course follows a 15-week curriculum progression:
1. Setup → Foundations → Data Exploration → Data Visualization → Statistics
2. Linear Regression → Model Evaluation → Multiple Regression → NumPy
3. Non-linear Models → Regularization → Machine Learning → Neural Networks
4. PyTorch → Advanced Evaluation → Capstone Projects

**Navigation Structure**: All pages must be added to `mkdocs.yml` nav section to appear in site navigation. The configuration uses Material for MkDocs theme with features: code copy, navigation expand/path/prune/indexes, toc follow, navigation top/footer, and content action edit.

### MicroSims Architecture

Interactive JavaScript simulations embedded throughout the course:
- **Learning Graph**: `docs/sims/learning-graph/` - vis.js network visualization of concept dependencies
- **Statistical Simulations**: Various probability and distribution visualizations
- **Template Structure**: `docs/sims/template/` - Base template for new MicroSims

**MicroSim Development Pattern**:
- Each sim has `index.md` (documentation) and `main.html` (interactive component)
- JavaScript files handle simulation logic and visualization
- Data flows: CSV → Python processing → JSON → JavaScript visualization
- Standard libraries: vis.js for networks, native HTML5 Canvas for custom graphics

### Content Creation Workflow

1. Write Markdown content in `docs/`
2. Update `mkdocs.yml` navigation if adding new pages
3. Use `mkdocs serve` to preview changes locally
4. Build with `mkdocs build` to verify no errors
5. Deploy with `mkdocs gh-deploy` when ready

### Data Science Concepts Management

Concept definitions and dependencies are managed through CSV files in `docs/prompts/`:
- `ds-concepts.csv` - Core concept definitions with ConceptID, ConceptLabel, Dependencies, TaxonomyID
- `dependencies.csv` - Concept prerequisite relationships
- `enumerated-concepts.csv` - Categorized concept listings

**Data Processing Workflow** (CSV → JSON → Visualization):
1. Edit CSV files in `docs/prompts/` to update concept data
2. Run `cd src/csv-to-json && python csv-to-json.py` to convert CSV to JSON
   - Script reads `graph-data.csv` and outputs `graph-data.json`
   - Creates vis.js compatible format with nodes (id, label, group) and edges (from, to)
   - Dependencies in CSV are pipe-separated (e.g., "1|2|3" for multiple prerequisites)
3. Generated JSON is consumed by `docs/sims/learning-graph/view-graph.html` for interactive visualization
4. Changes to CSV require re-running the converter and rebuilding the site to see updates

## Development Environment

- **Python 3.8+** required for all scripts and examples
- **Conda environment** recommended for dependency management
- **MkDocs Material theme** with social imaging capabilities
- **Visual Studio Code** recommended IDE (as mentioned in setup docs)

## Site Deployment

The site is deployed to GitHub Pages at https://dmccreary.github.io/data-science-course/

**Deployment process**:
1. Ensure all content changes are committed to main branch
2. Run `mkdocs gh-deploy` to build and push to gh-pages branch
3. Site updates automatically reflect at the public URL
4. The command builds the site, commits to gh-pages branch, and pushes automatically

**Important**: Never edit files in the `site/` directory directly - they are auto-generated. Always edit source files in `docs/` and use `mkdocs build` or `mkdocs gh-deploy`.

## Key Libraries and Dependencies

### Core Site Technology
- **MkDocs Material**: Static site generator with modern responsive theme
- **vis.js**: Interactive network/graph visualization library for concept dependency graphs
- **Native HTML5/CSS3/JavaScript**: MicroSims use vanilla web technologies, no heavy frameworks

### Python Dependencies
- **Python standard libraries**: csv, json for data processing utilities
- **matplotlib**: Basic plotting examples referenced in course content
- **pandas, numpy**: Referenced in course materials (students install as needed)
- **No requirements.txt**: Dependencies managed via conda or manual pip installation

### Build and Deployment
- **GitHub Pages**: Automatic deployment via `mkdocs gh-deploy` command
- **No CI/CD**: Manual build and deploy process, no automated testing
<<<<<<< HEAD

## Details
When generating chapter content, the chapter-content-generator skill will
add diagram/microsim placeholders using the <details> XML element.
Do not put leading spaces inside the <details> elements.  Do not indent the text within the <details> element.
- remember to use the p5.js's built-in textWrap(WORD) when wrapping text in a p5.js microsim
=======
- **No requirements.txt or package.json**: Dependencies installed manually via conda/pip

## Important Architectural Notes

### File Editing Guidelines
- **Never edit `site/` directory**: Auto-generated by MkDocs, changes will be overwritten
- **Always edit source files in `docs/`**: These are the source of truth
- **Update `mkdocs.yml` navigation**: When adding new pages, add them to the `nav` section or they won't appear
- **MicroSim structure**: Each sim has `index.md` (docs) and typically `.html` files (interactive UI)

### Data Flow for Learning Graph
1. **Source**: `docs/prompts/ds-concepts.csv` (and related CSV files)
2. **Processing**: `src/csv-to-json/csv-to-json.py` converts to `graph-data.json`
3. **Visualization**: `docs/sims/learning-graph/view-graph.html` renders interactive graph using vis.js
4. **Update workflow**: Edit CSV → Run Python script → Rebuild site → Deploy

### MkDocs Material Theme Configuration
- Theme configured in `mkdocs.yml` with custom CSS at `docs/css/extra.css`
- Logo: `docs/img/logo-192.png`, Favicon: `docs/img/favicon.ico`
- Color scheme: Primary blue, Accent orange
- Search plugin enabled by default
- Social plugin commented out (requires imaging dependencies)
>>>>>>> 47e3019 (udates)
