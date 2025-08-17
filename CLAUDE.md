# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data science education website built with MkDocs Material. The site serves as an interactive course on "AI Based Data Science with Python" featuring MicroSims (interactive simulations) and comprehensive learning materials for teaching data science concepts to high school and college students.

## Site Architecture

- **Documentation Source**: `docs/` - All Markdown content and assets
- **Generated Site**: `site/` - Built HTML files (should not be edited directly)
- **Python Scripts**: `src/` - Data processing utilities and example scripts
- **Configuration**: `mkdocs.yml` - Site configuration and navigation

### Key Content Structure

- `docs/concepts/` - Data science concept definitions, dependencies, and taxonomies with CSV data files
- `docs/chapters/` - Course chapters (setup, matplotlib-vs-plotly, etc.)
- `docs/sims/learning-graph/` - Interactive learning graph visualization using vis.js
- `docs/prompts/` - AI-generated content prompts for FAQs, glossary, and course descriptions
- `src/csv-to-json/` - Python utility to convert concept CSV data to JSON for visualizations
- `src/line-plot/` - Simple matplotlib example scripts

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

The course follows a 10-week curriculum progression:
1. Foundations → Statistical concepts → Linear regression → Model evaluation
2. Multiple regression → NumPy → Non-linear models → PyTorch → Capstone projects

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

Concept definitions and dependencies are managed through CSV files in `docs/prompts/` (note: not `docs/concepts/`):
- `ds-concepts.csv` - Core concept definitions with ConceptID, ConceptLabel, Dependencies, TaxonomyID
- `dependencies.csv` - Concept prerequisite relationships  
- `enumerated-concepts.csv` - Categorized concept listings

**Data Processing Workflow**:
1. Edit CSV files in `docs/prompts/` to update concept data
2. Run `cd src/csv-to-json && python csv-to-json.py` to convert CSV to JSON
3. Generated JSON feeds into vis.js learning graph visualization
4. CSV format: pipe-separated dependencies (e.g., "1|2|3" for multiple prerequisites)

## Development Environment

- **Python 3.8+** required for all scripts and examples
- **Conda environment** recommended for dependency management
- **MkDocs Material theme** with social imaging capabilities
- **Visual Studio Code** recommended IDE (as mentioned in setup docs)

## Site Deployment

The site is deployed to GitHub Pages at https://dmccreary.github.io/data-science-course/

Deployment process:
1. Ensure all content changes are committed to the repository
2. Run `mkdocs gh-deploy` to build and push to gh-pages branch
3. Site updates automatically reflect at the public URL

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