# AI Based Data Science with Python

[![Website](https://img.shields.io/website?url=https%3A%2F%2Fdmccreary.github.io%2Fdata-science-course%2F)](https://dmccreary.github.io/data-science-course/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Deployed-green)](https://dmccreary.github.io/data-science-course/)
[![MkDocs](https://img.shields.io/badge/Made%20with-MkDocs-blue)](https://www.mkdocs.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

An interactive online course on **AI-Based Data Science with Python** featuring comprehensive learning materials, MicroSims (interactive simulations), and hands-on exercises designed for high school and college students.

🌐 **Live Site**: [https://dmccreary.github.io/data-science-course/](https://dmccreary.github.io/data-science-course/)

## 📚 Course Overview

This 15-week course provides a comprehensive introduction to data science using Python and AI tools. The curriculum progresses from foundational concepts to advanced machine learning topics through interactive learning experiences.

### Course Structure
- **Foundations** → Statistical concepts → Linear regression → Model evaluation
- **Multiple regression** → NumPy → Non-linear models → PyTorch → Capstone projects
- **Interactive MicroSims** throughout each chapter
- **Real-world projects** and hands-on exercises

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Basic programming knowledge (helpful but not required)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/dmccreary/data-science-course.git
   cd data-science-course
   ```

2. **Set up Python environment**
   ```bash
   # Using conda (recommended)
   conda create -n ds python=3
   conda activate ds
   
   # Install required packages
   pip install matplotlib pandas numpy mkdocs mkdocs-material
   ```

3. **Run development server**
   ```bash
   # Start local development server with live reload
   mkdocs serve
   ```
   
   The site will be available at `http://localhost:8000`

4. **Build the site**
   ```bash
   # Generate static site files
   mkdocs build
   ```

## 📁 Project Structure

```
data-science-course/
├── docs/                      # Documentation source files
│   ├── chapters/             # Course chapters (00-setup through 15-projects)
│   ├── sims/                 # Interactive MicroSims
│   │   └── learning-graph/   # Learning dependency visualization
│   ├── concepts/             # Data science concept definitions (CSV files)
│   └── prompts/              # AI-generated content prompts
├── src/                      # Python utilities and examples
│   ├── csv-to-json/         # Data processing scripts
│   └── line-plot/           # Matplotlib examples
├── site/                     # Generated HTML (auto-generated, do not edit)
├── mkdocs.yml               # Site configuration
└── CLAUDE.md                # AI assistant instructions
```

## 🎯 Key Features

- **Interactive Learning Graph**: Visualize concept dependencies using vis.js
- **MicroSims**: Hands-on simulations for statistical concepts
- **AI Integration**: Course designed with AI tools for enhanced learning
- **Progressive Curriculum**: 15-week structured learning path
- **Mobile Responsive**: Works on all devices
- **Search Functionality**: Full-text search across all content

## 🛠 Development Commands

```bash
# Start development server
mkdocs serve

# Build static site
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy

# Process concept data
cd src/csv-to-json && python csv-to-json.py

# Run example scripts
cd src/line-plot && python line-plot.py
```

## 🔧 Technologies Used

- **[MkDocs](https://www.mkdocs.org/)** - Static site generator
- **[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)** - Modern documentation theme
- **[Python](https://www.python.org/)** - Programming language and examples
- **[vis.js](https://visjs.org/)** - Interactive network visualizations
- **[GitHub Pages](https://pages.github.com/)** - Static site hosting
- **[Conda](https://conda.io/)** - Environment management

## 📖 Course Content

### Core Chapters
1. **Setup** - Development environment configuration
2. **Foundations** - Basic data science concepts
3. **Data Exploration** - Working with datasets
4. **Data Visualization** - Creating effective visualizations
5. **Statistics** - Statistical analysis fundamentals
6. **Linear Regression** - Predictive modeling basics
7. **Model Evaluation** - Assessing model performance
8. **Multiple Regression** - Advanced regression techniques
9. **NumPy** - Numerical computing with Python
10. **Non-linear Models** - Beyond linear relationships
11. **Regularization** - Preventing overfitting
12. **Machine Learning** - ML fundamentals
13. **Neural Networks** - Deep learning introduction
14. **PyTorch** - Modern ML framework
15. **Capstone Projects** - Real-world applications

### Interactive Features
- **Learning Dependency Graph** - Visualize prerequisite relationships
- **Statistical MicroSims** - Interactive probability distributions
- **Hands-on Exercises** - Practical coding challenges

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:
- Reporting issues
- Suggesting enhancements
- Submitting pull requests
- Code style and standards

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with `mkdocs serve`
5. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](license.md) file for details.

## 🙏 Acknowledgments

This course is built with and inspired by many open source projects and educational resources:

### Core Technologies
- **[MkDocs](https://www.mkdocs.org/)** - Static site generator created by Tom Christie
- **[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)** - Beautiful documentation theme by Martin Donath
- **[Python](https://www.python.org/)** - Programming language by Python Software Foundation
- **[vis.js](https://visjs.org/)** - Dynamic, browser based visualization library
- **[GitHub Pages](https://pages.github.com/)** - Free static site hosting by GitHub

### Python Libraries
- **[NumPy](https://numpy.org/)** - Fundamental package for scientific computing
- **[pandas](https://pandas.pydata.org/)** - Data analysis and manipulation library
- **[Matplotlib](https://matplotlib.org/)** - Comprehensive plotting library
- **[PyTorch](https://pytorch.org/)** - Machine learning framework by Meta AI

### Development Tools
- **[Conda](https://conda.io/)** - Package and environment management by Anaconda Inc.
- **[Visual Studio Code](https://code.visualstudio.com/)** - Code editor by Microsoft
- **[Git](https://git-scm.com/)** - Version control system by Linus Torvalds

### Educational Inspiration
- Open courseware initiatives from MIT, Stanford, and other institutions
- The broader data science and Python communities
- AI/ML educational resources and best practices

## 📞 Contact

- **Author**: Dan McCreary
- **Repository**: [https://github.com/dmccreary/data-science-course](https://github.com/dmccreary/data-science-course)
- **Issues**: [Report bugs or suggest features](https://github.com/dmccreary/data-science-course/issues)

---

*Made with ❤️ for data science education*