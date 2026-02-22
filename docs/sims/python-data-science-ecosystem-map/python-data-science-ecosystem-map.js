// Python Data Science Ecosystem Map
// vis-network hub-and-spoke showing major library relationships
// Bloom Level: Understand (L2) - hover to see descriptions and import examples
// MicroSim template version 2026.02

document.addEventListener('DOMContentLoaded', function () {
  const mainEl = document.querySelector('main');
  mainEl.style.position = 'relative';
  mainEl.style.width = '100%';
  mainEl.style.height = '620px';

  // Network container
  const container = document.createElement('div');
  container.id = 'network-container';
  container.style.cssText = 'width:100%;height:560px;border:1px solid #ccc;background:#1e1e2e;border-radius:6px;';
  mainEl.appendChild(container);

  // Info panel
  const infoPanel = document.createElement('div');
  infoPanel.id = 'info-panel';
  infoPanel.style.cssText = `
    position:absolute;top:10px;right:10px;width:240px;background:rgba(30,30,46,0.95);
    color:#cdd6f4;padding:12px;border-radius:8px;border:1px solid #45475a;
    font-family:monospace;font-size:12px;line-height:1.5;display:block;
  `;
  infoPanel.innerHTML = '<b style="color:#cba6f7">Python Data Science Ecosystem</b><br><br>Hover a node to see its description and import statement.<br><br>Node colors indicate library category.';
  mainEl.appendChild(infoPanel);

  // Legend
  const legend = document.createElement('div');
  legend.style.cssText = 'position:absolute;top:10px;left:10px;background:rgba(30,30,46,0.92);color:#cdd6f4;padding:8px 12px;border-radius:6px;border:1px solid #45475a;font-size:11px;';
  legend.innerHTML = `
    <b>Legend</b><br>
    <span style="color:#f9e2af">●</span> Core &nbsp;
    <span style="color:#89dceb">●</span> Data &nbsp;
    <span style="color:#a6e3a1">●</span> Visualization<br>
    <span style="color:#cba6f7">●</span> ML &nbsp;
    <span style="color:#f38ba8">●</span> Deep Learning &nbsp;
    <span style="color:#bac2de">●</span> Utilities
  `;
  mainEl.appendChild(legend);

  // Node info
  const nodeInfo = {
    1:  { name: 'Python', desc: 'The language itself. All data science libraries are written for and run by Python.', imp: '# Python is the foundation — no import needed!' },
    2:  { name: 'NumPy', desc: 'Numerical Python. Provides fast, efficient n-dimensional arrays. The backbone of nearly all scientific computing in Python.', imp: 'import numpy as np' },
    3:  { name: 'pandas', desc: 'Panel Data. Provides DataFrame — a table-like structure for loading, cleaning, and analyzing data.', imp: 'import pandas as pd' },
    4:  { name: 'matplotlib', desc: 'The original Python plotting library. Create static, animated, or interactive visualizations.', imp: 'import matplotlib.pyplot as plt' },
    5:  { name: 'seaborn', desc: 'Statistical data visualization built on matplotlib. Makes beautiful charts with less code.', imp: 'import seaborn as sns' },
    6:  { name: 'Plotly', desc: 'Interactive web-based charts. Supports scatter, bar, line, 3D, maps, and more.', imp: 'import plotly.express as px' },
    7:  { name: 'scikit-learn', desc: 'The go-to ML library. Regression, classification, clustering, preprocessing — all with a consistent API.', imp: 'from sklearn.linear_model import LinearRegression' },
    8:  { name: 'SciPy', desc: 'Scientific computing — statistics, optimization, signal processing, linear algebra.', imp: 'from scipy import stats' },
    9:  { name: 'PyTorch', desc: 'Deep learning framework from Meta. Flexible, research-friendly, and widely used in AI.', imp: 'import torch\nimport torch.nn as nn' },
    10: { name: 'TensorFlow', desc: 'Deep learning framework from Google. Production-ready with Keras high-level API.', imp: 'import tensorflow as tf' },
    11: { name: 'Jupyter', desc: 'The interactive notebook environment. Write code, see output, add explanations — all in one document.', imp: '# Run in terminal:\njupyter notebook' },
    12: { name: 'XGBoost', desc: 'Extreme Gradient Boosting. High-performance ML for structured/tabular data — often wins ML competitions.', imp: 'import xgboost as xgb' }
  };

  const nodes = new vis.DataSet([
    { id: 1,  label: 'Python',       group: 'core',    size: 38, x: 0,    y: 0 },
    { id: 2,  label: 'NumPy',        group: 'core',    size: 32, x: 0,    y: -180 },
    { id: 3,  label: 'pandas',       group: 'data',    size: 28, x: 220,  y: -80 },
    { id: 4,  label: 'matplotlib',   group: 'viz',     size: 26, x: 200,  y: 120 },
    { id: 5,  label: 'seaborn',      group: 'viz',     size: 22, x: 340,  y: 60 },
    { id: 6,  label: 'Plotly',       group: 'viz',     size: 22, x: 340,  y: 180 },
    { id: 7,  label: 'scikit-learn', group: 'ml',      size: 28, x: -220, y: -80 },
    { id: 8,  label: 'SciPy',        group: 'util',    size: 22, x: -180, y: 120 },
    { id: 9,  label: 'PyTorch',      group: 'deep',    size: 26, x: -100, y: -260 },
    { id: 10, label: 'TensorFlow',   group: 'deep',    size: 26, x: 100,  y: -280 },
    { id: 11, label: 'Jupyter',      group: 'util',    size: 22, x: 0,    y: 220 },
    { id: 12, label: 'XGBoost',      group: 'ml',      size: 20, x: -340, y: -100 }
  ]);

  const edges = new vis.DataSet([
    { from: 1, to: 2,  label: 'core engine', dashes: false },
    { from: 2, to: 3,  label: 'built on' },
    { from: 2, to: 4,  label: 'uses arrays' },
    { from: 2, to: 7,  label: 'data format' },
    { from: 2, to: 8,  label: 'uses arrays' },
    { from: 2, to: 9,  label: 'similar API' },
    { from: 2, to: 10, label: 'similar API' },
    { from: 4, to: 5,  label: 'built on' },
    { from: 3, to: 5,  label: 'data source' },
    { from: 7, to: 12, label: 'similar API' },
    { from: 1, to: 11, label: 'interface' }
  ]);

  const groupColors = {
    core:  { background: '#f9e2af', border: '#e4b640', font: { color: '#1e1e2e' } },
    data:  { background: '#89dceb', border: '#30b0c7', font: { color: '#1e1e2e' } },
    viz:   { background: '#a6e3a1', border: '#4ec940', font: { color: '#1e1e2e' } },
    ml:    { background: '#cba6f7', border: '#9b60e0', font: { color: '#1e1e2e' } },
    deep:  { background: '#f38ba8', border: '#d0406c', font: { color: '#1e1e2e' } },
    util:  { background: '#bac2de', border: '#8090b0', font: { color: '#1e1e2e' } }
  };

  const options = {
    nodes: {
      shape: 'dot',
      font: { size: 13, color: '#cdd6f4', bold: { color: '#f9e2af' } },
      borderWidth: 2
    },
    edges: {
      color: { color: '#585b70', highlight: '#f9e2af' },
      font: { size: 10, color: '#a6adc8', align: 'middle' },
      smooth: { type: 'continuous' },
      arrows: { to: { enabled: true, scaleFactor: 0.6 } }
    },
    groups: groupColors,
    physics: {
      enabled: false
    },
    interaction: {
      hover: true,
      tooltipDelay: 0,
      zoomView: false,
      dragView: true
    },
    layout: { improvedLayout: false }
  };

  const network = new vis.Network(container, { nodes, edges }, options);

  network.on('hoverNode', function (params) {
    let nodeId = params.node;
    let info = nodeInfo[nodeId];
    if (info) {
      infoPanel.innerHTML = `
        <b style="color:#cba6f7;font-size:14px">${info.name}</b><br><br>
        <span style="color:#cdd6f4">${info.desc}</span><br><br>
        <code style="color:#a6e3a1;background:rgba(0,0,0,0.3);padding:4px 6px;border-radius:4px;display:block;white-space:pre">${info.imp}</code>
      `;
    }
  });

  network.on('blurNode', function () {
    infoPanel.innerHTML = '<b style="color:#cba6f7">Python Data Science Ecosystem</b><br><br>Hover a node to see its description and import statement.';
  });
});
