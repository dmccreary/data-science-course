---
title: Neural Network Architecture Builder
description: Build a fully connected neural network by adding layers and setting their sizes, and watch the diagram, the PyTorch nn.Sequential code, and the parameter count change together.
image: /sims/neural-network-architecture-builder/neural-network-architecture-builder.png
og:image: /sims/neural-network-architecture-builder/neural-network-architecture-builder.png
twitter:image: /sims/neural-network-architecture-builder/neural-network-architecture-builder.png
social:
   cards: false
quality_score: 0
---

# Neural Network Architecture Builder

<iframe src="main.html" height="632" width="100%" scrolling="no"></iframe>

[Run the Neural Network Architecture Builder MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The diagram shows a fully connected network: an input layer (green), one to five hidden layers (blue), and an output layer (orange). Every neuron is joined to every neuron in the next layer. A layer too large to draw is shown with a gap of three dots, and the number above it gives its true size.

Each set of connections between two layers is one `nn.Linear(n_in, n_out)` layer. It has one weight for every connection and one bias for every neuron that receives them, so its parameter count is $n_{in} \times n_{out} + n_{out}$. The number under each set of connections is that count, the code panel writes the same arithmetic as a comment on each line, and the bars compare the layers. The total is the sum over all layers, which is the value `sum(p.numel() for p in model.parameters())` returns for the code shown.

The default network is the chapter's example with layer sizes [4, 8, 4, 1] and $40 + 36 + 5 = 81$ parameters. The code puts ReLU after every hidden layer and no activation after the output layer. ReLU has no parameters, so it does not change the count.

Not included: the forward-pass animation, line thickness for trained weights, and a separate activation menu for each layer. The sim draws and counts architectures and does not train them.

## How to Use

1. Read the default network [4, 8, 4, 1] and check one line of the code by hand, for example $4 \times 8 + 8 = 40$.
2. Click a layer in the diagram to select it. The **Neurons** slider then sets the size of that layer, from 1 to 128.
3. Change the size of a hidden layer and watch two blue numbers, two highlighted code lines, and two blue bars change. A hidden layer is the output of one `nn.Linear` and the input of the next.
4. Press **Add Layer** to insert a hidden layer of 8 neurons after the selected layer, and **Remove Layer** to delete the selected hidden layer. A network can have one to five hidden layers.
5. Use the preset menu to load the Simple, Deep, Wide, and Classification networks and compare their totals.
6. Select the Input or Output layer and change its size to match a data set with a different number of features or classes.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/neural-network-architecture-builder/main.html"
        height="632"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
The artificial neuron with its weights and bias, and the roles of the input, hidden, and output layers.

### Bloom's Taxonomy Level
Create (L6)

### Learning Objective
Students will be able to design a fully connected network that meets a stated input size, output size, and parameter budget, calculate its number of parameters layer by layer, and write it as a PyTorch nn.Sequential model.

### Activities

1. **Calculate** (4 min): Students load the Deep preset [4, 32, 16, 8, 1], compute the parameters of each layer on paper, and check their numbers against the comments in the code panel.
2. **Compare** (5 min): Students compare Wide [4, 128, 1] with Deep [4, 32, 16, 8, 1]. They record which has more parameters, which layer holds the most in each, and why.
3. **Design** (6 min): Students build a network for 10 input features and 3 classes that has at least two hidden layers and fewer than 500 parameters, then copy its nn.Sequential code.

### Assessment
Without the sim, students calculate the number of parameters of a network with layer sizes [10, 32, 16, 1], then decide which change adds more parameters, doubling the first hidden layer or doubling the second, and justify the answer with the formula.

## References

1. PyTorch documentation. [torch.nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html).
2. PyTorch documentation. [torch.nn.Sequential](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html).
3. Wikipedia. [Multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron).
