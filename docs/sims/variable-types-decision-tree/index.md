# Variable Types Decision Tree

This interactive decision tree helps students classify any variable they encounter into its proper type: Continuous, Discrete, Ordinal, or Nominal.

<iframe src="./main.html" height="980px" width="100%" scrolling="no" style="overflow: hidden;"></iframe>

[Run the Variable Types Decision Tree in Full Screen](./main.html){ .md-button .md-button--primary }

## How to Use This Diagram

1. **Start** by examining the variable you want to classify
2. **Ask yourself** the key questions as you follow the flowchart
3. **Hover** over any node to see additional examples and explanations
4. **Click "Test Yourself"** to practice classifying random variables

## The Two Main Categories

### Numerical Data (Can do math with it)
Values that represent quantities or measurements where mathematical operations make sense.

- **Continuous**: Can take any value within a range, including decimals
    - Examples: Height (5.7 ft), Temperature (98.6F), Time (3.5 hours)
- **Discrete**: Can only take specific, countable values (usually whole numbers)
    - Examples: Number of siblings (2), Cars owned (3), Students in class (25)

### Categorical Data (Cannot do math with it)
Values that represent groups or categories where math operations are meaningless.

- **Ordinal**: Categories with a meaningful order or ranking
    - Examples: Grade levels (Freshman < Senior), Satisfaction ratings (Poor < Excellent)
- **Nominal**: Categories with no inherent order
    - Examples: Eye color (blue, brown, green), Blood type (A, B, AB, O)

## Key Decision Questions

| Question | YES means... | NO means... |
|----------|--------------|-------------|
| Can you do math with it? | Numerical data | Categorical data |
| Can it be any value (including decimals)? | Continuous | Discrete |
| Is there a meaningful order? | Ordinal | Nominal |

## Common Mistakes to Avoid

1. **ZIP codes are NOT numerical** - Even though they contain digits, you can't meaningfully add or average them
2. **Age in years can be continuous** - While often recorded as whole numbers, age is actually continuous (you can be 25.5 years old)
3. **Rating scales can be ordinal OR discrete** - A 1-10 pain scale is ordinal (the order matters), but if you're counting how many times something happened, it's discrete

## Related Concepts

- [Data Types Overview](../../chapters/01-intro-to-data-science/)
- [Descriptive Statistics](../../chapters/02-descriptive-statistics/)
