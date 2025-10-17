# Optimizing Prompts with Reinforcement Learning

A practical implementation of Thompson Sampling and evolutionary prompt optimization for LLM applications. This repository demonstrates how to move from manual prompt engineering to a self-improving system that continuously learns which prompts work best.

## Overview

As organizations scale their use of LLMs, prompt quality directly impacts conversion rates, handling time, and communication effectiveness. This codebase implements a two-stage reinforcement learning approach:

1. **Thompson Sampling (Multi-Armed Bandit)**: Efficiently learn which prompts perform best by dynamically allocating traffic to winners while preserving principled exploration
2. **Evolutionary Extension**: Periodically retire weak prompts and generate informed variations of strong ones to keep your prompt portfolio adaptive

## Key Features

- **Sample Efficient**: Learn from minimal interactions using Bayesian updating (Beta-Binomial conjugate priors)
- **Uncertainty-Aware**: Track both estimated performance and confidence through Beta posterior distributions
- **Self-Improving**: Automatically evolve prompts by retiring low-performers and mutating winners
- **Interpretable**: Clear visualizations of learning dynamics and distribution evolution

## How It Works

### Stage 1: Multi-Armed Bandit Learning

Each prompt starts with a uniform Beta(1,1) prior representing complete uncertainty. As interactions occur:
- Successes and failures update the Beta posterior: `Beta(successes + 1, failures + 1)`
- Thompson Sampling selects prompts by drawing one sample from each posterior and choosing the maximum
- Traffic naturally shifts toward high-performers while maintaining exploration

### Stage 2: Evolutionary Optimization

Every N interactions (e.g., daily):
- Compare each prompt's Beta distribution against the current best using analytical probability calculations
- Retire prompts where `P(prompt > best) < 5%` (configurable threshold)
- Generate new prompts by mutating the best performer via LLM
- Initialize newcomers with cautious priors that inherit the winner's estimated rate but reduced confidence

## Quick Start

```python
# Install dependencies
pip install numpy pandas matplotlib scipy openai tqdm

# Run the notebook
jupyter notebook prompt_optimization_with_evolution.ipynb
```

The notebook contains three cells:
1. **Visualization utilities**: Plotting functions for cumulative success and Beta distributions
2. **Baseline Thompson Sampling**: 30-day simulation with 5 initial prompts
3. **Evolutionary extension**: Continuous learning with prompt mutation and retirement

## Configuration

Key parameters in the code:
- `RETIREMENT_THRESHOLD = 0.05`: Probability threshold for retiring weak prompts
- `calls_per_day, days = 10, 30`: Simulation duration (300 total interactions)
- `OPENAI_MODEL_NAME = "gpt-4.1-mini"`: Model for generating prompt variations
- Beta distribution comparison uses analytical CDF methods for exact probability calculations

## Results

The system typically demonstrates:
- **Rapid convergence**: Strong prompts receive 60-80% of traffic after ~100 trials
- **Explicit uncertainty**: Wide posteriors for untested prompts, sharp peaks for proven winners
- **Living portfolio**: New variations emerge from successful parents while weak branches die out

## Technical Details

### Bayesian Framework
- **Prior**: Beta(1,1) = Uniform[0,1] for new prompts
- **Likelihood**: Bernoulli(p) for each success/failure
- **Posterior**: Beta(α, β) where α = successes + 1, β = failures + 1
- **Selection**: Thompson Sampling via posterior sampling

### Distribution Comparison
Prompt retirement uses exact Beta distribution comparison rather than approximate methods:
```python
# Calculate P(current < best) using cumulative distribution functions
# Retire if P(current > best) < threshold
```

### Prompt Mutation
- Inherits winner's estimated success rate but with reduced effective sample size
- Uses LLM to generate semantic variations while preserving intent
- Validates uniqueness before deployment

## Extensions

Natural directions for production use:
- **Contextual bandits**: Condition on user/channel features
- **Delayed rewards**: Multi-step attribution (click → meeting → revenue)
- **Safety constraints**: Brand and ethical guardrails on generated prompts
- **Offline evaluation**: Pre-screen mutations before live traffic
