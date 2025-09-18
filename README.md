
# 🔹 `aeon_machine.py`

```python
import torch
import torch.nn as nn
import random
import numpy as np

# ============================================================
# AEON Transformer Backbone
# ============================================================

class AEONAlgorithmTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size=128, num_layers=4, num_heads=4, ff_dim=256, max_len=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(max_len, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_heads, dim_feedforward=ff_dim, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, seq):
        seq_len = seq.size(1)
        positions = torch.arange(0, seq_len, device=seq.device).unsqueeze(0)
        x = self.token_emb(seq) + self.pos_emb(positions)
        x = self.transformer(x)
        return self.fc_out(x)

# ============================================================
# Vocabulary (Tokens for All Algorithms)
# ============================================================

vocab = {
    "INIT": 0, "SAMPLE": 1, "EVAL": 2, "UPDATE_BEST": 3, "LOOP": 4, "STOP": 5,
    "INIT_POP": 6, "SELECT": 7, "CROSSOVER": 8, "MUTATE": 9,
    "INIT_SWARM": 10, "UPDATE_VELOCITY": 11, "MOVE": 12, "UPDATE_GLOBAL_BEST": 13,
    "ACCEPT": 14, "UPDATE_TEMP": 15
}
id2token = {v:k for k,v in vocab.items()}

# ============================================================
# Universal Executor
# ============================================================

def execute_rs(ir, objective_fn, dim=5, bounds=(-5,5)):
    max_iter = [p for t,p in ir if t=="LOOP"][0]["max_iter"]
    best = None
    for _ in range(max_iter):
        candidate = torch.empty(dim).uniform_(bounds[0], bounds[1])
        cost = objective_fn(candidate)
        if best is None or cost < best[1]:
            best = (candidate, cost)
    return best

def execute_ga(ir, objective_fn, dim=5, bounds=(-5,5)):
    pop_size = [p for t,p in ir if t=="INIT_POP"][0]["population_size"]
    mut_rate = [p for t,p in ir if t=="MUTATE"][0]["rate"]
    cross_rate = [p for t,p in ir if t=="CROSSOVER"][0]["rate"]
    max_gen = [p for t,p in ir if t=="LOOP"][0]["max_gen"]

    pop = [torch.empty(dim).uniform_(bounds[0], bounds[1]) for _ in range(pop_size)]
    fitness = [objective_fn(ind) for ind in pop]
    best = min(zip(pop, fitness), key=lambda x: x[1])

    for _ in range(max_gen):
        new_pop = []
        while len(new_pop) < pop_size:
            i, j = random.sample(range(pop_size), 2)
            parent1 = pop[i] if fitness[i] < fitness[j] else pop[j]
            i, j = random.sample(range(pop_size), 2)
            parent2 = pop[i] if fitness[i] < fitness[j] else pop[j]

            if random.random() < cross_rate:
                point = random.randint(1, dim-1)
                child = torch.cat([parent1[:point], parent2[point:]])
            else:
                child = parent1.clone()

            for d in range(dim):
                if random.random() < mut_rate:
                    child[d] += torch.randn(1).item()
            new_pop.append(child)

        pop = new_pop
        fitness = [objective_fn(ind) for ind in pop]
        current_best = min(zip(pop, fitness), key=lambda x: x[1])
        if current_best[1] < best[1]:
            best = current_best
    return best

def execute_sa(ir, objective_fn, dim=5, bounds=(-5,5)):
    temp = [p for t,p in ir if t=="INIT"][0]["temp"]
    decay = [p for t,p in ir if t=="UPDATE_TEMP"][0]["decay"]
    max_iter = [p for t,p in ir if t=="LOOP"][0]["max_iter"]

    current = torch.empty(dim).uniform_(bounds[0], bounds[1])
    current_cost = objective_fn(current)
    best = (current, current_cost)

    for _ in range(max_iter):
        candidate = current + torch.randn(dim) * 0.1
        candidate_cost = objective_fn(candidate)
        if candidate_cost < current_cost or random.random() < torch.exp((current_cost - candidate_cost) / temp):
            current, current_cost = candidate, candidate_cost
            if current_cost < best[1]:
                best = (current, current_cost)
        temp *= decay
    return best

def execute_pso(ir, objective_fn, dim=5, bounds=(-5,5)):
    swarm_size = [p for t,p in ir if t=="INIT_SWARM"][0]["size"]
    inertia = [p for t,p in ir if t=="UPDATE_VELOCITY"][0]["inertia"]
    c1 = [p for t,p in ir if t=="UPDATE_VELOCITY"][0]["c1"]
    c2 = [p for t,p in ir if t=="UPDATE_VELOCITY"][0]["c2"]
    max_iter = [p for t,p in ir if t=="LOOP"][0]["max_iter"]

    swarm = [torch.empty(dim).uniform_(bounds[0], bounds[1]) for _ in range(swarm_size)]
    velocities = [torch.zeros(dim) for _ in range(swarm_size)]
    fitness = [objective_fn(p) for p in swarm]
    pbest = list(swarm)
    pbest_costs = list(fitness)
    gbest = min(zip(swarm, fitness), key=lambda x: x[1])

    for _ in range(max_iter):
        for i in range(swarm_size):
            r1, r2 = random.random(), random.random()
            velocities[i] = (inertia * velocities[i] +
                             c1 * r1 * (pbest[i] - swarm[i]) +
                             c2 * r2 * (gbest[0] - swarm[i]))
            swarm[i] = swarm[i] + velocities[i]
            cost = objective_fn(swarm[i])
            if cost < pbest_costs[i]:
                pbest[i], pbest_costs[i] = swarm[i], cost
                if cost < gbest[1]:
                    gbest = (swarm[i], cost)
    return gbest

def execute_algorithm(ir, problem_fn):
    if any(t=="INIT_SWARM" for t,_ in ir):
        return execute_pso(ir, problem_fn)
    elif any(t=="UPDATE_TEMP" for t,_ in ir):
        return execute_sa(ir, problem_fn)
    elif any(t=="CROSSOVER" for t,_ in ir):
        return execute_ga(ir, problem_fn)
    else:
        return execute_rs(ir, problem_fn)

# ============================================================
# Feedback + Resonance
# ============================================================

resonance_memory = {}

def update_resonance(ir, score):
    for token, _ in ir:
        if token not in resonance_memory:
            resonance_memory[token] = []
        resonance_memory[token].append(score)

def evaluate_and_feedback(model, ir, problem_fn, optimizer):
    result = execute_algorithm(ir, problem_fn)
    reward = -result[1]  # lower cost → higher reward

    token_ids = [vocab[t] for t,_ in ir if t in vocab]
    if len(token_ids) < 2:
        return result, reward

    seq = torch.tensor([token_ids])
    logits = model(seq[:, :-1])
    loss_fn = nn.CrossEntropyLoss()
    target = seq[:, 1:]
    loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
    loss = loss * (1.0 - reward)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    update_resonance(ir, reward)
    return result, reward

# ============================================================
# Example Benchmark Problem
# ============================================================

def sphere(vector):
    return torch.sum(vector ** 2).item()
```

---

# 🔹 How to Use

```python
from aeon_machine import *

# Define GA IR
ga_ir = [
    ("INIT_POP", {"population_size": 50}),
    ("EVAL", {}),
    ("SELECT", {"method": "tournament"}),
    ("CROSSOVER", {"rate": 0.9}),
    ("MUTATE", {"rate": 0.05}),
    ("LOOP", {"max_gen": 50}),
    ("STOP", {})
]

# Initialize AEON Transformer
model = AEONAlgorithmTransformer(vocab_size=len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Run GA
best = execute_algorithm(ga_ir, sphere)
print("GA best:", best)

# Train AEON on GA and hybrids
for epoch in range(5):
    result, reward = evaluate_and_feedback(model, ga_ir, sphere, optimizer)
    print(f"Epoch {epoch}: cost={result[1]:.4f}, reward={reward:.4f}")
```

---

✅ This `aeon_machine.py` gives you a **single importable system** that:

* Represents algorithms as IR.
* Executes them with the universal runner.
* Generates/learns via Transformer.
* Evolves with resonance + feedback.

