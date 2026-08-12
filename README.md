# PhysioShift

Repository for signal processing pipelines and model development.

---

## PyTorch Training Loop Mechanics

This section serves as a technical walkthrough for team members on the training mechanics implemented in `src/toy_training_loop.py`.

### Architectural Flow

```text
Dataset ---> DataLoader ---> Model (Forward Pass) ---> Loss Function ---> Backward Pass ---> Optimizer Step
```

### Key Components

#### 1. Custom Dataset (`torch.utils.data.Dataset`)
Handles dataset loading. Implements two mandatory protocols:
* `__len__()`: Returns total dataset size.
* `__getitem__(idx)`: Extracts a single feature-target tuple at index `idx`.

#### 2. DataLoader (`torch.utils.data.DataLoader`)
Wraps the dataset to provide batching, dataset shuffling, and parallel execution.

#### 3. Neural Network (`torch.nn.Module`)
Defines the graph layers inside `__init__()` and controls data propagation through `forward(x)`.

#### 4. The 5-Step Training Batch Cycle

For each iteration over mini-batches in an epoch:

1. **`optimizer.zero_grad()`**: Resets gradients stored from the previous iteration.
2. **`outputs = model(inputs)`**: Executes the forward pass to compute outputs.
3. **`loss = criterion(outputs, targets)`**: Calculates the discrepancy between prediction and target.
4. **`loss.backward()`**: Triggers Autograd to compute gradients across all network parameters.
5. **`optimizer.step()`**: Adjusts network weights according to computed gradients and the optimization rule.

---

### Verification Run

Execute the script to verify pipeline mechanics:

```bash
python src/toy_training_loop.py
```