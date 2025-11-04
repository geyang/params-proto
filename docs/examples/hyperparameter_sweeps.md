# Hyperparameter Sweeps

Learn how to use params-proto's powerful hyperparameter search and sweep functionality for machine learning experiments.

## Basic Hyperparameter Sweeps

The `Sweep` class allows you to systematically explore different parameter combinations for your experiments.

### Simple Grid Search

```python
from params_proto.v2.proto import ParamsProto, Proto
from params_proto.v2.hyper import Sweep

class ModelConfig(ParamsProto):
    learning_rate = Proto("Learning rate", default=0.001)
    batch_size = Proto("Batch size", default=32)
    model_type = Proto("Model type", default="resnet")

# Create a parameter sweep
sweep = Sweep(ModelConfig).product([
    ModelConfig.learning_rate << [0.001, 0.01, 0.1],
    ModelConfig.batch_size << [16, 32, 64],
    ModelConfig.model_type << ["resnet", "transformer"]
])

print(f"Total combinations: {len(sweep)}")  # 3 × 3 × 2 = 18 combinations

# Iterate through all combinations
for i, config in enumerate(sweep):
    print(f"Experiment {i+1}: lr={ModelConfig.learning_rate}, "
          f"bs={ModelConfig.batch_size}, model={ModelConfig.model_type}")
    
    # Your training code here
    # accuracy = train_model()
    # results.append(accuracy)
```

### Zip Sweep (Paired Parameters)

```python
# Sometimes you want to pair parameters instead of creating all combinations
sweep = Sweep(ModelConfig).zip([
    ModelConfig.learning_rate << [0.001, 0.01, 0.1],
    ModelConfig.batch_size << [16, 32, 64]
])

# This creates 3 combinations: (0.001, 16), (0.01, 32), (0.1, 64)
for config in sweep:
    print(f"Paired: lr={ModelConfig.learning_rate}, bs={ModelConfig.batch_size}")
```

### Chain Sweep (Sequential Experiments)

```python
# Chain multiple sweeps together
sweep1 = Sweep(ModelConfig).product([
    ModelConfig.model_type << ["resnet"],
    ModelConfig.learning_rate << [0.001, 0.01]
])

sweep2 = Sweep(ModelConfig).product([
    ModelConfig.model_type << ["transformer"],
    ModelConfig.learning_rate << [0.0001, 0.001]
])

# Combine sweeps
combined_sweep = sweep1.chain(sweep2)
print(f"Combined experiments: {len(combined_sweep)}")
```

## Advanced Sweep Patterns

### Multiple Configuration Classes

```python
class ModelConfig(ParamsProto, cli_parse=False):
    architecture = Proto("Model architecture", default="resnet50")
    dropout = Proto("Dropout rate", default=0.1)

class TrainingConfig(ParamsProto):
    learning_rate = Proto("Learning rate", default=0.001)
    optimizer = Proto("Optimizer", default="adam")

# Sweep multiple configuration classes together
with Sweep(ModelConfig, TrainingConfig) as sweep:
    with sweep.product:
        ModelConfig.architecture = ["resnet50", "resnet101"]
        ModelConfig.dropout = [0.1, 0.2]
        TrainingConfig.learning_rate = [0.001, 0.01]
        TrainingConfig.optimizer = ["adam", "sgd"]

for config in sweep:
    print(f"Model: {ModelConfig.architecture}, dropout: {ModelConfig.dropout}")
    print(f"LR: {TrainingConfig.learning_rate}, optimizer: {TrainingConfig.optimizer}")
    print("---")
```

### Conditional Sweeps

```python
class AdaptiveConfig(ParamsProto):
    model_type = Proto("Model type", default="cnn")
    learning_rate = Proto("Learning rate", default=0.001)
    
# Different learning rates for different model types
def create_conditional_sweep():
    sweeps = []
    
    # CNN models with higher learning rates
    cnn_sweep = Sweep(AdaptiveConfig).product([
        AdaptiveConfig.model_type << ["cnn"],
        AdaptiveConfig.learning_rate << [0.01, 0.1]
    ])
    sweeps.append(cnn_sweep)
    
    # Transformer models with lower learning rates
    transformer_sweep = Sweep(AdaptiveConfig).product([
        AdaptiveConfig.model_type << ["transformer"],
        AdaptiveConfig.learning_rate << [0.0001, 0.001]
    ])
    sweeps.append(transformer_sweep)
    
    # Chain all sweeps
    return sweeps[0].chain(*sweeps[1:])

conditional_sweep = create_conditional_sweep()
for config in conditional_sweep:
    print(f"Model: {AdaptiveConfig.model_type}, LR: {AdaptiveConfig.learning_rate}")
```

## Experiment Management

### Saving and Loading Sweeps

```python
import json
from pathlib import Path

class ExperimentConfig(ParamsProto):
    learning_rate = Proto(default=0.001)
    batch_size = Proto(default=32)
    model = Proto(default="resnet")

# Create sweep
sweep = Sweep(ExperimentConfig).product([
    ExperimentConfig.learning_rate << [0.001, 0.01, 0.1],
    ExperimentConfig.batch_size << [16, 32, 64]
])

# Save sweep configuration to JSONL file
sweep.save("experiments.jsonl")

# The file contains one JSON object per line:
# {"ExperimentConfig.learning_rate": 0.001, "ExperimentConfig.batch_size": 16}
# {"ExperimentConfig.learning_rate": 0.001, "ExperimentConfig.batch_size": 32}
# ...

# Load and replay experiments
def load_experiment_configs(filename):
    """Load experiment configurations from JSONL file"""
    configs = []
    with open(filename, 'r') as f:
        for line in f:
            config = json.loads(line.strip())
            configs.append(config)
    return configs

# Replay experiments
experiment_configs = load_experiment_configs("experiments.jsonl")
for i, config in enumerate(experiment_configs):
    print(f"Running experiment {i+1}")
    ExperimentConfig._update(config)
    
    # Run your experiment
    # result = train_model()
    print(f"Config: {config}")
```

### Experiment Tracking

```python
import time
import json
from datetime import datetime

class TrackingConfig(ParamsProto):
    learning_rate = Proto(default=0.001)
    batch_size = Proto(default=32)
    epochs = Proto(default=10)

def run_experiment_with_tracking():
    """Run experiments with comprehensive tracking"""
    
    # Create sweep
    sweep = Sweep(TrackingConfig).product([
        TrackingConfig.learning_rate << [0.001, 0.01],
        TrackingConfig.batch_size << [16, 32],
        TrackingConfig.epochs << [5, 10]
    ])
    
    results = []
    
    for i, config in enumerate(sweep):
        print(f"\n=== Experiment {i+1}/{len(sweep)} ===")
        
        # Record experiment metadata
        experiment = {
            'experiment_id': i + 1,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'learning_rate': TrackingConfig.learning_rate,
                'batch_size': TrackingConfig.batch_size,
                'epochs': TrackingConfig.epochs
            }
        }
        
        # Simulate training
        start_time = time.time()
        
        # Your actual training code would go here
        # For demo, we'll simulate some results
        import random
        accuracy = random.uniform(0.7, 0.95)
        loss = random.uniform(0.1, 0.5)
        
        end_time = time.time()
        
        # Record results
        experiment.update({
            'results': {
                'accuracy': accuracy,
                'loss': loss,
                'training_time_seconds': end_time - start_time
            }
        })
        
        results.append(experiment)
        
        print(f"Config: LR={TrackingConfig.learning_rate}, "
              f"BS={TrackingConfig.batch_size}, Epochs={TrackingConfig.epochs}")
        print(f"Results: Accuracy={accuracy:.4f}, Loss={loss:.4f}")
    
    # Save results
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find best experiment
    best = max(results, key=lambda x: x['results']['accuracy'])
    print(f"\n=== Best Experiment ===")
    print(f"ID: {best['experiment_id']}")
    print(f"Config: {best['config']}")
    print(f"Accuracy: {best['results']['accuracy']:.4f}")
    
    return results

# Run tracking experiment
results = run_experiment_with_tracking()
```

## Advanced Hyperparameter Search

### Random Search

```python
import random

class RandomSearchConfig(ParamsProto):
    learning_rate = Proto(default=0.001)
    weight_decay = Proto(default=1e-4)
    dropout = Proto(default=0.1)

def random_search(n_trials=20):
    """Perform random search over hyperparameters"""
    
    # Define search spaces
    lr_choices = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    weight_decay_choices = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    dropout_choices = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    results = []
    
    for trial in range(n_trials):
        # Sample random configuration
        config = {
            'learning_rate': random.choice(lr_choices),
            'weight_decay': random.choice(weight_decay_choices),
            'dropout': random.choice(dropout_choices)
        }
        
        # Apply configuration
        RandomSearchConfig._update(config)
        
        print(f"Trial {trial+1}: {config}")
        
        # Run experiment (simulated)
        accuracy = random.uniform(0.7, 0.95)
        results.append({
            'trial': trial + 1,
            'config': config.copy(),
            'accuracy': accuracy
        })
    
    # Find best configuration
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest configuration: {best['config']}")
    print(f"Best accuracy: {best['accuracy']:.4f}")
    
    return results

# Run random search
random_results = random_search(15)
```

### Bayesian Optimization

```python
# This example shows how to integrate with optimization libraries
try:
    from skopt import gp_minimize
    from skopt.space import Real, Categorical
    from skopt.utils import use_named_args
    
    class OptimizationConfig(ParamsProto):
        learning_rate = Proto(default=0.001)
        batch_size = Proto(default=32)
        optimizer = Proto(default="adam")
    
    # Define search space
    search_space = [
        Real(1e-5, 1e-1, name='learning_rate', prior='log-uniform'),
        Categorical([16, 32, 64, 128], name='batch_size'),
        Categorical(['adam', 'sgd', 'adamw'], name='optimizer')
    ]
    
    @use_named_args(search_space)
    def objective(**params):
        """Objective function to minimize (negative accuracy)"""
        
        # Update configuration
        OptimizationConfig._update(**params)
        
        print(f"Trying: LR={OptimizationConfig.learning_rate:.6f}, "
              f"BS={OptimizationConfig.batch_size}, "
              f"Opt={OptimizationConfig.optimizer}")
        
        # Simulate training (replace with your actual training code)
        import random
        accuracy = random.uniform(0.7, 0.95)
        
        # Return negative accuracy (since we're minimizing)
        return -accuracy
    
    def bayesian_optimization(n_calls=20):
        """Run Bayesian optimization"""
        
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=n_calls,
            random_state=42
        )
        
        # Best parameters
        best_params = dict(zip([dim.name for dim in search_space], result.x))
        best_score = -result.fun  # Convert back to positive accuracy
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best accuracy: {best_score:.4f}")
        
        return result
    
    # Run Bayesian optimization (requires scikit-optimize)
    # bayesian_result = bayesian_optimization(15)
    
except ImportError:
    print("Bayesian optimization requires scikit-optimize: pip install scikit-optimize")
```

### Multi-Objective Optimization

```python
class MultiObjectiveConfig(ParamsProto):
    learning_rate = Proto(default=0.001)
    model_complexity = Proto(default=1.0)

def multi_objective_sweep():
    """Sweep optimizing for both accuracy and efficiency"""
    
    sweep = Sweep(MultiObjectiveConfig).product([
        MultiObjectiveConfig.learning_rate << [0.001, 0.01, 0.1],
        MultiObjectiveConfig.model_complexity << [0.5, 1.0, 2.0]
    ])
    
    results = []
    
    for config in sweep:
        print(f"Config: LR={MultiObjectiveConfig.learning_rate}, "
              f"Complexity={MultiObjectiveConfig.model_complexity}")
        
        # Simulate training with multiple objectives
        import random
        accuracy = random.uniform(0.7, 0.95)
        
        # Higher complexity -> better accuracy but slower training
        complexity_bonus = MultiObjectiveConfig.model_complexity * 0.05
        accuracy += complexity_bonus
        
        training_time = 100 * MultiObjectiveConfig.model_complexity  # seconds
        
        results.append({
            'config': {
                'learning_rate': MultiObjectiveConfig.learning_rate,
                'model_complexity': MultiObjectiveConfig.model_complexity
            },
            'accuracy': min(accuracy, 1.0),
            'training_time': training_time,
            'efficiency': accuracy / training_time  # Accuracy per second
        })
    
    # Find Pareto optimal solutions
    pareto_optimal = []
    for i, result1 in enumerate(results):
        is_dominated = False
        for j, result2 in enumerate(results):
            if i != j:
                # result2 dominates result1 if it's better in both objectives
                if (result2['accuracy'] >= result1['accuracy'] and 
                    result2['efficiency'] >= result1['efficiency'] and
                    (result2['accuracy'] > result1['accuracy'] or 
                     result2['efficiency'] > result1['efficiency'])):
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_optimal.append(result1)
    
    print(f"\nFound {len(pareto_optimal)} Pareto optimal solutions:")
    for result in pareto_optimal:
        print(f"Config: {result['config']}")
        print(f"Accuracy: {result['accuracy']:.4f}, "
              f"Efficiency: {result['efficiency']:.6f}")
        print("---")
    
    return pareto_optimal

# Run multi-objective optimization
pareto_solutions = multi_objective_sweep()
```

## Parallel Experiment Execution

### Concurrent Sweeps

```python
import concurrent.futures
import time
from params_proto.v2.proto import ParamsProto, Proto
from params_proto.v2.hyper import Sweep

class ParallelConfig(ParamsProto):
    learning_rate = Proto(default=0.001)
    batch_size = Proto(default=32)

def run_single_experiment(config_dict):
    """Run a single experiment with given configuration"""
    
    # Create a local copy of configuration
    experiment_id = config_dict.get('_experiment_id', 0)
    
    print(f"Starting experiment {experiment_id}")
    
    # Simulate training time
    time.sleep(2)  # Replace with actual training
    
    # Simulate results
    import random
    accuracy = random.uniform(0.7, 0.95)
    
    result = {
        'experiment_id': experiment_id,
        'config': config_dict,
        'accuracy': accuracy
    }
    
    print(f"Completed experiment {experiment_id}: accuracy={accuracy:.4f}")
    return result

def parallel_sweep(max_workers=4):
    """Run hyperparameter sweep in parallel"""
    
    # Create sweep
    sweep = Sweep(ParallelConfig).product([
        ParallelConfig.learning_rate << [0.001, 0.01, 0.1],
        ParallelConfig.batch_size << [16, 32, 64]
    ])
    
    # Prepare experiment configurations
    experiment_configs = []
    for i, config in enumerate(sweep):
        config_dict = {
            '_experiment_id': i + 1,
            'learning_rate': ParallelConfig.learning_rate,
            'batch_size': ParallelConfig.batch_size
        }
        experiment_configs.append(config_dict)
    
    # Run experiments in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_config = {
            executor.submit(run_single_experiment, config): config 
            for config in experiment_configs
        }
        
        for future in concurrent.futures.as_completed(future_to_config):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                config = future_to_config[future]
                print(f"Experiment {config['_experiment_id']} failed: {exc}")
    
    # Sort results by experiment ID
    results.sort(key=lambda x: x['experiment_id'])
    
    # Print summary
    print(f"\n=== Parallel Sweep Results ===")
    for result in results:
        print(f"Exp {result['experiment_id']}: "
              f"LR={result['config']['learning_rate']}, "
              f"BS={result['config']['batch_size']}, "
              f"Acc={result['accuracy']:.4f}")
    
    return results

# Run parallel sweep
# parallel_results = parallel_sweep(max_workers=3)
```

## Integration with ML Frameworks

### PyTorch Integration

```python
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

class PyTorchConfig(ParamsProto):
    # Model parameters
    model_type = Proto("Model type", default="resnet")
    num_layers = Proto("Number of layers", default=18)
    
    # Training parameters
    learning_rate = Proto("Learning rate", default=0.001)
    batch_size = Proto("Batch size", default=32)
    epochs = Proto("Number of epochs", default=10)
    optimizer_type = Proto("Optimizer", default="adam")
    
    # Regularization
    weight_decay = Proto("Weight decay", default=1e-4)
    dropout = Proto("Dropout rate", default=0.1)

def pytorch_hyperparameter_sweep():
    """Example of hyperparameter sweep with PyTorch"""
    
    sweep = Sweep(PyTorchConfig).product([
        PyTorchConfig.learning_rate << [0.001, 0.01],
        PyTorchConfig.batch_size << [32, 64],
        PyTorchConfig.optimizer_type << ["adam", "sgd"]
    ])
    
    best_accuracy = 0
    best_config = None
    
    for i, config in enumerate(sweep):
        print(f"\n=== Experiment {i+1}/{len(sweep)} ===")
        print(f"LR: {PyTorchConfig.learning_rate}")
        print(f"Batch size: {PyTorchConfig.batch_size}")
        print(f"Optimizer: {PyTorchConfig.optimizer_type}")
        
        try:
            # Create model (replace with your model)
            # model = create_model(PyTorchConfig)
            
            # Create optimizer
            # if PyTorchConfig.optimizer_type == "adam":
            #     optimizer = optim.Adam(model.parameters(), 
            #                           lr=PyTorchConfig.learning_rate,
            #                           weight_decay=PyTorchConfig.weight_decay)
            # elif PyTorchConfig.optimizer_type == "sgd":
            #     optimizer = optim.SGD(model.parameters(),
            #                          lr=PyTorchConfig.learning_rate,
            #                          weight_decay=PyTorchConfig.weight_decay,
            #                          momentum=0.9)
            
            # Train model
            # accuracy = train_model(model, optimizer, PyTorchConfig)
            
            # Simulate training for demo
            import random
            accuracy = random.uniform(0.7, 0.95)
            
            print(f"Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = {
                    'learning_rate': PyTorchConfig.learning_rate,
                    'batch_size': PyTorchConfig.batch_size,
                    'optimizer_type': PyTorchConfig.optimizer_type
                }
                print("*** New best configuration! ***")
                
        except Exception as e:
            print(f"Experiment failed: {e}")
            continue
    
    print(f"\n=== Final Results ===")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best configuration: {best_config}")

# Run PyTorch sweep
pytorch_hyperparameter_sweep()
```

This comprehensive guide covers all aspects of hyperparameter sweeps with params-proto, from basic grid searches to advanced optimization techniques and parallel execution.