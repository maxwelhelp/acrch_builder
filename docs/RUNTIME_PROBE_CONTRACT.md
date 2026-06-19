# Runtime probe contract v1

The universal inspector does not know the project model.

A project-side command must write:

```text
reports/agent_inspector/runtime_probe.json
```

Required shape:

```json
{
  "schema": "runtime_probe_contract.v1",
  "project": "your_project",
  "metrics": {
    "loss": 1.0
  },
  "grad_norms": {
    "controller": 0.1,
    "scanner": 0.02,
    "simulator": 0.03,
    "executor": null,
    "memory": null,
    "classifier": 0.5,
    "model_core": 1.0
  },
  "credit_path": {
    "expected_edge_present": true,
    "candidate_present": true,
    "choice_mass": 0.9,
    "edge_active": 0.2,
    "recovery": 1.0,
    "loss_connected": true
  },
  "health": {
    "gradient_closed": true,
    "credit_closed": true
  }
}
```

For another project only replace `commands/probe_learning_loop.sh` and the project-side producer.
