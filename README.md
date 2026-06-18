# CBL Multi-disciplinary Project: UniCharger 

## Overview

This repository holds all data and scripts used in UniCharger modelling and optimization for the CBL group project.

The repository is split into the following parts:

1. `Data_Set` - the course-provided datasets;
2. `processed data` - data obtained from other sources and computed using our scripts ;
3. `output` - output of optimization and simulation scenarios;
4. `scripts` - all the scripts used for the project. Further split into:
    1. `analysis` - miscellaneous analysis tasks;
    2. `shared` - used as a common core for network distance walking;
    3. `Task 1`, `Task 2`, `Task 3` - scripts relevant to each phase of development

# Setup

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

All scripts should be executed via `uv run` to ensure the correct environment is used:

```bash
uv run './scripts/Task 1/remaining_grid_capacity.py'
```

