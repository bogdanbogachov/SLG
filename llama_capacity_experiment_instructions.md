# Llama Capacity Experiment Instructions

## Overview

**Goal**: Understand how much data Llama 3.2 1B Instruct can be trained on without losing its efficiency.

**Approach**: Create cumulative datasets of increasing sizes, split each into train/test sets, and run fine-tuning experiments 3 times per dataset.

---

## Experiment Setup on Compute Canada

### Step 1: Set Up Environment

1. Load required modules:
   ```bash
   module load python/3.11.5
   module load rust
   module load gcc cuda/12.2
   module load scipy-stack
   module load gcc arrow
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   export OPENAI_API_KEY="dummy"
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r cc_requirements.txt
   ```

### Step 2: Configure Training

In `config.yaml`, set:
```yaml
training_components:
  train_slg_system: false
  train_3_2_1b: true
  train_3_1_8b: false
```

### Step 3: Download Models (only llama 3.2 1B; it might not work though, if that is the case, another model will be shared)

```bash
python main.py --download_models=True
```

### Step 4: Create QA pairs

Extract from `question_answer/qa.json` a portion of data (5%). The idea is to extract 5% more data each 3 consecutive runs so that the qa.json file would include the new 5% + all previous data:
- File 1: x elements
- File 2: x+y elements (includes File 1)
- File 3: x+y+z elements (includes Files 1 and 2)
- Continue as needed

It can be done manually or automatically (a new function would be needed for the automatic approach).

### Step 5: Split QA pairs

```bash
python main.py --split_qa=True
```

---

## Running Experiments

### Cluster Submission Process

For each experiment:

job.sh setup:
- Enter your email to receive notifications about your jobs
- Estimate time required for a specific job
- Choose appropriate resources (for llama 2 fine-tuning GPU partitioning is desired)

   ```bash
   export EXP=experiment_name
   sbatch --job-name="$EXP" --output="${EXP}.out" --error="${EXP}.err" job.sh
   ```

**Important**: The `EXP` environment variable overrides the experiment name in `config.yaml`.

### Complete Workflow Example

For cumulative file 1:
```bash
# Run 1
export EXP=experiment_name
sbatch --job-name="$EXP" --output="${EXP}.out" --error="${EXP}.err" job.sh

# Run 2
export EXP=experiment_name
sbatch --job-name="$EXP" --output="${EXP}.out" --error="${EXP}.err" job.sh

# Run 3
export EXP=experiment_name
sbatch --job-name="$EXP" --output="${EXP}.out" --error="${EXP}.err" job.sh
```

Repeat for each cumulative file.

---

## Verification

### Check Job Status
```bash
squeue -u $USER
```

### Monitor Output
```bash
tail -f qa_cumulative_1_run1.out
tail -f qa_cumulative_1_run1.err
```
