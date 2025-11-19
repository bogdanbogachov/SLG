# Llama Capacity Experiment Instructions

## Overview

**Goal**: Understand how much data Llama 3.2 1B Instruct can be trained on without losing its efficiency.

**Approach**: Create cumulative datasets of increasing sizes, split each into train/test sets, and run fine-tuning experiments 3 times per dataset.

---

## Data Preparation

### Step 1: Create Cumulative QA Files

Split `question_answer/qa.json` into cumulative files where each file includes all previous data:
- File 1: x elements
- File 2: x+y elements (includes File 1)
- File 3: x+y+z elements (includes Files 1 and 2)
- Continue as needed

Save files in `question_answer/cumulative_datasets/` with names like `qa_cumulative_1.json`, `qa_cumulative_2.json`, etc.

**Important**: Shuffle data with `random_state=42` for reproducibility before creating cumulative files.

### Step 2: Split Each Cumulative File into Train/Test

For each cumulative file, create train/test splits:
- 80% train, 20% test
- Use `random_state=42` for reproducibility
- Remove instances where `title == answer`

Save as `qa_cumulative_{n}_train.json` and `qa_cumulative_{n}_test.json` for each file.

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

---

## Running Experiments

### Experiment Naming Convention

Format: `qa_cumulative_{file_number}_run{1,2,3}`

Examples:
- `qa_cumulative_1_run1`
- `qa_cumulative_1_run2`
- `qa_cumulative_1_run3`

Each cumulative dataset must be run 3 times for statistical reliability.

### Cluster Submission Process

For each experiment:

1. Copy the appropriate train/test files:
   ```bash
   cp question_answer/cumulative_datasets/qa_cumulative_{n}_train.json question_answer/qa_train.json
   cp question_answer/cumulative_datasets/qa_cumulative_{n}_test.json question_answer/qa_test.json
   ```

2. Submit job:
   ```bash
   export EXP=qa_cumulative_{n}_run{r}
   sbatch --job-name="EXP" --output="${EXP}.out" --error="${EXP}.err" job.sh
   ```

**Important**: The `EXP` environment variable overrides the experiment name in `config.yaml`.

### Complete Workflow Example

For cumulative file 1:
```bash
# Copy data files
cp question_answer/cumulative_datasets/qa_cumulative_1_train.json question_answer/qa_train.json
cp question_answer/cumulative_datasets/qa_cumulative_1_test.json question_answer/qa_test.json

# Run 1
export EXP=qa_cumulative_1_run1
sbatch --job-name="EXP" --output="${EXP}.out" --error="${EXP}.err" job.sh

# Run 2
export EXP=qa_cumulative_1_run2
sbatch --job-name="EXP" --output="${EXP}.out" --error="${EXP}.err" job.sh

# Run 3
export EXP=qa_cumulative_1_run3
sbatch --job-name="EXP" --output="${EXP}.out" --error="${EXP}.err" job.sh
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

---

## Checklist

Before starting:
- [ ] Cumulative QA files created and validated
- [ ] Train/test splits created for each file
- [ ] Compute Canada environment set up
- [ ] Models downloaded
- [ ] `config.yaml` configured correctly

For each experiment:
- [ ] Correct train/test files copied
- [ ] `EXPERIMENT` variable set correctly
- [ ] Job submitted
- [ ] Results verified after completion

