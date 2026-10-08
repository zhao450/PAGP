# PAGP

## Environment and Dependencies

Recommended environment:

- Python 3.10 or 3.11
- CPU environment with Python multiprocessing support

Required Python packages:

- `numpy`
- `scipy`
- `simpy`
- `deap`

### Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/zhao450/PAGP.git
cd PAGP
```

Create and activate a virtual environment:

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy scipy simpy deap
```

## How to Run

Run the project from the repository root directory.

Make sure the training output directory exists:

```bash
python -c "from pathlib import Path; Path('MO_GP_statistics_pc/train').mkdir(parents=True, exist_ok=True)"
```

Start the experiment:

```bash
python compare_me.py
```

The default configuration in `compare_me.py` runs the experiment over multiple random seeds using multiprocessing. To change the experiment size or runtime settings, edit the configuration constants at the top of `compare_me.py` before running.
