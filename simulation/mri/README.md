# MRI Department Discrete Event Simulation

## Prerequisites

The simulation requires a folder containing sql files:
- MRI_cancellation_rate.sql
- MRI_current_waiting_list.sql
- MRI_dna_rate.sql
- MRI_historic_waiting_list.sql
- engine.txt
- transformed_mri_scanners.json

MRI_cancellation_rate and MRI_dna_rate should return a table containing the % of cancellations and dnas respectively. 
MRI_current_waiting_list should return the waiting list as of the current date.
MRI_historic_waiting_list must return a historic waiting list and will be used to randomly select patients from.
engine.txt contains the string used to create a database engine and transformed_mri_scanners contains information on how
scanners are utilised in the form:

```json
{
  "name of scanner": {
    "site": "location",
    "room": "room",
    "scanner coordinates": "(X, Y)",
    "inpatient_scans": bool,
    "peadiatric_scans": bool,
    "day": {
      "Mon": [
        {
          "open": "08:00",
          "close": "17:00",
          "label": "tst_capacity"
        }
      ]
    }
  },
  "name of scanner 2": {...}
}
```

## Running the simulation

The simulation can be run with simple python code:

```python
from simulation.mri import MriSimulation

seed, sim, mridept = MriSimulation(
    path_to_sql_queries="mri_sql", 
    fu_rate=0,
)

sim.run_simulation()

sim.graph.nodes["Simulate"]["capacity"].metrics # Stores the metrics linked to the wait list
mridept.metrics # Access metrics such as underutilised time
```

