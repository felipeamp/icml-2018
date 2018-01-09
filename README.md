# ICML 2018

## Pre-requisites:
- Python 3
- Numpy

## How to run:
First edit the init_split_monte_carlo_experiment.py first lines to set the number of experiments, pair of (values, classes) to run and comment/uncomment the criteria you want/don't want ro tun. Then run the following command:

```python3 init_split_monte_carlo_experiment.py --csv_output_dir /path/to/output/folder --csv_experiments_filename name_of_experiments.csv --csv_table_filename name_of_table.csv```

## How to print the contingency table
When printing the contingency table, no experiments will be executed. You must pass the flag `print_contingency_table` and also set positive values for the flags `num_values`, `num_classes` and `experiment_num`. For instance:

```python3 init_split_monte_carlo_experiment.py --print_contingency_table --num_values=6 --num_classes=3 --experiment_num=1```
