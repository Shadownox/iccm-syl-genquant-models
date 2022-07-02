iccm-syl-genquant-models
========================

Companion repository for the 2022 article "Do Models of Syllogistic Reasoning extend to Generalized Quantifiers?" published for the 20th Annual Meeting of the International Conference on Cognitive Modelling.

## Overview

- `analysis`: Contains the analysis scripts generating the results and figures from the paper.
- `analysis/error_analysis.py`: Script for generating the error-analysis heatmaps (*Figure 3* in the paper).
- `analysis/evaluate_performance.py`: Script for generating the general performance plot (*Figure 1* in the paper). Also calculates the statistical significance.
- `analysis/param_distribution.py`: Script for generating the parameter distribution plots (*Figure 4* in the paper).
- `analysis/performance_by_tasks.py`: Script for generating the performance comparison plot between generalized and classic syllogisms (*Figure 2* in the paper).
- `benchmark`: Contains the CCOBRA benchmark specifications for the analysis.
- `benchmark/benchmark_21_22_coverage_classic.json`: Benchmark specification that runs on the classic syllogisms only (for parameter distribution analysis).
- `benchmark/benchmark_21_22_coverage_full.json`: Benchmark specification that runs on all tasks (for the performance analysis).
- `benchmark/benchmark_21_22_coverage_genquant.json`: Benchmark specification that runs on the generalized syllogisms only (for parameter distribution analysis).
- `data`: Contains the datasets and results from the benchmarking run of the models.
- `data/classicSyllog_Params.json`: Parameters for the models when fitted on the classic syllogisms.
- `data/genQuant_Params.json`: Parameters for the models when fitted on the generalized syllogisms.
- `data/Ragni21_22.csv`: Responses to all 144 syllogisms with the quantifiers *All*, *Some*, *Some not*, *No*, *Most* and *Most not*.
- `data/Ragni21_22_classic.csv`: Filtered version of the Ragni21_22 dataset that only contains the classic syllogisms.
- `data/Ragni21_22_gen.csv`: Filtered version of the Ragni21_22 dataset that only contains the generalized syllogisms.
- `data/results_full.csv`: Model benchmarking results (CCOBRA results) based on the Ragni21_22 dataset.
- `models`: Contains the CCOBRA-models.
- `models/mfa`: Contains the implementation of the MFA (most-frequent-answer model).
- `models/phm`: Contains the implementation of PHM.
- `models/pymreasoner`: Contains the implementation of mReasoner.
- `models/random`: Contains the implementation of the random baseline model.
- `models/ubcf`: Contains the implementation of the user-based collaborative filtering model.

## Analysis Scripts

### Dependencies

- Python 3
    - CCOBRA
    - pandas
    - numpy
	- matplotlib
    - seaborn
    - scipy

### Usage

After downloading the repository, navigate to the analysis subfolder:

```
cd /path/to/repository/analysis
```

All scripts can be executed without entering additional parameters. The scripts will create the plots in the same folder. To execute the scripts, enter: 

```
$> python [script].py
```

## Benchmarks

### Dependencies

- Python 3
    - CCOBRA
    - pandas
    - numpy

### Usage

Navigate to the benchmark-folder:

```
cd /path/to/repository/benchmark
```

Use CCOBRA to run a benchmark, e.g.:

```
ccobra benchmark_21_22_coverage_full.json
```

When CCOBRA is finished, a HTML-file will be created in the benchmark-folder. The HTML-file, when opened with a browser, allows to extract the parameters (as a JSON-file) and benchmark results (as a CSV). These files are (besides differences due to sampling) equivalent to the files already stored in the data-folder.

## References

Mittenbühler, M., Brand, D., & Ragni, M. (2022). Do Models of Syllogistic Reasoning extend to Generalized Quantifiers? In proceedings of the 20th Annual Meeting of the International Conference on Cognitive Modelling.
