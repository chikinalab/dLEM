# `dLEM`

dLEM: differentiable Loop Extrusion Model

**Usage**:

```console
$ dLEM [OPTIONS] DNA_INTS OUTPUT_LOCATION
```

**Arguments**:

* `DNA_INTS`: Chromatin input dataset in the form of a .cool file  [required]
* `OUTPUT_LOCATION`: Location of output directory, will be created if not existing  [required]

**Options**:

* `--device [gpu|cpu|tpu]`: Select computation device.  [default: cpu]
* `-r, --region TEXT`: Print details about a specific target chromosomal region 
(ex: chr16:51906425-55560999), this includes plotting and 
track output. By default dLEM learns the whole chromosome 
(ex: chr16) the target region lies in.
* `-b, --bed TEXT`: Define target regions in BED file to plot and output tracks for,
Plot and output tsv for all regions in BED file,
3 columns , ex: .
* `--chromosomes COMMA_LIST_PARSER`: Comma-separated list of chromosomes to train dLEM on (e.g., &#x27;chr1,chr2,chr3&#x27;).
* `--all`: Run dLEM across all chromosomes in input chromatin loop file.
* `-l, --resolution INTEGER`: Resolution to pull .cool file from .mcool file, must exist.
* `--start-diag INTEGER`: Diagonal offset where updates start during training.  [default: 5]
* `--train-rows INTEGER`: Number of diagonals to use during training dLEM.  [default: 170]
* `--steps INTEGER`: Number of steps for training.  [default: 10]
* `-w, --window-size INTEGER`: Column window size for slowdown fitting  [default: 3000]
* `-s, --window-step INTEGER`: Step between slowdown-fitting windows.  [default: 200]
* `-d, --decay-extent INTEGER`: Number of band rows to use when fitting slowdown.  [default: 500]
* `-i, --iterations INTEGER`: Number of iterations to fit the model over.  [default: 300]
* `-lr, --lr FLOAT`: Change speed of parameter tuning, useful if loss is too fast/slow or model not converging.  [default: 0.01]
* `-pm, --prediction-mode [mse|corr|final]`: Which parameter set to use for predictions.  [default: mse]
* `-pr, --prediction-rows INTEGER`: Number of diagonals to predict when outputting dLEM predictions  [default: 700]
* `--full-output`: Predict all diagonals in output instead of a set number
* `-esm, --early-stop-metric [mse|corr|none]`: Metric for early stopping  [default: mse]
* `--debug`: Verbose output during training
* `-c, --output-cool`: Output the prediction as a .cool file
* `-t, --output-tracks`: Output the parameters as genomic tracks for target regions in BED and whole genome in BigWig format
* `--norm`: Normalize the predicted band using geometric mean
* `-p, --plot`: Plot fits to compare to input
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.
