
localrules: data_partition


rule data_partition:
    input:
        features=DATA_DIR / "01_raw/features.zarr",
    output:
        expand("results/data_partition/{split}.json", split=range(N_SPLITS)),
        report(
            "results/data_partition/time_split.png",
            category="Cross-validation"
        )
    conda:
        "../envs/eda.yaml"
    script:
        "../scripts/data_partitioning.py"


rule preprocess:
    input:
        features=DATA_DIR / "01_raw/features.zarr",
        targets=DATA_DIR / "01_raw/targets.zarr",
    params:
    output:
        x=DATA_DIR / "02_preprocessed/x.nc",
        y=DATA_DIR / "02_preprocessed/y.nc",
        stations_list="results/preprocess/stations_list.json",
    resources:
        mem_mb=20000
    threads: 4
    conda:"../envs/model.yaml"
    log: "logs/preprocess/preprocess.log"
    script: "../scripts/preprocessing.py"

        
rule train:
    input:
        x=DATA_DIR / "02_preprocessed/x.nc",
        y=DATA_DIR / "02_preprocessed/y.nc",
        split="results/data_partition/{split}.json",
        stations_list="results/preprocess/stations_list.json",
    params:
        params_from_wildcards
    output:
        state="results/train/split~{split}/seed~{seed}/{approach}/{experiment}/{params}/state.pth",
        config="results/train/split~{split}/seed~{seed}/{approach}/{experiment}/{params}/config.json",
        scaling_values="results/train/split~{split}/seed~{seed}/{approach}/{experiment}/{params}/scaling_values.json",
    resources:
        mem_mb=20000
    threads: 2
    conda: "../envs/model.yaml"
    log: "logs/train/split~{split}/seed~{seed}/{approach}/{experiment}/{params}.log"
    script: "../scripts/training.py"


rule evaluate:
    input:
        x=DATA_DIR / "02_preprocessed/x.nc",
        y=DATA_DIR / "02_preprocessed/y.nc",
        stations_list="results/preprocess/stations_list.json",
        state="results/train/split~{split}/seed~{seed}/{approach}/{experiment}/{params}/state.pth",
        split="results/data_partition/{split}.json",
        config="results/train/split~{split}/seed~{seed}/{approach}/{experiment}/{params}/config.json",
        scaling_values="results/train/split~{split}/seed~{seed}/{approach}/{experiment}/{params}/scaling_values.json",
    params:
        params_from_wildcards
    output:
        predictions=DATA_DIR / "03_model_output/split~{split}/seed~{seed}/{approach}/{experiment}/{params}/{partition}_predictions.nc",
    resources:
        mem_mb=40000
    threads: 2
    conda: "../envs/model.yaml"
    log: "logs/test/split~{split}/seed~{seed}/{approach}/{experiment}/{params}/{partition}.log"
    script: "../scripts/prediction.py"


localrules: analysis
rule analysis:
    input:
        predictions=experiment_inputs,
        y=DATA_DIR / "02_preprocessed/y.nc"
    threads: 5
    output:
        directory("results/experiments/{experiment}/{partition}/analysis/")
    conda:
        "../envs/eda.yaml"
    notebook:
        "../notebooks/{wildcards.experiment}.py.ipynb"


localrules: performance
rule performance:
    input:
        predictions=experiment_inputs,
        y=DATA_DIR / "02_preprocessed/y.nc"
    threads: 5
    output:
        directory("results/experiments/{experiment}/{partition}/performance/")
    conda:
        "../envs/eda.yaml"
    notebook:
        "../notebooks/performance.py.ipynb"


localrules: physical_consistency
rule physical_consistency:
    input:
        predictions=experiment_inputs,
        y=DATA_DIR / "02_preprocessed/y.nc"
    threads: 5
    output:
        directory("results/experiments/{experiment}/{partition}/physical_consistency/")
    conda:
        "../envs/eda.yaml"
    notebook:
        "../notebooks/physical_consistency.py.ipynb"



rule tune:
    input: 
        x=DATA_DIR / "02_preprocessed/x.nc",
        y=DATA_DIR / "02_preprocessed/y.nc",
        splits=expand("results/data_partition/{split}.json", split=[0,1,2,3]),
        stations_list="results/preprocess/stations_list.json"
    params:
        lambda wildcards: config["tuning"][wildcards.experiment]
    output:
        results_dir=directory("results/tune/{experiment}/{approach}/")
    resources:
        mem_mb=200000, 
        time="12:00:00"
    threads: config["tuning"]["threads"]
    log: "logs/tune/{experiment}/{approach}/logging.log"
    conda: "../envs/tune.yaml"
    script: "../scripts/tuning.py"
