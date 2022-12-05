
rule explorative_data_analysis:
    input:
        features=DATA_DIR / "features.zarr",
        targets=DATA_DIR / "targets.zarr"
    output:
        station_list="results/eda/station_list.json"
    conda:
        "../envs/eda.yaml"
    script:
        "../scripts/eda.py"


        