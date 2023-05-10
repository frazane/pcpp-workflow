### optional rules

localrules: all_combine_figures
rule all_combine_figures:
    input:
        expand(
            "results/figures/{experiment}_{partition}_combined_{how}.png",
            experiment=["default"],
            partition=["test"],
            how=["mixed", "performance", "physical_consistency"]
        ),
        expand(
            "results/figures/{experiment}_{partition}_combined_mixed.png",
            experiment=["data_efficiency"],
            partition=["test"],
            how=["mixed"]
        )

localrules: combine_figures
rule combine_figures:
    shell:
        """
        convert -font DejaVu-Sans -pointsize 53 -gravity northwest -draw "text 30, 30 'a'" \
            {input.fig1} /tmp/fig1_{wildcards.experiment}_{wildcards.partition}_{params.how}.png &&
        
        convert -font DejaVu-Sans -pointsize 53 -gravity northwest -draw "text 30, 30 'b'" \
            {input.fig2} /tmp/fig2_{wildcards.experiment}_{wildcards.partition}_{params.how}.png &&
        
        convert /tmp/fig1_{wildcards.experiment}_{wildcards.partition}_{params.how}.png \
             /tmp/fig2_{wildcards.experiment}_{wildcards.partition}_{params.how}.png +append {output}
        """


use rule combine_figures as combine_figures_mixed with:
    input:
        fig1="results/experiments/{experiment}/{partition}/performance/mae_boxplots.png",
        fig2="results/experiments/{experiment}/{partition}/physical_consistency/relative_humidity_deviations.png"
    params:
        how="mixed"
    output:
        "results/figures/{experiment}_{partition}_combined_mixed.png"

use rule combine_figures as combine_figures_performance with:
    input:
        fig1="results/experiments/{experiment}/{partition}/performance/mae_boxplots.png",
        fig2="results/experiments/{experiment}/{partition}/performance/msss_boxplots.png"
    params:
        how="performance"
    output:
        "results/figures/{experiment}_{partition}_combined_performance.png"

use rule combine_figures as combine_figures_physical_consistency with:
    input:
        fig1="results/experiments/{experiment}/{partition}/physical_consistency/relative_humidity_deviations.png",
        fig2="results/experiments/{experiment}/{partition}/physical_consistency/mixing_ratio_deviations.png"
    params:
        how="physical_consistency"
    output:
        "results/figures/{experiment}_{partition}_combined_physical_consistency.png"

use rule combine_figures as combine_figures_data_efficiency with:
    input:
        dir1="results/experiments/{experiment}/{partition}/analysis/",
        fig1="results/experiments/{experiment}/{partition}/analysis/MAE_vs_reduction.png",
        fig2="results/experiments/{experiment}/{partition}/analysis/P_vs_reduction.png",
    params:
        how="mixed"
    output:
        "results/figures/{experiment}_{partition}_combined_mixed.png"