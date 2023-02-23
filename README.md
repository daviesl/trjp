# RJLab: Reversible Jump Laboratory for Python 3.x

Examples for AISTAT 2023 conference paper "Transport Reversible Jump Proposals"

To run the examples, first install Poetry, then install the project.

    python3 -m pip install poetry
    git clone git@github.com:daviesl/trjp.git
    cd trjp
    poetry install

Then run the below shell script to produce results and plots for all examples.

    #!/usr/bin/env bash
    
    if [ $# -eq 0 ]; then
    	export RUNSAS=1
    	export RUNFA=1
    	export RUNVS=1
    elif [ $1 == s ]; then
    	export RUNSAS=1
    	export RUNFA=0
    	export RUNVS=0
    elif [ $1 == f ]; then
    	export RUNSAS=0
    	export RUNFA=1
    	export RUNVS=0
    elif [ $1 == v ]; then
    	export RUNSAS=0
    	export RUNFA=0
    	export RUNVS=1
    fi
    
    cd examples
    
    if [ $RUNSAS -eq 1 ]; then
    # run SAS plots
    cd sas
    export OUT=rjmcmc_output/
    poetry run python3 plot_sinharcsinh_prop_all.py
    for i in $(seq 0 5); do
    	taskset --cpu-list 0-1 poetry run python3 run_RJMCMC_sinharcsinh_traces_af.py run $i $OUT &
    	taskset --cpu-list 2-3 poetry run python3 run_RJMCMC_sinharcsinh_traces_perfect.py run $i $OUT &  
    	taskset --cpu-list 4-5 poetry run python3 run_RJMCMC_sinharcsinh_traces_nf.py run $i $OUT &
    	wait
    done
    poetry run python3 plot_sas_mp_trace.py $OUT
    # uncomment below to obtain plot using previously run chains 
    #poetry run python3 plot_sas_mp_trace.py sas_rjmcmc_chains
    cd ..
    fi
    
    if [ $RUNFA -eq 1 ]; then
    # run FA plots
    cd fa
    poetry run python3 fa_plot_prop.py
    for i in $(seq 0 9); do
    	for j in 0 3; do
    		poetry run python3 generate_fa_NUTS_pySMC_posteriors.py $j $i run 0
    	done
    done
    for proptype in $(seq 0 2); do
    	for i in $(seq 0 9); do
    		for j in 0 3; do
    			poetry run python3 generate_fa_NUTS_pySMC_posteriors.py $j $i bridge $proptype
    		done
    	done
    done
    poetry run python3 fa_plot_bbe_tt.py
    for proptype in 0 2; do
    	for i in $(seq 0 9); do
    		for j in 3; do
    			poetry run python3 fa_rjmcmc.py $j $i reload
    		done
    	done
    done
    poetry run python3 plot_FA_lq_af_rq_rjmcmc_trace_pyplot_b_5.py output/
    cd ..
    fi
    
    if [ $RUNVS -eq 1 ]; then
    # run VS plots
    cd vs
    for i in $(seq 0 79); do
    	for j in 0 1 2 3; do
    		poetry run python3 sample_indiv_smc.py run $i $j
    		poetry run python3 bbe_affine.py reload $i $j
    		poetry run python3 bbe_naive.py reload $i $j
    		poetry run python3 bbe_indiv_tf.py reload $i $j
    		poetry run python3 bbe_cnf.py reload $i $j
    	done
    done
    poetry run python3 plot_bbe_violin_500_4000.py
    poetry run python3 plot_bbe_violin_all.py
    cd ..
    fi

