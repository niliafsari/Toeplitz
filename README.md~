How to run the code on real dataset:

python extract_realData.py gb057_1.input_baseline258_freq_03_pol_all.rebint.1.rebined 2048 330 100 100 32 8
mpirun -np 2 python run_real.py gate0_numblock_64_meff_40_offsetn_100_offsetm_100

To plot results:
python plot_real_basic.py gate0_numblock_32_meff_32_offsetn_100_offsetm_100

What above commands does:

extract_realData.py: preprocessed the real binary data file, padded zeros.
	inputs:
		name of raw binary file: e.g., gb057_1.input_baseline258_freq_03_pol_all.rebint.1.rebined
		number of frequency channels (num_rows): e.g., 2048
		number of timesteps (num_columns): e.g., 330
		offset along frequency dimension: e.g., 100
		offset along time dimension: e.g., 100
		size of frequency dimension (want to processed only a chunk of available data): e.g., 32
		size of time dimension (want to processed only a chunk of available data): e.g., 8
	output: 
	saves three files in processedData/ folder: 
		gate0_numblock_32_meff_32_offsetn_100_offsetm_100.dat
		gate0_numblock_32_meff_32_offsetn_100_offsetm_100_dynamic.npy
		gate0_numblock_32_meff_32_offsetn_100_offsetm_100_toep.npy
run_real.py: applies the algorithm on processed data, it can be run in parallel using mpirun and setting -np to the number of processors
	input:
		name of processed data file without extension: e.g., gate0_numblock_64_meff_40_offsetn_100_offsetm_100
	output: 
	saves a file in results/ folder:	
		gate0_numblock_32_meff_32_offsetn_100_offsetm_100_uc.npy
plot_real_basic.py: plots results
	input:
		name of processed data file without extension: e.g., gate0_numblock_64_meff_40_offsetn_100_offsetm_100	
					



	

