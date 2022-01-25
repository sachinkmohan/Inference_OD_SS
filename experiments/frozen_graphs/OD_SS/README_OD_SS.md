
#### Tested Individually

Test date : 25-01-22, 16:00

For OD
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
In individual run -> GPU - 14.7/32.7GB

For SS
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
In individual run -> GPU - 11.6/32.7GB


Combined run -> GPU - 23.8/32.7GB



-------------------------------------------------------------
#### On TRT graph run individually

For OD
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
In individual run -> GPU - 15.0/32.7GB

For SS
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
In individual run -> GPU - 12.4/32.7GB



Combined run -> GPU - 24.8/32.7GB


Please see the excel sheet for further details
