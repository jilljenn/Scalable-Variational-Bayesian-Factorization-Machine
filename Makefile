all:
	cd src/libfm; make all

libFM:
	cd src/libfm; make libFM

clean:
	# cd src/libfm; make clean
	rm results/mcmc_100k.csv

# all: results/ovbfm_100k results/mcmc_100k results/ovbfm_1M results/mcmc_1M results/ovbfm_10M results/mcmc_10M

results/ovbfm_100k.csv:
	cd bin && time ./libFM -task r -train ../../vae/data/movie100k/movie100k.trainval_libfm -test ../../vae/data/movie100k/movie100k.test_libfm -dim '1,1,20' -method vb_online -batch 200 -iter 200 -rlog ../results/ovbfm_100k.csv

results/mcmc_100k.csv:  # 8.64 s
	cd bin && time ./libFM -verbosity 1 -task r -train ../../vae/data/movie100k/movie100k.trainval_libfm -test ../../vae/data/movie100k/movie100k.test_libfm -dim '1,1,20' -method mcmc -iter 1000 -rlog ../results/mcmc_100k.csv -out ../results/mcmc_pred_100k.csv

results/ovbfm_100k:
	cd bin && time ./libFM -task r -train ../data/sa100k5.train_libfm -test ../data/sa100k5.test_libfm -dim '1,1,20' -method vb_online -batch 200 -iter 200 -rlog ../results/ovbfm_100k

results/mcmc_100k:
	cd bin && time ./libFM -task r -train ../data/sa100k5.train_libfm -test ../data/sa100k5.test_libfm -dim '1,1,20' -method mcmc -iter 200 -rlog ../results/mcmc_100k

results/ovbfm_1M:
	cd bin && ./libFM -task r -train ../data/sa1M5.train_libfm -test ../data/sa1M5.test_libfm -dim '1,1,20' -method vb_online -batch 200 -iter 400 -rlog ../results/ovbfm_1M

results/mcmc_1M:
	cd bin && ./libFM -task r -train ../data/sa1M5.train_libfm -test ../data/sa1M5.test_libfm -dim '1,1,20' -method mcmc -iter 400 -rlog ../results/mcmc_1M

results/ovbfm_10M:
	cd bin && ./libFM -task r -train ../data/sa10M5.train_libfm -test ../data/sa10M5.test_libfm -dim '1,1,20' -method vb_online -batch 200 -iter 400 -rlog ../results/ovbfm_10M

results/mcmc_10M:
	cd bin && ./libFM -task r -train ../data/sa10M5.train_libfm -test ../data/sa10M5.test_libfm -dim '1,1,20' -method mcmc -iter 400 -rlog ../results/mcmc_10M

results/ovbfm_100:
	cd bin && ./libFM -task c -train ../data/movie100.train_libfm -test ../data/movie100.test_libfm -dim '1,1,3' -method vb_online -batch 100 -iter 200 -rlog ../results/ovbfm_100

results/ovbfm_100.csv:
	cd bin && ./libFM -task c -train ../data/movie100.train_libfm -test ../data/movie100.test_libfm -dim '1,1,3' -method vb_online -batch 100 -iter 200 -rlog ../results/ovbfm_100.csv

results/mcmc_100: data/movie100.test_libfm
	cd bin && ./libFM -task c -train ../data/movie100.train_libfm -test ../data/movie100.test_libfm -dim '1,1,3' -method mcmc -iter 200 -rlog ../results/mcmc_100 -out mcmc_pred_100

results/mcmc_100.csv: data/movie100.test_libfm
	cd bin && time ./libFM -task c -train ../data/movie100.train_libfm -test ../data/movie100.test_libfm -dim '1,1,3' -method mcmc -iter 200 -rlog ../results/mcmc_100.csv -out mcmc_pred_100

results/ovbfm_1000: data/movie1000.test_libfm
	cd bin && ./libFM -task c -train ../data/movie1000.train_libfm -test ../data/movie1000.test_libfm -dim '1,1,3' -method vb_online -batch 1 -iter 200 -rlog ../results/ovbfm_100 -out ovbfm_pred_1000

results/mcmc_1000: data/movie1000.test_libfm
	cd bin && ./libFM -task c -train ../data/movie1000.train_libfm -test ../data/movie1000.test_libfm -dim '1,1,3' -method mcmc -iter 40 -rlog ../results/mcmc_1000 -out mcmc_pred_1000
