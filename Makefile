all:
	cd src/libfm; make all

libFM:
	cd src/libfm; make libFM

clean:
	cd src/libfm; make clean

all: results/ovbfm_100k results/mcmc_100k results/ovbfm_1M results/mcmc_1M results/ovbfm_10M results/mcmc_10M

results/ovbfm_100k:
	cd bin && ./libFM -task r -train ../data/sa100k5.train_libfm -test ../data/sa100k5.test_libfm -dim '1,1,3' -method vb_online -batch 200 -iter 200 -rlog ../results/ovbfm_100k

results/mcmc_100k:
	cd bin && ./libFM -task r -train ../data/sa100k5.train_libfm -test ../data/sa100k5.test_libfm -dim '1,1,3' -method mcmc -iter 200 -rlog ../results/mcmc_100k

results/ovbfm_1M:
	cd bin && ./libFM -task r -train ../data/sa1M5.train_libfm -test ../data/sa1M5.test_libfm -dim '1,1,20' -method vb_online -batch 200 -iter 400 -rlog ../results/ovbfm_1M

results/mcmc_1M:
	cd bin && ./libFM -task r -train ../data/sa1M5.train_libfm -test ../data/sa1M5.test_libfm -dim '1,1,20' -method mcmc -iter 400 -rlog ../results/mcmc_1M

results/ovbfm_10M:
	cd bin && ./libFM -task r -train ../data/sa10M5.train_libfm -test ../data/sa10M5.test_libfm -dim '1,1,20' -method vb_online -batch 200 -iter 400 -rlog ../results/ovbfm_10M

results/mcmc_10M:
	cd bin && ./libFM -task r -train ../data/sa10M5.train_libfm -test ../data/sa10M5.test_libfm -dim '1,1,20' -method mcmc -iter 400 -rlog ../results/mcmc_10M
