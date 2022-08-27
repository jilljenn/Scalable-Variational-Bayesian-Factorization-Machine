all:
	cd src/libfm; make all

libFM:
	cd src/libfm; make libFM

clean:
	cd src/libfm; make clean

100k:
	cd bin && ./libFM -task r -train ../data/sa100k5.train_libfm -test ../data/sa100k5.test_libfm -dim '1,1,20' -method vb_online -batch 100

1M:
	cd bin && ./libFM -task r -train ../data/sa1M5.train_libfm -test ../data/sa1M5.test_libfm -dim '1,1,20' -method vb_online -batch 100

10M:
	cd bin && ./libFM -task r -train ../data/sa1M5.train_libfm -test ../data/sa1M5.test_libfm -dim '1,1,20' -method vb_online -batch 100 -iter 21
