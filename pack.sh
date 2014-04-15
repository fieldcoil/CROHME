#!/bin/bash

echo "Start zipping..."
cp code/README README

zip CROHME2014_Wang\&Yao_$1.zip README \
	libsvm-3.17.tar.gz \
	code/crohme.py \
	code/InkML.py \
	code/routines.py \
	code/svm_test.py \
	code/svm.py \
	code/svmutil.py \
	code/libsvm.so.2 \
	code/folds.dump \
	code/symb.dump \
	code/PCA_*.dump \
	code/PCA_parsing_*.dump \
	code/svm_model_* \
	code/seg_model_* \
	code/parsing_model_* \
	code/scaling_* \
	code/seg_scaling_* \
	code/parsing_scaling_* \
	code/LICENSE_CROHMElib \
	code/COPYRIGHT_libsvm \
	code/README \
	code/linux\ x86_64/libsvm.so.2 \
	code/macos/libsvm.so.2 \

rm README
