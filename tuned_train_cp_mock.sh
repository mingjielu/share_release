DEVICES_IDS=`python -c "print(' '.join([str(a) for a in range($GPUS_PER_NODE)]))"`
ROCBLAS_FILE=experiment/tuned/rocblas_7B_mbs7_gbs280_tp1_pp1_seq4096_true_true_true_false.yaml
ROCBLAS_LOG=experiment/tuned/rocblas_7B_mbs7_gbs280_tp1_pp1_seq4096_true_true_true_false.log
ROCBLAS_DIR=experiment/tuned/rocblas_7B_mbs7_gbs280_tp1_pp1_seq4096_true_true_true_false

#TEE_OUTPUT=1 TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_LAYER=4 PYTORCH_TUNABLEOP_ENABLED=0 bash train_cp_mock.sh TP=1 CP=1 PP=1 MBS=7 BS=280 TE_FP16=1 MODEL_SIZE=7 SEQ_LENGTH=4096 GEMM_TUNING=0 TOTAL_ITERS=6 2>&1 | grep "\- { rocblas_function:" | uniq > $ROCBLAS_FILE
#
#echo "Run GEMM tunning..."
#afo tune $ROCBLAS_FILE --cuda_device $DEVICES_IDS >& $ROCBLAS_LOG
#mkdir -p $ROCBLAS_DIR
##mv full_tuned*.csv $ROCBLAS_DIR
#mv *.csv $ROCBLAS_DIR

TEE_OUTPUT=1 TORCH_BLAS_PREFER_HIPBLASLT=0  PYTORCH_TUNABLEOP_FILENAME=$ROCBLAS_DIR/afo_tune_device_%d_full.csv PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_ENABLED=1 bash train_cp_mock.sh TP=1 CP=1 PP=1 MBS=7 BS=280 TE_FP16=0 MODEL_SIZE=7 SEQ_LENGTH=4096 GEMM_TUNING=0 ENABLE_PROFILING=1
