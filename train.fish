#! /bin/env fish

set d_m 256

set DATADIR "$argv[1]"
set RESDIR $argv[2]/$d_m

if test ! -d $DATADIR
    echo "data directory '$DATADIR' does not exist or is not a directory"
    exit 1
end


set max_lr (math "640 / $d_m * 0.003")
if test $max_lr -gt 0.01
    set max_lr 0.01
end

mkdir -p $RESDIR

python train.py $DATADIR \
    --arch delight_transformer_wmt14_en_de \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --keep-last-epochs 9 \
    --max-tokens 4096 \
    --lr-scheduler cosine --lr-shrink 1 --max-lr $max_lr --lr 1e-7 --min-lr 1e-9 \
    --max-update 30000 --warmup-updates 10000 \
    --t-mult 1 \
    --delight-emb-map-dim 128 \
    --delight-emb-out-dim $d_m \
    --delight-enc-min-depth 4 --delight-enc-max-depth 8 --delight-enc-width-mult 2 \
    --delight-dec-min-depth 4 --delight-dec-max-depth 8 --delight-dec-width-mult 2 \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $RESDIR \
    | tee -a $RESDIR/logs.txt
