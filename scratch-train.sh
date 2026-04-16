./prepare_mixed_corpus.sh

./tokenize_corpus.sh \
    training/corpus_mixed \
    learned/corpus_mixed_tokenized_65536 \
    --max-vocab-size 65536

./python/1_train_gpt_all_works.py \
    --token-dir learned/corpus_mixed_tokenized_65536 \
    --output-dir learned/gpt_corpus_mixed_65536 \
    --context-len 256 \
    --d-model 384 \
    --n-heads 8 \
    --n-layers 4 \
    --d-ff 1024 \
    --max-vocab-size 65536 \
    --batch-size 8 \
    --grad-accum-steps 2 \
    --num-workers 6 \
    --persistent-workers \
    --prefetch-factor 6 \
    --eval-batches 50 \
    --log-every 2000 \
    --save-every 6000 \
    --no-compile-model