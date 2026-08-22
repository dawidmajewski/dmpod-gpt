dataset = "shakespeare_char"

gradient_accumulation_steps = 1
batch_size = 64

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
warmup_iters = 100

eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = True
