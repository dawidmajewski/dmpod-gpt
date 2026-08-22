dataset = "shakespeare"

gradient_accumulation_steps = 1
batch_size = 8

learning_rate = 3e-4
max_iters = 1000
lr_decay_iters = 1000
min_lr = 3e-5
warmup_iters = 100

eval_interval = 100
eval_iters = 50
log_interval = 10
always_save_checkpoint = True
