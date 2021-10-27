# DistilledCTR
CTR prediction using model distillation

```
# set up
git clone https://github.com/aljo-jose/DistilledCTR.git
cd DistilledCTR
python setup.py develop
```

# Training
```
# training individual models
python distilled_ctr/base_trainer.py  --dataset_name=avazu --dataset_path=data/avazu/train --model_name=wd --experiment=experiment_name --epoch=10 --workers=8 --learning_rate=0.001 --batch_size=1024 --weight_decay=1e-6

# training ensemble
python distilled_ctr/base_trainer.py  --dataset_name=avazu --dataset_path=data/avazu/train --model_name=gated_ensemble --experiment=experiment_name --epoch=10 --workers=8 --learning_rate=0.001 --batch_size=1024 --weight_decay=1e-6

# distillation training
python distilled_ctr/distill_trainer.py  --dataset_name=avazu --dataset_path=data/avazu/train --teacher=gated_ensemble --student=dnn --experiment=experiment_name --epoch=10 --workers=8 --learning_rate=0.001 --batch_size=1024 --weight_decay=1e-6

```

```
# tensorboard
tensorboard --logdir=logs/ --host=0.0.0.0
```


