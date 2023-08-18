# Engine_PHM_2008

## build image

```bash
docker build  -t enginer_phm_2008 .
```

## run the example

```bash 
docker run --gpus all -v $(pwd):/exp -ti enginer_phm_2008 /bin/bash

# inside container
cd exp/
python main.py

# ====================================== Training Epoch 1 =====================================
# 以引擎 unit: 164 做為training data.
# 以引擎 unit: 157 做為validation data.
# 2020-08-30 16:29:46.700029: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
# 2020-08-30 16:29:46.826877: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
# 13/13 - 1s - loss: 9234.4268 - accuracy: 0.0049 - val_loss: 7553.2451 - val_accuracy: 0.0051
# Running function: RNN_model_training  cost time: 3.975 seconds.
# [9234.4267578125] [7553.2451171875]
# ======================================= Training Epoch 2 =====================================
# 以引擎 unit: 39 做為training data.
# 以引擎 unit: 131 做為validation data.
# 10/10 - 1s - loss: 5280.0518 - accuracy: 0.0064 - val_loss: 7563.7578 - val_accuracy: 0.0039
# Running function: RNN_model_training  cost time: 3.214 seconds.
# [5280.0517578125] [7563.7578125]
# ======================================= Training Epoch 3 =====================================
# 以引擎 unit: 152 做為training data.
# 以引擎 unit: 125 做為validation data.
# 13/13 - 1s - loss: 5748.6797 - accuracy: 0.0048 - val_loss: 5626.8765 - val_accuracy: 0.0042
# Running function: RNN_model_training  cost time: 2.834 seconds.
# [5748.6796875] [5626.87646484375]
# ======================================= Training Epoch 4 =====================================
# 以引擎 unit: 83 做為training data.
# 以引擎 unit: 131 做為validation data.
# 15/15 - 1s - loss: 4574.1890 - accuracy: 0.0044 - val_loss: 4365.0757 - val_accuracy: 0.0039
# Running function: RNN_model_training  cost time: 3.242 seconds.
# [4574.18896484375] [4365.07568359375]
# ======================================= Training Epoch 5 =====================================
# 以引擎 unit: 129 做為training data.
# 以引擎 unit: 217 做為validation data.
# 9/9 - 1s - loss: 2105.6060 - accuracy: 0.0070 - val_loss: 4033.5544 - val_accuracy: 0.0035
# Running function: RNN_model_training  cost time: 3.283 seconds.
# [2105.60595703125] [4033.554443359375]
# ======================================= Training Epoch 6 =====================================
```