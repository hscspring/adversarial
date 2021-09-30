# README

本文档主要描述在文本分类任务上使用不同对抗训练方法的实验。文本分类基线模型为 TextCNN。

## 数据说明

本次实验数据集来自[THUCTC: 一个高效的中文文本分类工具](http://thuctc.thunlp.org/)，数据包括财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐等 10 个类别。

为了便于快速测试，本次通过均匀采样使用小部分数据集：

- 训练集：1.8 万条
- 验证集：1000 条
- 测试集：1000 条

## 实验配置

所有的对抗训练都是针对 Embedding 进行扰动，基本配置如下：

| 模型 | Replay | epsilon | alpha |
| ---- | ------ | ------- | ----- |
| FGSM | /      | 0.05    | /     |
| PGD  | 3      | 1.0     | 0.3   |
| Free | 3      | 1.0     | /     |

其他参数如下：

- Adam，学习率 1e-3
- 每 100 Step 验证一次，并保存结果，如果满足提前终止条件则停止训练
- 提前终止条件为：连续 300 个 Step 验证集 Loss 未下降
- Batch 大小为 128
- 句子长度 32

## 实验结果

最终得到的结果如下：

| 模型            | P      | R      | F1     |
| --------------- | ------ | ------ | ------ |
| Base（TextCNN） | 0.8567 | 0.8608 | 0.8573 |
| FGSM            | 0.8752 | 0.8710 | 0.8709 |
| PGD             | 0.8552 | 0.8585 | 0.8551 |
| Free            | 0.8520 | 0.8508 | 0.8481 |

## 使用方法

```bash
$ python run_adv.py --model TextCNN --adv FGSM
```

## 附：训练日志

**base**


```bash
Epoch [1/20]
Iter:      0,  Train Loss:   2.3,  Train Acc: 11.72%,  Val Loss:   2.4,  Val Acc:  8.60%,  Time: 0:00:02 *
Iter:    100,  Train Loss:  0.83,  Train Acc: 75.00%,  Val Loss:  0.68,  Val Acc: 78.90%,  Time: 0:01:08 *
Epoch [2/20]
Iter:    200,  Train Loss:  0.63,  Train Acc: 80.47%,  Val Loss:  0.56,  Val Acc: 82.50%,  Time: 0:02:04 *
Epoch [3/20]
Iter:    300,  Train Loss:  0.35,  Train Acc: 89.06%,  Val Loss:  0.52,  Val Acc: 83.40%,  Time: 0:03:15 *
Iter:    400,  Train Loss:  0.35,  Train Acc: 91.41%,  Val Loss:   0.5,  Val Acc: 83.50%,  Time: 0:04:18 *
Epoch [4/20]
Iter:    500,  Train Loss:  0.26,  Train Acc: 92.19%,  Val Loss:   0.5,  Val Acc: 84.60%,  Time: 0:05:31
Epoch [5/20]
Iter:    600,  Train Loss:  0.15,  Train Acc: 95.31%,  Val Loss:  0.48,  Val Acc: 85.00%,  Time: 0:06:51 *
Iter:    700,  Train Loss:  0.16,  Train Acc: 92.97%,  Val Loss:  0.49,  Val Acc: 85.00%,  Time: 0:08:10
Epoch [6/20]
Iter:    800,  Train Loss:  0.12,  Train Acc: 95.31%,  Val Loss:  0.52,  Val Acc: 84.20%,  Time: 0:09:35
Epoch [7/20]
Iter:    900,  Train Loss:  0.13,  Train Acc: 93.75%,  Val Loss:  0.53,  Val Acc: 85.00%,  Time: 0:12:35
No optimization for a long time, auto-stopping...
Test Loss:  0.47,  Test Acc: 85.80%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.8571    0.9000    0.8780       100
       realty     0.9320    0.8727    0.9014       110
       stocks     0.7978    0.7030    0.7474       101
    education     0.8021    0.9277    0.8603        83
      science     0.8144    0.7453    0.7783       106
      society     0.8913    0.9011    0.8962        91
     politics     0.8585    0.8585    0.8585       106
       sports     0.8119    0.9318    0.8677        88
         game     0.8919    0.8761    0.8839       113
entertainment     0.9100    0.8922    0.9010       102

     accuracy                         0.8580      1000
    macro avg     0.8567    0.8608    0.8573      1000
 weighted avg     0.8589    0.8580    0.8571      1000

Confusion Matrix...
[[90  0  3  3  1  1  1  0  1  0]
 [ 3 96  1  3  0  1  1  2  1  2]
 [ 8  2 71  1  6  1  9  1  1  1]
 [ 0  1  1 77  1  0  1  1  0  1]
 [ 1  1  8  4 79  3  1  2  5  2]
 [ 1  0  0  3  2 82  0  2  0  1]
 [ 1  1  4  2  3  3 91  1  0  0]
 [ 0  0  1  1  0  0  1 82  2  1]
 [ 1  1  0  1  4  0  0  6 99  1]
 [ 0  1  0  1  1  1  1  4  2 91]]
```

**fgsm**

```bash
Epoch [1/20]
Iter:      0,  Train Loss:   2.4,  Train Acc:  0.78%,  Val Loss:   2.5,  Val Acc:  8.60%,  Time: 0:00:03 *
Iter:    100,  Train Loss:   1.5,  Train Acc: 53.91%,  Val Loss:  0.83,  Val Acc: 75.00%,  Time: 0:01:55 *
Epoch [2/20]
Iter:    200,  Train Loss:   1.3,  Train Acc: 60.94%,  Val Loss:  0.66,  Val Acc: 79.30%,  Time: 0:03:53 *
Epoch [3/20]
Iter:    300,  Train Loss:   1.0,  Train Acc: 68.75%,  Val Loss:  0.58,  Val Acc: 83.30%,  Time: 0:06:16 *
Iter:    400,  Train Loss:   1.1,  Train Acc: 64.06%,  Val Loss:  0.55,  Val Acc: 83.60%,  Time: 0:08:59 *
Epoch [4/20]
Iter:    500,  Train Loss:   1.1,  Train Acc: 67.97%,  Val Loss:  0.51,  Val Acc: 83.70%,  Time: 0:11:47 *
Epoch [5/20]
Iter:    600,  Train Loss:  0.92,  Train Acc: 77.34%,  Val Loss:  0.49,  Val Acc: 85.20%,  Time: 0:14:30 *
Iter:    700,  Train Loss:  0.98,  Train Acc: 71.88%,  Val Loss:  0.48,  Val Acc: 86.10%,  Time: 0:17:13 *
Epoch [6/20]
Iter:    800,  Train Loss:   0.7,  Train Acc: 76.56%,  Val Loss:  0.47,  Val Acc: 85.90%,  Time: 0:19:59 *
Epoch [7/20]
Iter:    900,  Train Loss:  0.88,  Train Acc: 78.12%,  Val Loss:  0.47,  Val Acc: 85.20%,  Time: 0:22:44
Epoch [8/20]
Iter:   1000,  Train Loss:  0.69,  Train Acc: 80.47%,  Val Loss:  0.47,  Val Acc: 85.20%,  Time: 0:25:32 *
Iter:   1100,  Train Loss:  0.67,  Train Acc: 80.47%,  Val Loss:  0.45,  Val Acc: 85.20%,  Time: 1:18:46 *
Epoch [9/20]
Iter:   1200,  Train Loss:  0.69,  Train Acc: 78.12%,  Val Loss:  0.45,  Val Acc: 85.60%,  Time: 1:21:24 *
Epoch [10/20]
Iter:   1300,  Train Loss:  0.51,  Train Acc: 85.16%,  Val Loss:  0.45,  Val Acc: 85.50%,  Time: 1:23:10 *
Iter:   1400,  Train Loss:  0.47,  Train Acc: 82.81%,  Val Loss:  0.45,  Val Acc: 86.80%,  Time: 1:25:33 *
Epoch [11/20]
Iter:   1500,  Train Loss:  0.45,  Train Acc: 85.16%,  Val Loss:  0.46,  Val Acc: 86.40%,  Time: 1:27:55
Epoch [12/20]
Iter:   1600,  Train Loss:  0.48,  Train Acc: 85.94%,  Val Loss:  0.45,  Val Acc: 85.70%,  Time: 1:30:27
Epoch [13/20]
Iter:   1700,  Train Loss:  0.35,  Train Acc: 88.28%,  Val Loss:  0.46,  Val Acc: 86.90%,  Time: 1:33:12
No optimization for a long time, auto-stopping...
Test Loss:  0.42,  Test Acc: 87.10%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.8824    0.9000    0.8911       100
       realty     0.9259    0.9091    0.9174       110
       stocks     0.8081    0.7921    0.8000       101
    education     0.9268    0.9157    0.9212        83
      science     0.8617    0.7642    0.8100       106
      society     0.9111    0.9011    0.9061        91
     politics     0.8158    0.8774    0.8455       106
       sports     0.7589    0.9659    0.8500        88
         game     0.9375    0.7965    0.8612       113
entertainment     0.9126    0.9216    0.9171       102

     accuracy                         0.8710      1000
    macro avg     0.8741    0.8743    0.8720      1000
 weighted avg     0.8752    0.8710    0.8709      1000

Confusion Matrix...
[[ 90   0   4   1   1   0   2   1   0   1]
 [  3 100   3   0   0   0   1   1   0   2]
 [  5   2  80   0   2   1  10   1   0   0]
 [  2   1   0  76   0   0   2   2   0   0]
 [  1   1   7   2  81   4   2   2   4   2]
 [  0   0   1   2   2  82   1   3   0   0]
 [  0   1   4   1   3   2  93   2   0   0]
 [  0   0   0   0   0   0   1  85   0   2]
 [  1   3   0   0   5   1   0  11  90   2]
 [  0   0   0   0   0   0   2   4   2  94]]
```

**pgd**

```bash
Epoch [1/20]
Iter:      0,  Train Loss:   2.3,  Train Acc: 10.94%,  Val Loss:   2.5,  Val Acc:  8.60%,  Time: 0:00:04 *
Iter:    100,  Train Loss:  0.87,  Train Acc: 73.44%,  Val Loss:  0.66,  Val Acc: 79.50%,  Time: 0:03:35 *
Epoch [2/20]
Iter:    200,  Train Loss:  0.55,  Train Acc: 82.81%,  Val Loss:  0.55,  Val Acc: 83.30%,  Time: 0:06:58 *
Epoch [3/20]
Iter:    300,  Train Loss:  0.42,  Train Acc: 88.28%,  Val Loss:  0.51,  Val Acc: 84.40%,  Time: 0:10:23 *
Iter:    400,  Train Loss:  0.36,  Train Acc: 89.06%,  Val Loss:  0.49,  Val Acc: 84.10%,  Time: 0:13:52 *
Epoch [4/20]
Iter:    500,  Train Loss:  0.32,  Train Acc: 88.28%,  Val Loss:   0.5,  Val Acc: 84.60%,  Time: 0:17:48
Epoch [5/20]
Iter:    600,  Train Loss:  0.17,  Train Acc: 94.53%,  Val Loss:  0.49,  Val Acc: 84.60%,  Time: 0:22:17 *
Iter:    700,  Train Loss:  0.24,  Train Acc: 92.19%,  Val Loss:  0.51,  Val Acc: 84.30%,  Time: 0:27:21
Epoch [6/20]
Iter:    800,  Train Loss:  0.13,  Train Acc: 98.44%,  Val Loss:  0.53,  Val Acc: 84.20%,  Time: 0:31:28
Epoch [7/20]
Iter:    900,  Train Loss:  0.13,  Train Acc: 96.88%,  Val Loss:  0.52,  Val Acc: 85.70%,  Time: 0:35:18
No optimization for a long time, auto-stopping...
Test Loss:  0.46,  Test Acc: 85.60%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.8349    0.9100    0.8708       100
       realty     0.9259    0.9091    0.9174       110
       stocks     0.8046    0.6931    0.7447       101
    education     0.8242    0.9036    0.8621        83
      science     0.8144    0.7453    0.7783       106
      society     0.7925    0.9231    0.8528        91
     politics     0.8447    0.8208    0.8325       106
       sports     0.8542    0.9318    0.8913        88
         game     0.9000    0.8761    0.8879       113
entertainment     0.9570    0.8725    0.9128       102

     accuracy                         0.8560      1000
    macro avg     0.8552    0.8585    0.8551      1000
 weighted avg     0.8575    0.8560    0.8550      1000

Confusion Matrix...
[[ 91   0   3   1   1   4   0   0   0   0]
 [  2 100   1   2   0   1   1   0   1   2]
 [ 10   2  70   1   7   1   9   0   0   1]
 [  2   2   0  75   0   1   1   2   0   0]
 [  1   1   8   4  79   5   1   2   5   0]
 [  1   0   0   2   2  84   1   1   0   0]
 [  1   1   4   4   3   5  87   1   0   0]
 [  0   1   1   1   0   1   0  82   2   0]
 [  1   1   0   0   4   1   1   5  99   1]
 [  0   0   0   1   1   3   2   3   3  89]]
```

**free**

```bash
Epoch [1/20]
Iter:      0,  Train Loss:   2.1,  Train Acc: 27.34%,  Val Loss:   2.2,  Val Acc: 21.20%,  Time: 0:00:05 *
Iter:    100,  Train Loss:  0.75,  Train Acc: 79.69%,  Val Loss:  0.64,  Val Acc: 79.70%,  Time: 0:04:23 *
Epoch [2/20]
Iter:    200,  Train Loss:  0.46,  Train Acc: 87.50%,  Val Loss:  0.56,  Val Acc: 83.00%,  Time: 0:07:52 *
Epoch [3/20]
Iter:    300,  Train Loss:  0.36,  Train Acc: 89.06%,  Val Loss:  0.52,  Val Acc: 83.80%,  Time: 0:11:26 *
Iter:    400,  Train Loss:   0.3,  Train Acc: 88.28%,  Val Loss:  0.54,  Val Acc: 83.30%,  Time: 0:14:17
Epoch [4/20]
Iter:    500,  Train Loss:  0.19,  Train Acc: 96.09%,  Val Loss:   0.6,  Val Acc: 81.90%,  Time: 0:16:56
Epoch [5/20]
Iter:    600,  Train Loss:  0.23,  Train Acc: 92.19%,  Val Loss:  0.62,  Val Acc: 82.20%,  Time: 0:19:45
No optimization for a long time, auto-stopping...
Test Loss:  0.49,  Test Acc: 84.80%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.8257    0.9000    0.8612       100
       realty     0.9474    0.8182    0.8780       110
       stocks     0.8571    0.6535    0.7416       101
    education     0.8636    0.9157    0.8889        83
      science     0.7456    0.8019    0.7727       106
      society     0.9186    0.8681    0.8927        91
     politics     0.8426    0.8585    0.8505       106
       sports     0.7455    0.9318    0.8283        88
         game     0.8981    0.8584    0.8778       113
entertainment     0.8762    0.9020    0.8889       102

     accuracy                         0.8480      1000
    macro avg     0.8520    0.8508    0.8481      1000
 weighted avg     0.8534    0.8480    0.8474      1000

Confusion Matrix...
[[90  0  2  2  2  2  0  1  1  0]
 [ 5 90  2  2  2  0  2  2  2  3]
 [10  2 66  1  9  0 10  3  0  0]
 [ 2  0  0 76  0  0  0  2  0  3]
 [ 1  1  3  2 85  2  1  3  5  3]
 [ 1  0  0  2  5 79  1  2  0  1]
 [ 0  1  4  1  3  3 91  3  0  0]
 [ 0  0  0  0  2  0  1 82  1  2]
 [ 0  1  0  1  5  0  0  8 97  1]
 [ 0  0  0  1  1  0  2  4  2 92]]
```

