# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif, clamp
from tensorboardX import SummaryWriter
from pnlp import Dict as ADict


def get_args():
    return ADict({
        "epochs": 10,
        "lr_schedule": "cyclic", # multistep
        "lr_min": 0.0,
        "lr_max": 0.04, 
        "weight_decay": 5e-4,
        "momentum": 0.9, 
        "epsilon": 16,          ## 8
        "minibatch_replays": 2  ## 8
    })


def train(config, model, train_iter, dev_iter, test_iter):
    ad_args = get_args()
    start_time = time.time()

    if config.embedding_pretrained is not None:
        print("Using pretrained embedding...")
        embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=config.fix_embedding)
    else:
        embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
    
    model.train()

    if config.fix_embedding:
        params = model.parameters()
    else:
        embedding.train()
        params = [{"params": model.parameters()}, {"params": embedding.parameters()}]
    optimizer = torch.optim.SGD(
        params, lr=ad_args.lr_max, momentum=ad_args.momentum, weight_decay=ad_args.weight_decay
    )
    # optimizer = torch.optim.Adam(
    #     [{"params": model.parameters()}, {"params": embedding.parameters()}], 
    #     lr=config.learning_rate, weight_decay=ad_args.weight_decay
    # )

    criterion = nn.CrossEntropyLoss()

    delta = torch.zeros(config.batch_size, config.pad_size, config.embed)
    delta.requires_grad = True

    lr_steps = ad_args.epochs * len(train_iter) * ad_args.minibatch_replays
    msg = 'Epochs: {0},  Train Data Length: {1},  Steps: {2}'
    print(msg.format(ad_args.epochs, len(train_iter), lr_steps))
    
    # if ad_args.lr_schedule == 'cyclic':
    #     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=ad_args.lr_min, max_lr=ad_args.lr_max, step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    # elif ad_args.lr_schedule == 'multistep':
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
    # ??????????????????????????????epoch???????????? = gamma * ?????????
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    total_batch = 0  # ?????????????????????batch
    dev_best_loss = float('inf')
    last_improve = 0  # ?????????????????????loss?????????batch???
    flag = False  # ????????????????????????????????????
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(ad_args.epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, ad_args.epochs))
        for i, (trains, labels) in enumerate(train_iter):
            # 128??? 32??? 300
            X = embedding(trains[0])
            mu = X.mean(dim=(0, 2), keepdim=True)
            std = X.std(dim=(0, 2), keepdim=True)
            epsilon = (ad_args.epsilon / config.n_vocab) / std
            upper_limit = ((1 - mu)/ std)
            lower_limit = ((0 - mu)/ std)
            for _ in range(ad_args.minibatch_replays):
                outputs = model(X + delta[: X.size(0)])
                loss = criterion(outputs, labels)
                # loss = F.cross_entropy(outputs, labels)
                # optimizer.zero_grad()
                # ???????????????????????? opt?????????????????????????????? opt ??????opt.zero_grad() ????????? model.zero_grad()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                grad = delta.grad.detach()
                delta.data = clamp(delta + epsilon * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)

                optimizer.step()
                delta.grad.zero_()
                # scheduler.step()

            if total_batch % 100 == 0:
                # ??????????????????????????????????????????????????????
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, embedding, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path + "free.ckpt")
                    torch.save(embedding.state_dict(), config.save_embedding_path + "free.ckpt")
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                if not config.fix_embedding:
                    embedding.train()
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # ?????????loss??????1000batch????????????????????????
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, embedding, model, test_iter)


def test(config, embedding, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path + "free.ckpt"))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, embedding, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, embedding, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            embed = embedding(texts[0])
            outputs = model(embed)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)