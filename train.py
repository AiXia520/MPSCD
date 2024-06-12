def train(cfg, epoch, train_loader, learner):
    print("Epoch:%2d" % epoch, end="  ")
    learner.train()
    for i, (input, target) in enumerate(train_loader):
        if cfg.method =="SKD" and cfg.teacher_pretrained==True:
            inputData = input[0]
            inp = input[1]
            inputData = inputData.cuda()
            target = target.cuda()
            loss, output = learner.learn(inputData, target, inp=inp)
        else:
            inputData = input
            inputData = inputData.cuda()
            target = target.cuda()
            loss, output = learner.learn(inputData, target)
        if i == len(train_loader) - 1:
            print("[{}/{}], LR {:.4e}, Loss: {:.4e}"
                  .format(i + 1, len(train_loader), learner.optimizer.param_groups[0]["lr"], loss.item()), end="  ")
