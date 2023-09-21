最好的结果是 test 和 train 0.78,0.78

model = XGBRegressor(
    # objective=fair_obj, #use the customized loss function
    # tree_method='gpu_hist', #if use the gpu, 'gpu_hist' can accelerate the speed
    n_estimators=10000,
    max_depth=18,
    learning_rate=0.1,
    gamma=0.9,
    min_child_weight=9,
    max_delta_step=0.9,
    subsample=0.9,
    colsample_bytree=1,
    reg_alpha=0.5,
    reg_lambda=0.1,
    nthread=-1
    )
不进行五折交叉验证，然后shuffle =ture，岁