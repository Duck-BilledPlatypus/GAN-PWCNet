
def create_model(opt):
    print(opt.model)
    if opt.model == 'wsupervised':
        from .T2model import T2NetModel
        model = T2NetModel()
    elif opt.model == 'supervised':
        from .TaskModel import TNetModel
        model = TNetModel()
    elif opt.model == 'test':
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'tmodel':
        from .TModel import TModel
        model = TModel()
    elif opt.model == 'tpmodel':
        from .TPModel import TPModel
        model = TPModel()
    elif opt.model == 'tpomodel':
        from .TPOModel import TPOModel
        model = TPOModel()
    elif opt.model == 'tpovrmodel':
        from .TPOVRModel import TPOVRModel
        model = TPOVRModel()
    elif opt.model == 'tpovmodel':
        from .TPOVModel import TPOVModel
        model = TPOVModel()
    elif opt.model == 'testtpov':
        from .test_tpov_model import TestTPOVModel
        model = TestTPOVModel()
    elif opt.model == 'testtp':
        from model.test_tp_model import TestTPModel
        model = TestTPModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created." % (model.name()))
    return model