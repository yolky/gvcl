from networks.gvcl_model_classes import MultiHeadFiLMCNN

class BabyNetNoFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((1,32,32), [(16,3), 'pool', (32,3), 'pool'], [100], heads)

class BabyNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((1,32,32), [(16,3), 'pool', (32,3), 'pool'], [100], heads, film_type = 'point')

class ZenkeNetNoFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(32,3), (32,3), 'pool', (64,3), (64,3), 'pool'], [512], heads)

class ZenkeNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(32,3), (32,3), 'pool', (64,3), (64,3), 'pool'], [512], heads, film_type = 'point')

class SMNISTNetNoFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((1,28,28), [], [256,256], heads)

class SMNISTNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((1,28,28), [], [256,256], heads, film_type = 'point')

class AlexNetNoFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(64, 4, 0), 'pool', (128, 3, 0), 'pool', (256, 2, 0), 'pool'], [2048,2048], heads, prior_var = 0.01)

class AlexNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(64, 4, 0), 'pool', (128, 3, 0), 'pool', (256, 2, 0), 'pool'], [2048,2048], heads, film_type = 'point', prior_var = 0.01)
