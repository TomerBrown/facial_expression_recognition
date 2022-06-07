from utils.loops import evaluate
from finetune import fine_tune
import experiments
from evaluate import evaluate

if __name__ == '__main__':

    train_df, val_df, test_df, name = experiments.experiment3()
    #vgg_model = fine_tune(train_df, val_df, name ,n_epochs=1)
    #evaluate(vgg_model,train_df, val_df, test_df, name)


