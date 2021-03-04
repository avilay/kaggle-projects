# Learnings

  * Seperate the data exploration notebook from the model building notebook.
  * Distill the data exploration in a well defined feature engineering function that can be easily changed to try out different feature engineering strategies.
  * Define a baseline model and do a wire check by running a few instances from the dataset through the model and then running the output and targets through the loss function as well as the metrics functions.
  * Ensure that metrics functions are outputting standard Python `float` types. For some reason the json serializer does not work with `np.float32`. It does work with `np.float64` though.
  * When creating the trainset and valset, remember to first create the full dataset and then apply `random_split`. After I have tuned the hyperparams I'll need to train on the full dataset before running the test set.
  * Import both the csv `ml_loger` and the stdout `ml_logger`. During training I'll use csv logger with no stdout. When I am training the full dataset I usually like using the stdout logger so that I don't pollute the experiment logs.
  * Use a `model_factory` dict to specify the model that I want to use and use that in the `build_trainer` function. As I keep defining new models keep updating the `model_factory` dict.
  * Remeber to load the testset when doing data exploration. Sometimes there are surprises in there.
  
  
# Musings

There are two types of experiments that I conduct. One is with trying out different features and another is with trying out different models. My process for trying out different models is pretty streamlined. But trying out new features needs some tlc. One idea I can try is to have different feature engineering functions as part of my training notebook. Similar to the `model_factory` have a `feature_factory` that will call different functions which implement different feature engineering strategies. This means that I don't pickle the datasets, but rather prepare them during each run. While this works for small datasets, its going to get annoying for larger datasets. I can implement these functions as cache functions where they will look on disk for existing datasets and will only preprocess if needed.

After I have ironed out the wrinkles in my training pipelie, ALWAYS make it into a stand-alone program. See `cifaf` as a example. This will let me run multiple runs in parallel when I am searching for the right hyper params. When tuning the hyper params, I typically follow this process:

  1. Manual search
In this I usually first try out extreme values and then depending on the performance, will try out a middle value. Typically I should have two notebooks open for this - one to run `compare` and another to run `analyze` on ad-hoc runs. I should have the first cell in the `compare` notebook as a Markdown cell for capturing my notes during this search.

  2. Automated search
After having gotten a sense of the parameter bounds, I will be in a good position to create the hyper parameter spec. Again, I should have two notebooks open for this - one to run `compare` and another to `analyze`. I can just use the `analyze` notebook from the manual search. As usual the `compare` notebook should have the first cell as a markdown cell for my running notes.