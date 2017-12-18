# hyperspectral_exp_orz

*author：Luo Ya'nan*

* ***caffe_try***

    **Classification use caffe's python API.**

    **!!!not correct yet.**

    - findTheSameData: Use for validating correction.
    - readLMDB: Transfrom data from lmdb to numpy.
    - save_feature: Save fc's feature maps and predicitions.
    - validate_param: Validate whether the paramerters of the deploy and train_test prototxt's net are the same.


* ***mnist***

    **Try mnist data set use caffe's python API, get correct result.**

   -  load_mnist_data: Transform mnist data into the visiable form.
   -  myload_mnist_data: Change some of the load data function, 具体什么忘记了...太久没用了
   -  predict_label: Use API to batch classify test data.
   -  mnist_solver: Train net hyperparams.
   -  mnist_train_test: Train net model.
   -   mnist_deploy: Prediction use, which is not include data layer.
   -   mnist_mnist: Train net use Caffe command.

* ***tf_try***

    **Classification use Tensorflow.**
    - matlab_plot
       - plotting: Plorring curves.
       - read_test:  Read test accuracy, loss, etc.
       - read_train: Read train accuracy, loss, etc.

    - python_analyze
       - center_or_border_statistics: Get center and border samples statistics
       - data_analysis: Get data mean, std, etc. And plot corresponding curves. 
       - plotting: Plotting accuracy and loss curves verses iterations.
       - read_data: Get the data after train and test.
       - t_sne: Dimensionality reduction.
     
  
    - data_preprocessing: Extract and divide the original data set into train and test data set according to ratio.
    - data_preprocess_pos: Data preprocessing increases save position information.
    - load_data_tricks: Add class-0 for train.
    - generate_gaussian_noise: Generate gaussian noise.
    

    - test: Validate whether the data extract is correct with BPN net model.
    

    - deep_cnn: Deep net model.
    - original_cnn: Original net model.
    - train_original: Train original method.
    - train_deep: Train deep net method.
  

    - cnxgboost: Use cnn-fc output as feature to train xgboost.
    

    - train_test: 没有在做了，不完整的代码......
    