### How to use it
#### Set arguments in args.py
1. Set crop_size is a tuple now instead of a int in original version
    - In training, crop size need to be as large as image size in training depends on the gpu memory
    - In elva/inference, crop size must be larger than image size
2. Set batch_size
3. Set dataset name