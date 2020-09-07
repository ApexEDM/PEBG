# PEBG

This repository contains the source code of the mdoel PEBG in our paper "Improving Knowledge Tracing via Pre-training Question Embeddings", which is accepted by IJCAI 2020.

## Data Preparation
### Assist09
- data_assist09.py: Data pre-process. You should download original assist09 dataset from [here](https://drive.google.com/file/d/1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE/view). Details are [here](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010).

### EdNet
- ednet/data.py: Data pre-process for EdNet dataset. 
we use the 'ednet-kt1' dataset. You can download it [here](https://drive.google.com/file/d/1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw/view). The question information file is [here](https://drive.google.com/file/d/117aYJAWG3GU48suS66NPaB82HwFj6xWS/view). For more information about [EdNet](https://github.com/riiid/ednet)

Once you have the ednet-kt1 dataset, you can enter the folder "ednet" and run 'data.py' to pre-process EdNet dataset.


## Model Code
- extract.py: Extract the implicit similarity between questions and skills.
- PNN.py: Implement the product layer.
- pebg.py: The PEBG model.
- pebg_dkt.py: The PEBG+DKT model. 



If you need more information about our experiments, you can contact us. 
email: liuyunfei@sjtu.edu.cn