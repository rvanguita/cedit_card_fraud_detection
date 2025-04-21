# Credit Card Fraud Detection


![](assets/img/wallpaper.png)


## Introduction

Currently, more than **1.47 million credit card transactions are recorded per minute** worldwide. Assuming that approximately 1% of these transactions are fraudulent, this could amount to nearly **7.76 billion fraudulent transactions per year**. If this scenario were entirely accurate and left unmitigated, the financial losses to the banking sector could reach **trillions of dollars annually**, solely due to fraud. However, this impact is significantly reduced by the adoption of effective fraud detection and prevention mechanisms. Among the most widely used approaches are **automated analysis techniques**, which enable real-time validation of each transaction. One such strategy involves the application of machine learning models, which are capable of learning complex patterns and classifying transactions as either legitimate or fraudulent based on their behavior and specific characteristics.


The problem of fraud detection poses a significant challenge in the field of machine learning, primarily because it is an extremely imbalanced classification task, where fraudulent samples account for less than 1% of all transactions. In this context, the present work proposes the use of real-world banking data for the preprocessing, training, and evaluation of machine learning models, with the goal of efficiently identifying potential financial fraud. This approach aims to address the effects of class imbalance and enhance the model's predictive performance for rare but highly critical cases.


## Objective

This work aims to implement a comprehensive analysis by applying feature engineering techniques and training a classification model for a highly imbalanced problem. The main challenge lies in the correct interpretation of performance metrics, as the imbalance often distorts the results, leading many metrics to appear artificially inflated. Additionally, the dataset underwent a dimensionality reduction process using PCA (Principal Component Analysis) in order to remove any personally identifiable information from the financial transactions, thereby preserving user privacy. At the conclusion of this analysis, a trained, validated, and production-ready model will be presented, with a strong focus on ensuring efficient and secure fraud detection.



### Repository Structure

The `main.ipynb` notebook contains the core code responsible for executing the analyses conducted on the dataset. All visual assets used in this document are located in the `assets/img/` directory.

The `src/` directory houses the Python scripts developed throughout the analytical process. These scripts implement functions and classes designed to streamline future analyses by promoting code reusability, graphical standardization, and workflow organization. Each file follows a modular structure to minimize code redundancy and maintain the visual consistency adopted across all project visualizations.

The `data/` directory contains a `.zip` archive with the original raw data, as well as six `.csv` files representing the different turbines analyzed. Lastly, the `requirements.txt` file provides a complete list of libraries and dependencies used in this project, enabling straightforward environment replication.


## [Data set](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset used in this project was obtained from the Kaggle platform, where additional information is available, including detailed descriptions of each column and the corresponding data types.

We deliberately chose not to include these descriptions directly in this `README.md` to keep the content concise and avoid overloading the document. For further details regarding the dataset structure, we recommend referring to the original project page on Kaggle.

The dataset used in this project is not made available in this repository due to its large size. It is recommended to access the data through the original source or request it directly from the author, if necessary.

The `models/` directory contains the trained models ready for deployment in production, following the structure defined by the training pipeline. The files in this folder have already undergone the full process of preprocessing, training, and validation, and are prepared for use in real-world applications.



## Methodology and Results


Como comentado anetriormente, esse dataset é de um banco real, cujo os dados sensiceis foram passados pelo metodo PCA. Dessa forma, temos uma tabela com uma coluna Time (representa em segundos, quando ouve o registro da transação), Amount (o valor monetario da transação), V1 a V28, oque não sabemos o seu significado e se aquela transação foi considerada como fraude ou nao. A primeira analise foi feita para averiguar possiveis dados faltantes e se o tipo de cada coluna estava sendo respistado. Quando foi analisar se avia linhas duplicadas, em um primeiro momento foi identificado que havia, mas uma analise mais pronfunda, demonstrou que como os addos são muito pequenos, algumas colunas davam um falso positivo. 